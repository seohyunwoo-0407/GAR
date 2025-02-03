import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import losses
from huggingface_hub import HfApi, login
import torch
import gc

class EmbedderFinetuner:
    def __init__(self):
        self.model = None
        
    @staticmethod
    def format_query(query: str) -> str:
        return f'Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {query}'

    @staticmethod
    def format_text(title: str, text: str) -> str:
        return f"Title: {title}\nText: {text}"

    def load_datasets(self, names: list, markdown_dir: str = "Markdown"):
        corpus_df = pd.DataFrame()
        queries_df = pd.DataFrame()
        relevant_docs_data = pd.DataFrame()
        all_data = pd.DataFrame()
        
        for name in names:
            print(f"Loading data for {name} ...")
            qrels_path = f'{name}_qrels.tsv'
            queries_path = os.path.join(markdown_dir, name, 'queries.jsonl')
            corpus_path = os.path.join(markdown_dir, name, 'corpus.jsonl')
            
            qrels = pd.read_csv(qrels_path, sep='\t')
            queries_temp = pd.read_json(queries_path, lines=True)
            corpus_temp = pd.read_json(corpus_path, lines=True)
            
            corpus_df = pd.concat([corpus_df, corpus_temp], ignore_index=True)
            queries_df = pd.concat([queries_df, queries_temp], ignore_index=True)
            relevant_docs_data = pd.concat([relevant_docs_data, qrels], ignore_index=True)
            
            queries_temp.rename(columns={'_id': 'query_id', 'title': 'title_queries', 'text': 'text_queries'}, inplace=True)
            corpus_temp.rename(columns={'_id': 'corpus_id', 'title': 'title_corpus', 'text': 'text_corpus'}, inplace=True)
            
            data = qrels.merge(queries_temp, on='query_id').merge(corpus_temp, on='corpus_id')
            all_data = pd.concat([all_data, data], ignore_index=True)
        
        return corpus_df, queries_df, relevant_docs_data, all_data

    def split_train_val(self, all_data: pd.DataFrame, relevant_docs_data: pd.DataFrame, 
                       test_size: float = 0.2, random_state: int = 42):
        train_rel, val_rel = train_test_split(relevant_docs_data, test_size=test_size, random_state=random_state)
        train_data = all_data[all_data['query_id'].isin(train_rel['query_id'])]
        val_data = all_data[all_data['query_id'].isin(val_rel['query_id'])]
        return train_data, val_data, train_rel, val_rel

    def create_train_samples(self, train_data: pd.DataFrame) -> list:
        train_samples = []
        for _, row in train_data.iterrows():
            sample = InputExample(
                texts=[
                    self.format_query(self.format_text(row['title_queries'], row['text_queries'])),
                    self.format_text(row['title_corpus'], row['text_corpus'])
                ]
            )
            train_samples.append(sample)
        return train_samples

    def add_ir_doc(self, df: pd.DataFrame) -> pd.DataFrame:
        irrelevant_docs = []
        for _, row in df.iterrows():
            text = row['text_corpus']
            candidates = df[df['text_corpus'] != text]
            if len(candidates) > 0:
                irrelevant = candidates.sample(n=1)
                irrelevant_docs.append(self.format_text(irrelevant.iloc[0]['title_corpus'], irrelevant.iloc[0]['text_corpus']))
            else:
                irrelevant_docs.append("")
        df = df.copy()
        df['irrelevant_docs'] = irrelevant_docs
        return df

    def create_eval_examples(self, val_data: pd.DataFrame) -> list:
        examples = []
        val_data_ir = self.add_ir_doc(val_data)
        for _, row in val_data_ir.iterrows():
            examples.append(InputExample(
                texts=[
                    self.format_query(self.format_text(row['title_queries'], row['text_queries'])),
                    self.format_text(row['title_corpus'], row['text_corpus'])
                ],
                label=1.0
            ))
            examples.append(InputExample(
                texts=[
                    self.format_query(self.format_text(row['title_queries'], row['text_queries'])),
                    row['irrelevant_docs']
                ],
                label=0.0
            ))
        return examples

    def prepare_corpus_queries(self, corpus_df: pd.DataFrame, queries_df: pd.DataFrame, 
                             val_rel: pd.DataFrame, random_sample: int = 3000):
        corpus_df['text'] = corpus_df.apply(lambda row: self.format_text(row['title'], row['text']), axis=1)
        corpus_df = corpus_df.drop(columns=['title'])
        
        queries_df['text'] = queries_df.apply(lambda row: self.format_query(self.format_text(row['title'], row['text'])), axis=1)
        queries_df = queries_df.drop(columns=['title'])
        
        required_corpus_ids = set(map(str, val_rel["corpus_id"]))
        all_ids = corpus_df["_id"].tolist()
        additional_ids = set(random.sample(all_ids, k=random_sample)) if len(all_ids) >= random_sample else set(all_ids)
        required_corpus_ids |= additional_ids
        
        corpus_df = corpus_df.loc[corpus_df["_id"].astype(str).isin(required_corpus_ids)]
        corpus_dict = dict(zip(corpus_df["_id"].astype(str), corpus_df["text"]))
        queries_dict = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
        
        return corpus_dict, queries_dict

    def build_relevant_docs(self, val_rel: pd.DataFrame) -> dict:
        relevant_docs = {}
        for qid, corpus_id in zip(val_rel["query_id"], val_rel["corpus_id"]):
            qid_str = str(qid)
            corpus_str = str(corpus_id)
            if qid_str not in relevant_docs:
                relevant_docs[qid_str] = set()
            relevant_docs[qid_str].add(corpus_str)
        return relevant_docs

    def create_ir_evaluator(self, queries: dict, corpus: dict, relevant_docs: dict, 
                          batch_size: int = 4, name: str = "Evaluate") -> InformationRetrievalEvaluator:
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=name,
            batch_size=batch_size
        )

    def train_model(self, model: SentenceTransformer, train_samples: list, evaluator: InformationRetrievalEvaluator, 
                   output_path: str, epochs: int = 2, learning_rate: float = 2e-5, 
                   warmup_ratio: float = 0.1, batch_size: int = 4):
        self.model = model
        loader = NoDuplicatesDataLoader(train_samples, batch_size=batch_size)
        loss = losses.MultipleNegativesRankingLoss(model)
        warmup_steps = int(len(loader) * epochs * warmup_ratio)
        
        model.fit(
            train_objectives=[(loader, loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True,
            use_amp=True,
            evaluation_steps=len(loader),
            save_best_model=True,
        )

    def upload_model_to_hub(self, save_path: str, repo_id: str, hf_token: str, repo_owner: str = None):
        login(token=hf_token)
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id)
        except Exception as e:
            print(f"Repo creation: {e}")
        full_repo_id = repo_id if repo_owner is None else f"{repo_owner}/{repo_id}"
        api.upload_folder(
            folder_path=save_path,
            repo_id=full_repo_id,
            repo_type="model",
        )

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
