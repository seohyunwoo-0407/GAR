import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from rank_bm25 import BM25Okapi
from huggingface_hub import login


from sentence_transformers import SentenceTransformer, CrossEncoder
from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.tasks import FinDER, FinQABench, FinanceBench, TATQA, FinQA, ConvFinQA, MultiHiertt
from financerag.common import Retrieval

logging.basicConfig(level=logging.INFO)



def tokenize_list(input_list):
    from nltk.tokenize import word_tokenize
    return list(map(word_tokenize, input_list))


class BM25Retriever(Retrieval):
    def __init__(self, model, tokenizer=tokenize_list):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def retrieve(self, corpus, queries, top_k=None, score_function=None, return_sorted=False, **kwargs):
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_lower_tokens = self.tokenizer([queries[qid].lower() for qid in queries])
        corpus_ids = list(corpus.keys())

        for qid, query in tqdm(zip(query_ids, query_lower_tokens), total=len(query_ids), desc="BM25 Retrieval"):
            scores = self.model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]
        return self.results


class HybridSearcher:
    def __init__(self):
        self.retrieval_results = {}
    
    @staticmethod
    def softmax_normalize(retrieval_results):
        for query_id, results in retrieval_results.items():
            scores = torch.tensor(list(results.values()), dtype=torch.float32)
            probs = softmax(scores, dim=0).tolist()
            doc_ids = list(results.keys())
            for i, doc_id in enumerate(doc_ids):
                results[doc_id] = probs[i]
        return retrieval_results
    
    def setup_task(self):
        finder_task=FinDER()
        finqabench_task=FinQABench()
        finqa_task=FinQA()
        financebench_task=FinanceBench()
        tatqa_task=TATQA()
        convfinqa_task=ConvFinQA()
        multihiertt_task=MultiHiertt()
        tasks=[finder_task, finqabench_task, finqa_task, financebench_task, tatqa_task, convfinqa_task, multihiertt_task]
        names=["Finder", "FinQABench", "FinQA", "FinanceBench", "TATQA", "ConvFinQA", "MultiHiertt"]
        
        for i, task in enumerate(tasks):
            task.metadata.dataset['path']='thomaskim1130/FinanceRAG-Lingua'
            task.metadata.dataset['qrels']= f'markdown-{names[i]}'
            task.load_data()
        return tasks, names
    
    def retrieval_model_setup(self):
        login(token=os.getenv('HF_TOKEN'))
        stella=SentenceTransformer(
            model_name_or_path='thomaskim1130/stella_en_1.5B_v5-FinanceRAG-TAT-MH-v2',
            trust_remote_code=True,
            query_prompt='Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ',
            doc_prompt='',
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )
        retrieval_stella=DenseRetrieval(
            model=stella,
        )

        return retrieval_stella

    def reranker_model_setup(self):
        reranker=CrossEncoderReranker(
            model=CrossEncoder('BAAI/bge-reranker-v2-e3',
                                trust_remote_code=True,
                                config_args={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map":"auto", "offload_folder":"offload"}
                                ),
        )
        return reranker

    def get_sparse_score(self, task):
        document_list = task.corpus  
        query_list = task.queries    
        tokenized_corpus = tokenize_list([doc["title"].lower() + ' ' + doc["text"].lower() 
                                        for doc in document_list.values()])
        bm25 = BM25Okapi(tokenized_corpus)
        retriever = BM25Retriever(bm25)
        top_k = len(document_list)
        sparse_retrieval_results = retriever.retrieve(corpus=document_list, queries=query_list, top_k=top_k)
        return sparse_retrieval_results

    def get_dense_score(self, task, retrieval_model):
        document_list = task.corpus
        query_list = task.queries
        dense_retrieval_results = retrieval_model.retrieve(
            corpus=document_list,
            queries=query_list,
            top_k=100
        )
        return dense_retrieval_results

    def get_hybrid_score(self, task, alpha, retrieval_model):
        sparse_alpha = 1 - alpha
        hybrid_retrieval_result = {}
        dense_retrieval_result = self.get_dense_score(task, retrieval_model)
        dense_retrieval_result = self.softmax_normalize(dense_retrieval_result)

        for query_id, results in dense_retrieval_result.items():
            for doc_id, dense_score_val in results.items():
                if query_id not in hybrid_retrieval_result:
                    hybrid_retrieval_result[query_id] = {}
                hybrid_retrieval_result[query_id][doc_id] = alpha * dense_score_val

        sparse_retrieval_result = self.get_sparse_score(task)
        sparse_retrieval_result = self.softmax_normalize(sparse_retrieval_result)

        for query_id, results in sparse_retrieval_result.items():
            for doc_id, sparse_score_val in results.items():
                if doc_id in hybrid_retrieval_result.get(query_id, {}):
                    hybrid_retrieval_result[query_id][doc_id] += sparse_alpha * sparse_score_val

        return hybrid_retrieval_result

    def evaluate_hybrid(self, task, qrels_dict, hybrid_retrieval_result):
        result = task.evaluate(qrels_dict, hybrid_retrieval_result, [10])
        return result[0]['NDCG@10']

    def tune_alpha(self, task, retrieval_model):
        alpha_values = np.linspace(0, 1, 41)
        ndcg_values = []
        qrels_path = f'Dataset/{task}_qrels.tsv' 
        qrels_df = pd.read_csv(qrels_path, sep='\t')
        qrels_dict = qrels_df.groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()

        for alpha in alpha_values:
            hybrid_retrieval_result = self.get_hybrid_score(task, alpha=alpha, retrieval_model=retrieval_model)
            ndcg_value = self.evaluate_hybrid(task, qrels_dict, hybrid_retrieval_result)
            ndcg_values.append(ndcg_value)
            max_ndcg_index = np.argmax(ndcg_values)
            optimal_alpha = alpha_values[max_ndcg_index]
        return optimal_alpha
    
    def get_reranker_score(self, task, hybrid_retrieval_result, reranker_model):
        reranker_result = task.rerank(
            reranker=reranker_model,
            results=hybrid_retrieval_result,
            top_k=10,
            batch_size=32
        )
        return reranker_result

    def get_ndcg_score(self, task, name):
        qrels = pd.read_csv(f'Dataset/{name}_qrels.tsv', sep='\t').groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()
        return task.evaluate(qrels, task.retrieve_results, [10])[0]['NDCG@10']

    def get_final_ndcg(self, tasks, names):
        result = 0
        task_lengths = []

        for n, task in enumerate(tasks):
            task_lengths.append(len(task.queries))
            print(f"{names[n]} : {len(task.queries)} Queries")
            result += self.get_ndcg_score(task, names[n])*task_lengths[-1]

        result /= sum(task_lengths)
        return result

    @staticmethod
    def merge_csv_results(output_dir):
        import glob
        csv_files = glob.glob(os.path.join(output_dir, "*", "*.csv"))
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_csv_path = os.path.join(output_dir, 'merged_output.csv')
        merged_df.to_csv(merged_csv_path, index=False)
        logging.info(f"Merged CSV saved to {merged_csv_path}")
