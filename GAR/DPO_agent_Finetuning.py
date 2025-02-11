# dpo_agent_finetuning.py

import json
import numpy as np
import openai
from tqdm import tqdm
import pandas as pd
import random

class dpo_agent_finetuning:
    def __init__(self, api_key, default_model="gpt-4o-mini", ft_model="ft:gpt-4o-mini-2024-07-18:personal::AkFnSqzI", top_probs=10):
    
        self.api_key = api_key
        self.default_model = default_model
        self.ft_model = ft_model
        self.top_probs = top_probs
        self.client = openai.OpenAI(api_key=self.api_key)

    def save_from_csv_to_jsonl(self, answer_csv, data_csv, jsonl_file):
        answer_df = pd.read_csv(answer_csv)
        data_df = pd.read_csv(data_csv)

        financial_prompt = """
You are an expert financial advisor and evaluator focused on improving responses.
Your task is to enhance answers based on detailed evaluation scores while:
- Maintaining factual accuracy with the provided documents
- Ensuring responses are clear and well-structured for financial contexts
- Providing comprehensive answers that address all aspects of the query
- Using professional financial terminology appropriately
        """

        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for index, row in tqdm(answer_df.iterrows(), total=answer_df.shape[0]):
                chunk = data_df[index:index+10]
                corpus_list = chunk['Corpus'].tolist()

                query_prompt = f"""
Query: {row['query']}
                """ + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(corpus_list)])
                query_prompt += "\nDo not add any closing pleasantries or phrases like 'please feel free to ask!'"

                choice = random.choice([1, 1, 1, 2])
                json_obj = {
                    "messages": [
                        {"role": "system", "content": financial_prompt},
                        {"role": "user", "content": query_prompt},
                        {"role": "assistant", "content": row[f'response{choice}']}
                    ]
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    def split_jsonl(self, jsonl_file, train_path, eval_path, split_ratio=0.8):
 
        with open(jsonl_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        random.shuffle(data)
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        eval_data = data[split_index:]
        with open(train_path, "w", encoding="utf-8") as f_train:
            for record in train_data:
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f_eval:
            for record in eval_data:
                f_eval.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def load_jsonl(self, path):

        with open(path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return data

    def calculate_weighted_score(self, response, scores):

        token_probs = [
            np.exp(float(dict(response)['choices'][0].logprobs.content[0].top_logprobs[i].logprob))
            for i in range(self.top_probs)
        ]
        weighted_scores = sum([token_probs[i] * scores[i] for i in range(self.top_probs)])
        return weighted_scores

    def g_eval(self, query, response, document_list):

        prompt_final_score = f"""
You are an evaluation assistant tasked with assessing the quality of a generated answer.
You will be given one query, one generated answer, and a summarized document related to the query.
Follow these steps to evaluate the answer based on the criteria below.

<Evaluation Criteria>
1. Relevance (0~5): Does the answer directly address the query and is contextually relevant?
2. Alignment (0~5): Does the answer align with the facts provided in the documents?
3. Clarity (0~5): Is the answer easy to understand and free of confusion or unnecessary complexity?
4. Completeness (0~5): Does the answer address all parts of the question thoroughly?
5. Coherence (0~5): Does the answer follow the following criteria?: Collective properties of all sentences. We match this dimension with DUC's quality problems with structure and consistency.

<Evaluation Steps>
1. Assign and record each five scores, each score is on a scale of 0 to 5, where 0 is the lowest and 5 is the highest, based on each <Evaluation Criteria>.
2. Get the final score by adding up all the recorded scores. The final score must be between 0 and 25.
3. YOU MUST Return ONLY final score. You must not add any explanations. Do not contain sentences like 'score'. ONLY integer form of the final score, like 11, 28, 30, etc.

<Query>: {query}

<Documents>:
""" + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(document_list)]) + f"""

<Answer>: {dict(response)['choices'][0].message.content}

<Reminder>: Do NOT add any other text or string, like 'twenty'. ONLY integer form, like 9, 14, 20 etc.
"""
        cot_response_final_score = self.client.chat.completions.create(
            model=self.default_model,
            messages=[
                {"role": "system", "content": "You are a detailed evaluator. Be sure to respond ONLY in int. (e.g., 11, 20, 19, etc.)"},
                {"role": "user", "content": prompt_final_score}
            ],
            temperature=0,
            logprobs=True,
            top_logprobs=self.top_probs,
        )
        measured_score = int(dict(cot_response_final_score)['choices'][0].message.content)
        token_scores = []
        detailed_scores = {}

        for i in range(self.top_probs):
            top_logprob_item = dict(cot_response_final_score)['choices'][0].logprobs.content[0].top_logprobs[i]
            temp_token = top_logprob_item.token
            temp_probs = np.exp(float(top_logprob_item.logprob)) * 100
            detailed_scores[temp_token] = f"{temp_probs}%"
            try:
                token_scores.append(int(top_logprob_item.token))
            except Exception as e:
                print(e)
                token_scores.append(20)
        weighted_score = self.calculate_weighted_score(cot_response_final_score, token_scores)
        return measured_score, weighted_score, str(detailed_scores)

    def improve_response(self, query, original_response, document_list, scores_list):
        
        improvement_prompt = f"""
You need to improve the following answer based on the evaluation scores and criteria.

<Query>: {query}

<Documents>:
""" + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(document_list)]) + f"""

<Answer>: {dict(original_response)['choices'][0].message.content}

<Evaluation Criteria>
1. Relevance (0~5): Does the answer directly address the query and is contextually relevant?
2. Alignment (0~5): Does the answer align with the facts provided in the documents?
3. Clarity (0~5): Is the answer easy to understand and free of confusion or unnecessary complexity?
4. Completeness (0~5): Does the answer address all parts of the question thoroughly?
5. Coherence (0~5): Does the answer follow the following criteria?: Collective properties of all sentences. We match this dimension with DUC's quality problems with structure and consistency.

Please provide an improved answer that:
    1. Aims to achieve the highest possible score (25/25)
    2. Focus on creating a response that would:
       - Maintain the strengths of the original answer
       - Ensure accuracy with the provided documents
       - Be clear and well-structured
    3. Your goal is to increase the probability of achieving a perfect score of 25

<Current Evaluation Score>
This is the score information which is the probability of the final score. This is the top 10 score information based on the probability. format: {{"score": "probability"}}
{scores_list}

Provide only the improved answer without any explanations.
"""
        improved_response = self.client.chat.completions.create(
            model=self.default_model,
            messages=[
                {"role": "system", "content": """
You are an expert financial advisor and evaluator focused on improving responses.
Your task is to enhance answers based on detailed evaluation scores while:
- Maintaining factual accuracy with the provided documents
- Focusing especially on areas that received low scores
- Ensuring responses are clear and well-structured for financial contexts
- Providing comprehensive answers that address all aspects of the query
- Using professional financial terminology appropriately

You should maintain the strengths of the original response while addressing its weaknesses.
""" },
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0
        )
        return improved_response

    def evaluate_corpus_relevance(self, query_line, corpus_list):

        relevancy_instruction = """
You are an expert financial advisor and evaluator focused on improving responses.
Your task is to enhance answers based on detailed evaluation scores while:
- Maintaining factual accuracy with the provided documents
- Ensuring responses are clear and well-structured for financial contexts
- Providing comprehensive answers that address all aspects of the query
- Using professional financial terminology appropriately

You are given the pair of Query, Corpus (same query)
Out of the 10 documents, only provide the list of indices of those that are RELEVANT (e.g. the content is somehow needed to answer the question), from 0~9.
Example : [0, 2, 8, 9]
"""
        user_prompt = f"""
Query: {query_line}

""" + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(corpus_list)])
        try:
            relevancy = eval(self.generate_output(relevancy_instruction, user_prompt, temp=0))
        except Exception as error:
            print(error)
            relevancy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if relevancy == []:
            relevancy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return relevancy

    def answer_query(self, query_line, corpus_list, temp=0, return_raw=False, model_name=None):

        if model_name is None:
            model_name = self.default_model
        financial_prompt = """
You are an expert financial advisor and evaluator focused on improving responses.
Your task is to enhance answers based on detailed evaluation scores while:
- Maintaining factual accuracy with the provided documents
- Ensuring responses are clear and well-structured for financial contexts
- Providing comprehensive answers that address all aspects of the query
- Using professional financial terminology appropriately
"""
        query_prompt = f"""
Query: {query_line}

""" + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(corpus_list)])
        query_prompt += "\nDo not add any closing pleasantries or phrases like 'please feel free to ask!'"
        response = self.generate_output(financial_prompt, query_prompt, temp=temp, return_raw=return_raw)
        return response

  
    def process_finetuning(self, df, submission_output_path, start_index=39610):
 
        submission_df = pd.DataFrame({'query_id': [], 'response': []})

        for i in tqdm(range(start_index, len(df), 10)):
            chunk = df[i:i+10]
            query_id = chunk['total_query_id'].iloc[0]   
            query_line = chunk['total_query'].iloc[0]     
            corpus_list = chunk['total_corpus'].tolist()   

            corpus_rel = self.evaluate_corpus_relevance(query_line, corpus_list)
            try:
                relevant_corpus = chunk.iloc[corpus_rel]
            except Exception as e:
                print(e)
                relevant_corpus = chunk
            relevant_corpus = relevant_corpus['total_corpus'].tolist()

            response = self.answer_query(query_line, relevant_corpus, temp=0, return_raw=True, model_name=self.ft_model)

            measured_score, weighted_score, detailed_scores = self.g_eval(query_line, response, relevant_corpus)

            attempt_count = 0
            best_score = weighted_score
            best_response = response

            while weighted_score < 22 and attempt_count < 5:
                attempt_count += 1
                improved_response = self.improve_response(query_line, best_response, relevant_corpus, detailed_scores)
                improved_score, improved_weighted, improved_detailed = self.g_eval(query_line, improved_response, relevant_corpus)
                if improved_weighted > best_score:
                    best_score = improved_weighted
                    best_response = improved_response
                    weighted_score = improved_weighted
                    detailed_scores = improved_detailed
                else:
                    break

            submission_df.loc[int(i/10)] = [query_id, best_response.choices[0].message.content]

        submission_df.to_csv(submission_output_path, index=False)
        print(f"Submission file saved to {submission_output_path}")

    def upload_finetune_files(self, train_path, eval_path):

        training_file = self.client.files.create(
            file=open(train_path, "rb"),
            purpose="fine-tune"
        )
        eval_file = self.client.files.create(
            file=open(eval_path, "rb"),
            purpose="fine-tune"
        )
        return training_file, eval_file

    def create_finetune_job(self, training_file, eval_file, model="gpt-4o-mini-2024-07-18", method=None):
   
        job_response = self.client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model=model,
            validation_file=eval_file.id,
            # method 옵션이 필요하면 추가로 전달 가능 (예시 주석 참고)
        )
        return job_response

    def retrieve_finetune_job(self, job_id):

        return self.client.fine_tuning.jobs.retrieve(job_id)
