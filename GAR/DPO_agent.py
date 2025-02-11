from tqdm import tqdm
from openai import OpenAI
import os
import json
import numpy as np
import openai
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
class DPOTrainer:
    def __init__(self, model_name):
        load_dotenv()
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate_output(self, system_prompt, user_prompt, temp=0, returnRaw=False):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp
        )
        text = response.choices[0].message.content
        return response if returnRaw else text
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
    def answer_query(self, query_line, corpus_list, temp=0, returnRaw=False):
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
        response = self.generate_output(financial_prompt, query_prompt, temp=temp, returnRaw=returnRaw)
        return response
    def process_data(self, input_csv_path, output_csv_path):
        df = pd.read_csv(input_csv_path)
        dpo_data = pd.DataFrame({'query': [], 'response1': [], 'response2': []})
        for i in tqdm(range(0, len(df), 10)):
            chunk = df[i:i+10]
            query_line = chunk['Query'].iloc[0]
            corpus_list = chunk['Corpus'].tolist()
            corpus_rel = self.evaluate_corpus_relevance(query_line, corpus_list)
            try:
                relevant_corpus = chunk.iloc[corpus_rel]
            except:
                relevant_corpus = chunk
            relevant_corpus = relevant_corpus['Corpus'].tolist()
            response1 = self.answer_query(query_line, relevant_corpus, temp=0)
            response2 = self.answer_query(query_line, relevant_corpus, temp=0.5)
            dpo_data.loc[i] = [query_line, response1, response2]
        dpo_data.to_csv(output_csv_path)
        return dpo_data
if __name__ == "__main__":
    model_name = "gpt-4o-mini"  # or "ft:gpt-4o-mini-2024-07-18:personal::AkFnSqzI"
    trainer = DPOTrainer(model_name)
    
    input_path = '/content/drive/MyDrive/FinanceRAG-GAR/본선/data_files/sampled_data.csv'
    output_path = '/content/drive/MyDrive/FinanceRAG-GAR/본선/data_files/sampled_answer_4o.csv'
    
    trainer.process_data(input_path, output_path)