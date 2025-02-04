# main.py
import os
import pandas as pd
from Selection_agent import SelectionAgent
from DPO_agent import DPO_Agent
from DPO_agent_finetuning import dpo_agent_finetuning
from Finetuned_DPO_agent import Finetuned_DPO_agent
from G_Eval_agent import g_eval_agent
from openai import OpenAI

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    # 데이터 경로 설정
    dataset_path = 'data/raw/dataset.csv'
    selected_docs_path = 'data/processed/selected_docs.csv'
    dpo_responses_path = 'data/processed/dpo_responses.csv'
    final_answers_path = 'data/processed/final_answers.csv'
    
    # 1. Selection Agent
    selection_agent = SelectionAgent(API_KEY)
    df = pd.read_csv(dataset_path)
    selection_agent.process_data(df, selected_docs_path)
    
    # 2. DPO Agent
    dpo_agent = DPO_Agent(API_KEY)
    selected_df = pd.read_csv(selected_docs_path)
    dpo_agent.process_data(selected_df, dpo_responses_path)
    
    # 3. DPO Finetuning
    finetuning_agent = dpo_agent_finetuning(API_KEY)
    train_path = 'data/models/fine_tuned/train.jsonl'
    eval_path = 'data/models/fine_tuned/eval.jsonl'
    finetuning_agent.save_from_csv_to_jsonl(dpo_responses_path, selected_docs_path, train_path)
    finetuning_agent.split_jsonl(train_path, train_path, eval_path)
    
    # 4. Finetuned DPO Agent
    finetuned_agent = Finetuned_DPO_agent(API_KEY)
    finetuned_agent.process_finetuning(df, final_answers_path)
    
    # 5. G-Eval Agent
    evaluator = g_eval_agent(API_KEY)
    evaluator.process_pipeline(df, final_answers_path, final_answers_path)

if __name__ == '__main__':
    main()

