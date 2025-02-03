import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import dotenv
import os

dotenv.load_dotenv()

def save_from_csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            json_obj = {
                "input": {
                    "messages": [
                        {"role": "user", "content": row['query']}
                    ],
                    "tools": [],
                    "parallel_toolcalls": True
                },
                "preferred_output": [
                    {"role": "assistant", "content": row['response1']}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": row['response2']}
                ]
            }

            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


path = "/content/drive/MyDrive/FinanceRAG-GAR/본선/data_files/dpo_data_to67.csv"
saving_path = "/content/drive/MyDrive/FinanceRAG-GAR/본선/data_files/dpo_data_to67.jsonl"
save_from_csv_to_jsonl(path, saving_path)

"""
query | response1 | response2 |
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "Hello, can you tell me how cold San Francisco is today?"
      }
    ],
    "tools": [],
    "parallel_tool_calls": true
  },
  "preferred_output": [
    {
      "role": "assistant",
      "content": "Today in San Francisco, it is not quite cold as expected. Morning clouds will give away to sunshine, with a high near 68°F (20°C) and a low around 57°F (14°C)."
    }
  ],
  "non_preferred_output": [
    {
      "role": "assistant",
      "content": "It is not particularly cold in San Francisco today."
    }
  ]
}
...
"""


model_name = 'gpt-4o-mini' # 'gpt-4o'

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    return data


training_data = load_jsonl(saving_path)


client = openai.OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

response = client.fine_tuning.jobs.create(
    training_file=training_data,
    model=model_name,
    method={
        "type": "dpo",
        "dpo": {
            "hyperparameters": {"beta": 0.1}
        }
    }
)

response
