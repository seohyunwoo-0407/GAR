import openai
import os
from dotenv import load_dotenv
import json 
load_dotenv()
client = openai.OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)
model_name = 'gpt-4o-mini' 
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data
def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        for i in data:
            file.write(json.dumps(i, ensure_ascii=False) + '\n')
def evaluate_corpus_relevance(query_line, corpus_list):
    prompt = f"""Given the following query and 10 corpus passages, determine if each corpus is relevant for answering the query.
Respond ONLY with 10 letters (T or F) in sequence, without any additional text.
T means the corpus is relevant, F means it is not relevant.
Query: {query_line}
""" + "\n".join([f"###CORPUS_{i+1}\n{corpus}\n" for i, corpus in enumerate(corpus_list)])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    relevance = dict(response)['choices'][0].message.content
    selected_corpus = [corpus for corpus, is_relevant in zip(corpus_list, relevance) if is_relevant == 'T']
    combined_corpus = "\n".join([f"###DOCUMENT_{i+1}\n{corpus}\n" for i, corpus in enumerate(selected_corpus)])
    print(f"relevance: {relevance}")
    return combined_corpus