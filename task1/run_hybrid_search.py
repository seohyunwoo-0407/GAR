import os
import json
import logging
import nltk
from financerag.tasks import FinDER, FinQABench, FinanceBench, TATQA, FinQA, ConvFinQA, MultiHiertt
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
from hybrid_search import HybridSearcher

logging.basicConfig(level=logging.INFO)

nltk.download('punkt')
hybrid_searcher = HybridSearcher()

tasks, names = hybrid_searcher.setup_task()

retrieval_model = hybrid_searcher.retrieval_model_setup()
reranker_model = hybrid_searcher.reranker_model_setup()

output_dir = "C:/Users/shw41/GAR/data/task1"

for task, name in zip(tasks, names):
    optimal_alpha = hybrid_searcher.tune_alpha(task, retrieval_model)
    hybrid_retrieval_results = hybrid_searcher.get_hybrid_score(task, optimal_alpha, retrieval_model)
    reranked_results = hybrid_searcher.get_reranker_score(task, hybrid_retrieval_results, reranker_model)
    ndcg_score = hybrid_searcher.get_final_ndcg(tasks, names)

    print(f"Task: {name}, NDCG Score: {ndcg_score}")
    print(f"Optimal Alpha: {optimal_alpha}")
    task.save_results(output_dir, name, reranked_results)

hybrid_searcher.merge_csv_results(output_dir)
