import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from Embedder_Finetuning import EmbedderFinetuner

def run_finetuning():
    load_dotenv()
    

    names = ['FinDER', 'FinQABench', 'FinanceBench', 'FinQA', 'TATQA', 'ConvFinQA', 'MultiHeirtt']
    markdown_dir = "Markdown"        
    output_path = "./fine-tune-0203"   
    hf_token = os.getenv("HUGGINGFACE_API_KEY")  
    repo_id = os.getenv("repo_id")
    repo_owner = os.getenv("repo_owner")
    epochs = 2
    learning_rate = 2e-5
    batch_size = 4
    warmup_ratio = 0.1


    finetuner = EmbedderFinetuner()
    corpus_df, queries_df, relevant_docs_data, all_data = finetuner.load_datasets(names, markdown_dir)
    train_data, val_data, train_rel, val_rel = finetuner.split_train_val(all_data, relevant_docs_data)


    train_samples = finetuner.create_train_samples(train_data)
    print(f"Training samples: {len(train_samples)}")
    eval_examples = finetuner.create_eval_examples(val_data)
    eval_loader = DataLoader(eval_examples, shuffle=False, batch_size=batch_size)
    

    corpus_dict, queries_dict = finetuner.prepare_corpus_queries(corpus_df, queries_df, val_rel)
    relevant_docs = finetuner.build_relevant_docs(val_rel)
    ir_evaluator = finetuner.create_ir_evaluator(queries_dict, corpus_dict, relevant_docs, batch_size=batch_size)
 

    model = SentenceTransformer(
        'NovaSearch/stella_en_1.5B_v5',
        trust_remote_code=True,
        config_kwargs={"use_memory_efficient_attention": True, "unpad_inputs": False}
    )

  
    print("Evaluating before fine-tuning:")
    ir_evaluator(model)
    

    finetuner.train_model(model, train_samples, ir_evaluator, output_path, epochs, learning_rate, warmup_ratio, batch_size)
    

    finetuner.upload_model_to_hub(output_path, repo_id, hf_token, repo_owner)


    finetuner.clear_gpu_memory()

if __name__ == "__main__":
    run_finetuning() 