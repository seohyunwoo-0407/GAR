# GAR 

This project is a RAG-based (Retrieval-Augmented Generation) answer generation and refinement pipeline specifically designed for question answering in the finance domain.

## Project Structure    

GAR/
├── task1/ # Hybrid Search implementation
│   └── Embedder_Finetuning.py
│   └── hybrid_search.py
│   └── run_finetuning.py
│   └── run_hybrid_search.py
│   └── main.py                  # Main pipeline execution
└── requirements.txt             # List of required packages


## Pipeline Description
Embedder_Finetuning.py: Fine-tuning retrieval model 
hybrid_search.py: Execute hybrid search, and find optimal alpha for each task 
run_finetuning.py: Run Embedder_Finetuning.py 
run_hybrid_search.py: Run hybrid_search.py

## Installation

1. Clone the repository
git clone https://github.com/seohyunwoo-0407/GAR.git
cd GAR


2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows

3. Install required packages
pip install -r requirements.txt
