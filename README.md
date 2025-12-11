# GenAIFinalProject
Embedding Pre-training, LoRA Fine-Tuning, and RAG Benchmarking on Paul Graham Essays

# Overview
This project implements a full three-stage GenAI pipeline:

1. Part 1: Embedding Model Pre-Training
* General-purpose embedding model trained on BookCorpus
* Domain-specific embedding model trained on Paul Graham essays
* Contrastive learning with InfoNCE
* Retrieval evaluation: MRR, Precision@k, Recall@k, nDCG

2. Part 2: LoRA Fine-Tuning of Gemma-3-1B-IT
* Three Q/A datasets generated: Synthetic, Base, Combined
* LoRA-tuned decoder model for each dataset
* Benchmark scoring using ROUGE-L, BERTScore F1, Token-F1

3. Part 3: Full RAG System Benchmark
* The embedding models used as retrievers
* Gemma-3-1B-IT used as generator
* Evaluation on the instructor-provided benchmark CSVs
(customgpt_run.csv and openai_run.csv)
* Metrics computed: ROUGE-L, BERTScore, retrieval quality

This README explains how to reproduce the pipeline and interpret results.

# Setup Instructions
## Clone or open the notebook
This project is fully contained in:


```GenAI_FINAL_RECENT-5.ipynb```

## Required Python Packages
Run inside Colab:

```
pip install -q datasets transformers accelerate peft bitsandbytes sentencepiece
faiss-cpu evaluate beautifulsoup4 lxml
```

## Hardware
GPU required (I personally used A100)

# Part 1: Embedding Model Pre-Training

## Datasets
* BookCorpus (general-purpose): sampled and cleaned to ~20k documents

* Paul Graham essays (domain-specific): scraped and filtered to 228 essays

## Training Method
Trained two embedding models using contrastive InfoNCE

Models are BERT-style encoders with:

* 512 hidden size
* 6 transformer layers
* Mean-pooling encoder output
* L2-normalized output embeddings

## Outputs
Each model prints:

* Training loss per epoch
* Validation loss
* Retrieval metrics that will be seen:
  * MRR (Mean Reciprocal Rank)
  * Precision@5
  * Recall@5
  * nDCG@5

## Expected Findings
* PG-specific embeddings outperform general embeddings on PG-domain queries

* BookCorpus model performs better for general queries

# Part 2: LoRA Fine-Tuning (Gemma-3-1B-IT)

## Datasets Used
* Synthetic:	Auto-generated using Gemma and prompting
* Base:	Manually created, cleaned Q/A pairs
* Combined:	Synthetic and Base

## Training Method
Each dataset is trained separately using:
* LoRA rank = 8
* LoRA α = 16
* Dropout = 0.1
* 3–5 epochs depending on the dataset size

Each fine-tuned model is saved under:
```
/content/gemma3_lora_synthetic
/content/gemma3_lora_base
/content/gemma3_lora_both
```

## Evaluation Metrics
These are the metrics that will be available for each condition:
* ROUGE-L
* BERTScore F1
* Token-F1

## Typical Results Interpretation
* Synthetic: High fluency, weaker accuracy
* Base: Most faithful, best correctness
* Both: Balanced performance (typically the best, but the results will tell)

# Part 3: RAG Benchmark

## Inputs
Upload the following instructor benchmark CSVs:
* customgpt_run.csv  (benchmark questions)
* openai_run.csv     (OpenAI GPT-4.1-mini outputs for comparison)

## RAG Pipeline
1. Retrieve the top-k PG chunks using each embedding model:
* General embeddings
* PG-specific embeddings
2. Pass the retrieved context and question into Gemma-3-1B-IT
3. Score the generated answers against the reference answers using:
* ROUGE-L
* BERTScore F1

## Retrieval Metrics
These are the metrics that will be computed for each encoder:
* MRR: How high the true chunk appears
* P@k: Does the correct chunk appear in the top-k?
* R@k: Proportion of relevant chunks recovered
* nDCG: Rank-weighted relevance

## Expected Conclusion
* PG-specific encoder: Higher MRR, P@5, Recall@5
* Better retrieval = Higher downstream RAG answer quality
* RAG model may outperform OpenAI baseline on domain-specific tasks

# Project Folder Structure
.

├── GenAI_FINAL_RECENT-5.ipynb

├── customgpt_run.csv

├── openai_run.csv

├── /content/gemma3_lora_synthetic/

├── /content/gemma3_lora_base/

├── /content/gemma3_lora_both/

└── README.md

# How to Run Everything

1. Hit "run all"
2. There should be a point in Part 2 where it stops to ask for the HuggingFace token to access Gemma3
3. You need to make sure you have the two benchmark CSVs in your folder structure (for Part 3)
