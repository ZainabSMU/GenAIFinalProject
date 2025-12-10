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

## Required Python packages

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
* Retrieval metrics:
  * MRR (Mean Reciprocal Rank)
  * Precision@5
  * Recall@5
  * nDCG@5

## Expected Findings

* PG-specific embeddings outperform general embeddings on PG-domain queries

* BookCorpus model performs better for general queries
