# Vietnamese Summarization Pre-trained Experiments

This repository contains the pre-trained-model experiments for Vietnamese abstractive text summarization. The main model direction is Qwen2.5-1.5B-Instruct fine-tuned with QLoRA under Kaggle GPU constraints.

## Included Files

- `qwen3-1.5B-Instruct.ipynb`: baseline QLoRA fine-tuning notebook using the original train and validation splits.
- `qwen_continue_training.ipynb`: continued QLoRA fine-tuning notebook. It can start from a previous LoRA adapter and train on an additional data variant.
- `benchmark.ipynb`: generation and evaluation notebook for a saved LoRA adapter. It computes ROUGE, BLEU, and BERTScore.
- `artifacts/train_entity_substitution_deepseek_100pct_improved_alias_safe.parquet`: training data containing the original train split plus entity-substitution augmented examples.
- `artifacts/train_entity_substitution_deepseek_100pct_improved_alias_safe_plus_summary_rewrite_20pct.parquet`: training data containing the original train split, entity-substitution augmentation, and additional LLM-rewrite examples.

## Experiment Summary

The baseline notebook fine-tunes Qwen2.5-1.5B-Instruct on Vietnamese article-summary pairs. The model is loaded with 4-bit NF4 quantization and adapted using LoRA adapters on attention and MLP projection layers. Prompt tokens are masked during supervised fine-tuning, so the loss is computed only on the target summary tokens.

Two data augmentation settings are included:

- Entity substitution: replaces selected entities while keeping the article-summary format consistent.
- Entity substitution plus LLM rewrite: combines entity-substituted data with additional rewritten-summary examples.

The continued-training notebook is used to train additional adapters from previous checkpoints or adapters. The benchmark notebook loads a trained adapter, generates summaries, and evaluates them against reference summaries using automatic metrics.

## Notes

The notebooks were designed for Kaggle. Paths inside the notebooks may need to be updated depending on the Kaggle dataset names or local directory layout.
