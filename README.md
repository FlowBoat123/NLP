# Vietnamese Abstractive Text Summarization

This repository contains two parts of the course project:

- `transformer/`: Transformer-based summarization implementation, imported from `fiddsa/text_summarization_v2`.
- `pretrain/`: pre-trained-model experiments with Qwen2.5-1.5B-Instruct and QLoRA.

## Pre-trained Experiments

The `pretrain/` folder includes:

- `qwen3-1.5B-Instruct.ipynb`: baseline QLoRA fine-tuning notebook on the original Vietnamese article-summary data.
- `qwen_continue_training.ipynb`: continued QLoRA fine-tuning notebook for training from an existing LoRA adapter.
- `benchmark.ipynb`: generation and evaluation notebook for ROUGE, BLEU, and BERTScore.
- `artifacts/train_entity_substitution_deepseek_100pct_improved_alias_safe.parquet`: original training data plus entity-substitution augmentation.
- `artifacts/train_entity_substitution_deepseek_100pct_improved_alias_safe_plus_summary_rewrite_20pct.parquet`: original training data plus entity-substitution augmentation and LLM-rewrite examples.

## Transformer Experiments

The `transformer/` folder contains the Transformer-based implementation used as the second modeling direction in the report.

## Notes

Notebook paths may need to be adjusted when running on Kaggle or a local machine.
