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

The `transformer/` folder contains a from-scratch Transformer-style sequence-to-sequence summarization system. It is configured through `transformer/config.py`, where the dataset paths, SentencePiece vocabulary size, model size, training hyperparameters, checkpoint paths, and beam-search settings are defined.

Main components:

- `data/tokenizer.py`: trains or loads a SentencePiece unigram tokenizer from the training split.
- `data/dataset.py` and `data/dataloader.py`: build PyTorch datasets and padded batches for source-target summarization pairs.
- `modeling/models/seq2seq.py`: implements the encoder-decoder summarization model with shared token embeddings, RMSNorm, stacked encoder/decoder blocks, and beam-search decoding.
- `training/train.py` and `training/trainer.py`: train the model with AdamW, cross-entropy loss, warmup scheduling, checkpoint saving, and validation tracking.
- `evaluation/evaluate.py`: generates summaries on the test split and reports ROUGE, BLEU, and BERTScore.

Typical commands:

```bash
cd transformer
bash scripts/train.sh
bash scripts/eval.sh
```

Before running, update `config.py` with the train/dev/test CSV paths and the SentencePiece model path.

## Notes

Notebook paths may need to be adjusted when running on Kaggle or a local machine.
