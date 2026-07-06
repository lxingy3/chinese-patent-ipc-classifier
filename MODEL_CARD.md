# Model Card

## Task

Predict IPC labels from Chinese patent title and abstract text.

## Baseline

The included baseline uses character n-gram TF-IDF features and a Linear SVM classifier. It is a transparent reference point before using transformer models.

```bash
python experiments/baseline_tfidf_svm.py --label ipc_level_2
```

## Transformer summary

Chinese-XLNet was evaluated in a separate cross-validation pipeline. This repository includes metric summaries, not checkpoints.

## Limits

IPC labels are imbalanced. Level 2 is the recommended default target. Deeper labels require stronger smoothing, grouping, or multi-label handling.

