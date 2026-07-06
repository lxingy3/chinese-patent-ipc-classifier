# Chinese Patent IPC Classifier

A compact benchmark for Chinese patent IPC prediction from public patent titles and abstracts.

This project focuses on supervised classification: label coverage, IPC level selection, classical baselines, and transformer evaluation summaries. It is designed to be easy to clone, inspect, and rerun.

## What this repo covers

- Cleaned patent titles and abstracts with IPC level 1 to level 4 labels
- IPC label distribution analysis for choosing a prediction target
- A fast TF-IDF + Linear SVM baseline
- Chinese-XLNet cross-validation metric summaries
- Release validation for duplicated rows and accidental local artifacts

## Dataset

Main file:

```text
data/processed/patent_ipc_clean.csv
```

Recommended target:

```text
ipc_level_2
```

Level 1 is broad. Level 3 and level 4 are useful for analysis, but they become sparse quickly for single-label classification.

## Quick start

```bash
pip install -r requirements.txt
python scripts/validate_release.py
python experiments/baseline_tfidf_svm.py --label ipc_level_2
```

By default the baseline uses a deterministic sample so it runs quickly. Use `--max-samples 0` for the full dataset.

## Results included

```text
experiments/results/chinese_xlnet_accuracy_summary.csv
experiments/results/chinese_xlnet_fold_metrics.csv
```

Chinese-XLNet checkpoints are not included. The repo keeps metrics and code lightweight instead of storing large model files.

## License

Code is MIT licensed. The dataset is compiled from public patent metadata; see `DATA_NOTICE.md` before redistribution or commercial use.

