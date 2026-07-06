# Chinese Patent IPC Classifier

Chinese patent IPC classification experiments built from public patent titles and abstracts.

This repository is a released experiment bundle, not just a toy baseline. It includes the cleaned dataset, model comparison tables, out-of-fold prediction caches, a trained IPCPrediction checkpoint set, and optional training scripts for TF-IDF/SVM, a ratio-attention neural model, and Chinese-XLNet.

## Contents

```text
data/processed/                         Cleaned patent text and IPC labels
experiments/                            Runnable experiment entrypoints
experiments/results/model_comparison/   Model comparison tables and prediction caches
checkpoints/ipcprediction_retrain_3ep/  Released 5-fold PyTorch checkpoints
```

## Released Experiments

The main comparison covers:

- IPCPrediction ratio-attention model: 3, 8, 8-title-weighted, and 16 epoch runs
- TF-IDF + Linear SVM: multiple `C`, n-gram, and title-weight settings
- Chinese-XLNet: five-fold OOF metric summaries from a GPU run

The best classical model in this release is:

```text
TF-IDF + Linear SVM
C=4, character n-gram 1-4, title_weight=10
IPC level 3 OOF accuracy: 0.4656
```

Chinese-XLNet reached IPC level 3 OOF accuracy around `0.4875` in the released summary. Full transformer training is GPU-heavy, so the repository publishes the metric tables and optional training script instead of asking every reader to rerun the whole pipeline.

## Quick Start

Install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

Validate the release:

```bash
python scripts/validate_release.py
```

Run a small TF-IDF baseline:

```bash
python experiments/baseline_tfidf_svm.py --label ipc_level_2
```

Inspect the released model comparison:

```text
experiments/results/model_comparison/ipcprediction_summary.csv
experiments/results/model_comparison/ipcprediction_fold_metrics.csv
experiments/results/model_comparison/prediction_caches/
experiments/results/best_tfidf_svm_level3.json
```

## Optional Training

These scripts are included for transparency. They are not required to use the released results.

```bash
python experiments/tfidf_svm_cv.py --max-samples 2000
```

For PyTorch experiments:

```bash
pip install -r requirements-torch.txt
python experiments/ratio_attention_ipc.py --max-samples 2000 --epochs 3
```

For Chinese-XLNet:

```bash
pip install -r requirements-xlnet.txt
python experiments/chinese_xlnet_oof.py --max-samples 1000 --epochs 1
```

Running the full Chinese-XLNet five-fold setup is intended for a GPU environment.

## Data Notice

The dataset contains public patent metadata: titles, abstracts, IPC labels, and derived experiment outputs. See `DATA_NOTICE.md` before redistribution or commercial use.

