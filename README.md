# Chinese Patent IPC Classifier

Chinese patent IPC classification experiments built from public patent titles and abstracts.

This repository releases a cleaned patent IPC dataset, model comparison tables, out-of-fold prediction caches, a trained IPCPrediction checkpoint set, and optional training scripts for TF-IDF/SVM, a ratio-attention neural model, and Chinese-XLNet.

## What This Project Shows

The project covers the full supervised patent-classification workflow:

- building a cleaned IPC-labeled dataset from public patent metadata
- parsing IPC hierarchy levels and checking label sparsity before training
- comparing a strong sparse-text baseline, a hierarchy-aware neural model, and a pretrained Chinese transformer
- releasing prediction caches and checkpoint artifacts instead of only reporting final numbers
- documenting the practical limits of title-and-abstract classification

The full methodology and result discussion are in [`docs/technical_report.md`](docs/technical_report.md).
Large downloadable artifacts are described in [`docs/release_assets.md`](docs/release_assets.md).

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

## Method Summary

The raw collection contained 10,000 public patent records. After removing rows without IPC codes and deduplicating by `title + abstract`, the supervised table contains 9,867 samples. The project uses the first IPC code as the main single-label target and evaluates IPC level 1, level 2, and level 3.

The best lightweight model is character TF-IDF plus Linear SVM. It works well because Chinese patent titles and abstracts contain repeated technical phrases that character n-grams can capture without a tokenizer. Chinese-XLNet gives the best released accuracy, but the gain is moderate compared with its GPU cost.

Level-4 IPC labels are kept in the data, but they are too sparse for a stable model here: the released distribution table contains 4,485 level-4 classes, and the median class has only one sample.

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
