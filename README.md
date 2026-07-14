# Chinese Patent IPC Classifier

[![CI](https://github.com/lxingy3/chinese-patent-ipc-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/lxingy3/chinese-patent-ipc-classifier/actions/workflows/ci.yml)

A hierarchical multilabel IPC benchmark for Chinese patent titles and abstracts.

The dataset has 9,867 deduplicated documents and 32,831 IPC code assignments. The benchmark keeps every subclass assigned to a document instead of reducing the task to its first IPC code. It includes five-fold out-of-fold predictions, per-label scores, and the threshold search from each fold.

## Results

The reference model is character TF-IDF with one-vs-rest linear SVMs. All values below come from one five-fold out-of-fold run.

| Evaluation | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Benchmark subclasses | 0.4563 | 0.5380 | 0.4938 |
| All true subclasses | 0.4563 | 0.5307 | 0.4907 |
| IPC classes | 0.5562 | 0.6093 | 0.5815 |
| IPC sections | 0.7083 | 0.7487 | 0.7279 |
| Expanded hierarchy | 0.5575 | 0.6191 | 0.5867 |

The benchmark subclass set contains 342 labels with at least five documents. It covers 97.82% of the subclass assignments and 98.71% of the documents. The five subclass micro-F1 scores range from 0.4885 to 0.5070.

The long tail is still difficult. Subclass macro-F1 is 0.2978, and 92 of the 342 benchmark labels have zero out-of-fold F1. The repository publishes those failures in `label_metrics.csv` rather than reporting only the aggregate score.

## Run it

Install the Python dependencies, validate the checked-in artifacts, and run the tests:

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_release.py
python -m pytest -q
```

Run the full benchmark:

```bash
python experiments/multilabel_tfidf_svm.py
```

For a smaller local check, sample the dataset and write to a separate directory:

```bash
python experiments/multilabel_tfidf_svm.py \
  --max-samples 600 \
  --output-dir experiments/results/multilabel_tfidf_svm_smoke
```

## Dataset

| Statistic | Value |
| --- | ---: |
| Documents | 9,867 |
| Documents with more than one IPC code | 8,005 |
| Complete IPC code assignments | 32,831 |
| Document-subclass assignments | 16,687 |
| Unique sections / classes / subclasses | 8 / 120 / 510 |
| Benchmark subclasses | 342 |
| Evaluated documents | 9,740 |

`data/processed/patent_ipc_clean.csv` contains the title, abstract, publication number, complete IPC code list, and the derived section, class, and subclass sets. The earlier single-label columns remain in the file for compatibility. See [`data/metadata/FIELD_SCHEMA.md`](data/metadata/FIELD_SCHEMA.md) for the column definitions.

## Evaluation protocol

1. Define the released label set as the 342 subclasses that occur in at least five documents.
2. Keep the 9,740 documents that have at least one released label.
3. Build five outer folds with iterative multilabel stratification.
4. Within each outer training fold, hold out 15% of the training data and choose one decision threshold by validation micro-F1.
5. Refit TF-IDF and all binary classifiers on the full outer training fold, then score the untouched test fold.
6. Combine the five test folds into one out-of-fold prediction file.

The label set is fixed benchmark metadata. Text features, SVM weights, and decision thresholds are fitted without the outer test fold. Each document receives at least one prediction; when no score reaches the selected threshold, the highest-scoring subclass is used.

The full protocol, fold results, and support analysis are in [`docs/technical_report.md`](docs/technical_report.md).

## Result files

| File | Contents |
| --- | --- |
| `summary.json` | Dataset coverage, run configuration, package versions, aggregate metrics, and fold variation |
| `fold_metrics.csv` | Threshold and metrics for each outer fold |
| `threshold_search.csv` | Validation micro-F1 for every threshold tested in each fold |
| `oof_predictions.csv` | Fold, source dataset row, publication number, true labels, and predictions |
| `label_metrics.csv` | Support, precision, recall, and F1 for each benchmark subclass |

All five files are under `experiments/results/multilabel_tfidf_svm/` and are checked by `scripts/validate_release.py`.

## Repository layout

```text
data/processed/                         Released dataset and label statistics
data/metadata/                          Field definitions
experiments/multilabel_tfidf_svm.py     Main benchmark
experiments/results/multilabel_tfidf_svm/
                                        Out-of-fold results
scripts/build_multilabel_dataset.py     Complete-label dataset builder
scripts/validate_release.py             Data and result checks
tests/                                  Builder and metric tests
```

## Earlier experiments

The repository also retains the single-label TF-IDF, ratio-attention, and Chinese-XLNet experiments from v0.2.0. Those experiments predict the first IPC code attached to each record. They are separate from the multilabel results above.

Large v0.2.0 artifacts are listed in [`docs/release_assets.md`](docs/release_assets.md).

## Limits

- The input is limited to titles and abstracts. It does not include claims or specification text.
- The benchmark has no external test collection, so the reported scores measure performance on this corpus only.
- The model cannot emit the 168 subclasses with fewer than five documents. Metrics against all true subclasses count those assignments as false negatives.
- Feature settings reuse the strongest earlier single-label SVM configuration. Only the decision threshold is selected inside each outer fold.
- This is a research benchmark, not a patent search or filing system.

The MIT license covers the code. Patent metadata remains subject to the terms of its public sources; see [`DATA_NOTICE.md`](DATA_NOTICE.md). Citation metadata is available in [`CITATION.cff`](CITATION.cff).
