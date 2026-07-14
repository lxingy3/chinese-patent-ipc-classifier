# Hierarchical multilabel benchmark

## Task

A patent may have several IPC codes. Predicting only the first code turns that set into an arbitrary single label and drops information that is present in the source record. This benchmark predicts the full set of IPC subclasses from a Chinese patent title and abstract.

The target is the subclass level, such as `G06F`. Class and section scores are derived from the predicted subclasses. A separate hierarchy score gives credit for correctly predicted ancestors while keeping subclass errors visible.

The older `ipc_level_*` fields and single-label experiments remain in the repository, but they are not used as targets in this benchmark.

## Dataset construction

The source collection contains 10,000 public Chinese patent records. The cleaned table removes missing IPC records and duplicate normalized title and abstract pairs, leaving 9,867 documents.

`scripts/build_multilabel_dataset.py` joins the cleaned table to captured IPC lookup results by normalized title and abstract. It performs the following checks before writing a row:

- IPC codes must match the expected section, class, subclass, and group syntax.
- Repeated lookup results for the same source row must not disagree.
- Duplicate source texts must have the same complete IPC set.
- The first IPC code used by the older dataset must appear in the complete set.

All 9,867 cleaned rows matched the source collection. The 119 duplicate text groups agreed on their IPC sets. Publication numbers are retained so a released row can be traced back to its public record.

| Statistic | Value |
| --- | ---: |
| Documents | 9,867 |
| Documents with more than one complete IPC code | 8,005 |
| Complete IPC code assignments | 32,831 |
| Mean complete codes per document | 3.327 |
| Maximum complete codes on one document | 35 |
| Unique sections | 8 |
| Unique classes | 120 |
| Unique subclasses | 510 |
| Document-subclass assignments | 16,687 |

Several full IPC codes can share one subclass, which is why the subclass assignment count is lower than the complete code count.

## Benchmark label set

The released label set contains every subclass found in at least five documents. This leaves 342 of the 510 observed subclasses. It retains 16,324 of 16,687 subclass assignments, or 97.82%.

Of the 9,867 documents, 9,740 have at least one released label. The other 127 remain in the dataset but are not part of the model evaluation. For the evaluated documents, metrics against all true subclasses still count excluded rare labels as false negatives.

The label set is defined once from the released corpus. It is benchmark metadata, not a parameter selected from model scores.

## Model

The reference model uses the same sparse-text settings as the strongest earlier single-label baseline:

- character TF-IDF with 1 to 4 character n-grams
- at most 200,000 features and a minimum document frequency of 2
- the title repeated ten times before the abstract
- one `LinearSVC` per subclass with `C=4` and balanced class weights

Repeating the title is a simple weighting method. It keeps the feature pipeline inspectable and avoids a tokenizer or a separate feature branch.

Each binary classifier returns a decision score. A shared threshold converts those scores to a label set. If no score reaches the threshold, the model emits the highest-scoring subclass so that every document has a prediction.

## Cross-validation and calibration

The outer evaluation uses five-fold iterative multilabel stratification with seed 42. Iterative stratification preserves rare label frequencies more closely than ordinary random splits.

Each outer fold is evaluated as follows:

1. Split the outer training documents into an inner training set and a 15% validation set.
2. Fit TF-IDF and the one-vs-rest classifiers on the inner training set.
3. Search 31 thresholds from -1.00 to 0.50 in increments of 0.05. Select the threshold with the highest validation micro-F1.
4. Fit a new TF-IDF vocabulary and classifiers on the full outer training set.
5. Apply the selected threshold to the untouched outer test fold.

The five outer test predictions are combined in dataset order. No document is scored by a model trained on that document. The test folds do not choose text features, SVM weights, or thresholds.

## Metrics

`benchmark_subclass` metrics use only the 342 released labels in both truth and predictions.

`all_subclass` metrics use the complete true subclass set for each evaluated document. Predictions remain limited to the released labels, so a rare true subclass is a false negative.

`class` and `section` metrics collapse each subclass to its three-character class or one-character section. `hierarchy` metrics expand every subclass into three typed labels: its section, class, and subclass.

The report also includes subclass macro-F1, samples-F1, exact set match, and true and predicted label cardinality.

## Results

### Aggregate out-of-fold scores

| Evaluation | Precision | Recall | F1 | Fold F1 standard deviation |
| --- | ---: | ---: | ---: | ---: |
| Benchmark subclasses | 0.4563 | 0.5380 | 0.4938 | 0.0067 |
| All true subclasses | 0.4563 | 0.5307 | 0.4907 | 0.0065 |
| IPC classes | 0.5562 | 0.6093 | 0.5815 | 0.0049 |
| IPC sections | 0.7083 | 0.7487 | 0.7279 | 0.0073 |
| Expanded hierarchy | 0.5575 | 0.6191 | 0.5867 | 0.0068 |

Subclass samples-F1 is 0.4824 and exact set match is 0.1843. Macro-F1 is 0.2978.

### Fold results

| Fold | Threshold | Subclass precision | Subclass recall | Subclass F1 | Hierarchy F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.45 | 0.4658 | 0.5203 | 0.4916 | 0.5848 |
| 2 | -0.40 | 0.5009 | 0.5132 | 0.5070 | 0.6002 |
| 3 | -0.50 | 0.4409 | 0.5608 | 0.4937 | 0.5852 |
| 4 | -0.55 | 0.4200 | 0.5838 | 0.4885 | 0.5817 |
| 5 | -0.45 | 0.4690 | 0.5121 | 0.4896 | 0.5827 |

All selected thresholds are negative. The resulting mean predicted cardinality is 1.976 labels per document, compared with a true benchmark cardinality of 1.676. This accounts for the higher recall and lower precision. The threshold is selected by validation score in each fold rather than set after looking at test results.

### Label support

| Documents per label | Labels | Mean F1 | Median F1 | Labels with zero F1 |
| --- | ---: | ---: | ---: | ---: |
| 5 to 9 | 87 | 0.1169 | 0.0000 | 61 |
| 10 to 19 | 79 | 0.1988 | 0.1667 | 28 |
| 20 to 49 | 96 | 0.3780 | 0.3649 | 3 |
| 50 to 99 | 46 | 0.4615 | 0.4754 | 0 |
| 100 or more | 34 | 0.5426 | 0.5430 | 0 |

Fold scores are stable, while performance changes sharply with label support. The benchmark contains 92 subclasses with zero F1, and 89 of them have fewer than 20 examples. Class and section scores are higher because some subclass errors remain within the correct ancestor.

The all-subclass F1 is 0.0031 below the released-label score. Rare labels matter at the label level, but they account for 2.18% of the assignments in this corpus.

## Published artifacts

`experiments/results/multilabel_tfidf_svm/` contains:

- `summary.json`: data coverage, configuration, package versions, aggregate scores, fold means, and fold standard deviations
- `fold_metrics.csv`: one row per outer fold
- `threshold_search.csv`: all 155 inner validation threshold results
- `oof_predictions.csv`: one row per evaluated document with a stable `dataset_row`, fold, publication number, truth, and prediction
- `label_metrics.csv`: support and out-of-fold precision, recall, and F1 for all 342 labels

`scripts/validate_release.py` checks that every OOF row maps back to the current dataset, that each fold is present, and that no prediction is empty. Unit tests cover code normalization, duplicate handling, threshold selection, the nonempty prediction rule, and hierarchy scoring.

## Reproduction

The checked-in run used Python 3.12.13, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, and iterative-stratification 0.1.9. The exact configuration and package versions are stored in `summary.json`.

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_release.py
python -m pytest -q
python experiments/multilabel_tfidf_svm.py
```

Rebuilding the enriched CSV also requires the original source workbook and the captured lookup JSON files. Those raw inputs are not stored in this git repository. With those files available, run:

```bash
python scripts/build_multilabel_dataset.py \
  --workbook path/to/source.xlsx \
  --results-dir path/to/lookup-results \
  --clean-data path/to/single-label-clean.csv \
  --output data/processed/patent_ipc_clean.csv \
  --stats-output data/processed/multilabel_dataset_stats.json
```

## Earlier single-label results

The v0.2.0 experiments treat the first IPC code as the only target. Their best level-3 accuracy is 0.4656 for TF-IDF with linear SVM and about 0.4875 for Chinese-XLNet. These numbers are not directly comparable with multilabel F1.

The older scripts and result caches remain under `experiments/` and `experiments/results/model_comparison/` so the release history stays inspectable.

## Limits

The corpus comes from one source collection and has no held-out external dataset. Results may change on patents from another time period or source.

Titles and abstracts omit much of the technical detail found in claims and specifications. Some IPC distinctions cannot be resolved from the released text alone.

The 342-label vocabulary excludes 168 rare subclasses. The all-subclass metric measures this gap, but the model cannot produce an excluded label.

The benchmark reports one sparse linear baseline. It does not establish a state-of-the-art result, and it should not be used to automate patent filing or legal decisions.

## References

- K. Sechidis, G. Tsoumakas, and I. Vlahavas. [On the Stratification of Multi-Label Data](https://doi.org/10.1007/978-3-642-23808-6_10). ECML PKDD, 2011.
- World Intellectual Property Organization. [International Patent Classification FAQ](https://www.wipo.int/en/web/classification-ipc/faq).
