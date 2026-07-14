# Multilabel TF-IDF baseline

## Model

The benchmark fits 342 independent `LinearSVC` classifiers on character TF-IDF features. Each classifier predicts one IPC subclass. The title is repeated ten times before the abstract, and the vectorizer uses 1 to 4 character n-grams with at most 200,000 features.

There is no deployable checkpoint for this baseline. The repository releases the training code and out-of-fold predictions, and the five models are fitted again when the benchmark runs.

## Input and output

Input: a Chinese patent title and abstract.

Output: one or more IPC subclasses from the released 342-label vocabulary. Every prediction contains at least one subclass.

## Training and evaluation

The model is evaluated with five-fold iterative multilabel stratification on 9,740 documents. A 15% split inside each outer training fold selects one decision threshold. The outer test fold is used once for scoring.

```bash
python experiments/multilabel_tfidf_svm.py
```

## Out-of-fold results

| Metric | Value |
| --- | ---: |
| Subclass micro-F1 | 0.4938 |
| Subclass macro-F1 | 0.2978 |
| Subclass samples-F1 | 0.4824 |
| Exact set match | 0.1843 |
| Class micro-F1 | 0.5815 |
| Section micro-F1 | 0.7279 |
| Hierarchy micro-F1 | 0.5867 |

The result files are under `experiments/results/multilabel_tfidf_svm/`.

## Intended use

This model is a reference point for Chinese patent multilabel classification and IPC hierarchy experiments. The released OOF file can also be used for error analysis or for evaluating another model on the same folds.

## Limits

- The training text contains titles and abstracts, not claims or specifications.
- The vocabulary excludes subclasses found in fewer than five documents.
- Ninety-two benchmark labels have zero out-of-fold F1.
- The corpus has no external test set.
- The output is not suitable for patent filing, legal review, or production classification without further validation.

See [`docs/technical_report.md`](docs/technical_report.md) for the protocol and support analysis.
