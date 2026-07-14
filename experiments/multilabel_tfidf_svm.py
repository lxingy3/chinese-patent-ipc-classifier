from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def split_labels(value: object) -> list[str]:
    return [label for label in str(value or "").split("|") if label]


def weighted_text(title: object, abstract: object, title_weight: int) -> str:
    title_text = "" if pd.isna(title) else str(title)
    abstract_text = "" if pd.isna(abstract) else str(abstract)
    return ((title_text + "\n") * max(title_weight, 1)) + abstract_text


def build_vocabulary(label_sets: Sequence[set[str]], min_count: int) -> list[str]:
    support = Counter(label for labels in label_sets for label in labels)
    return sorted(label for label, count in support.items() if count >= min_count)


def decode_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    predictions = scores >= threshold
    empty_rows = np.flatnonzero(~predictions.any(axis=1))
    if len(empty_rows):
        predictions[empty_rows, scores[empty_rows].argmax(axis=1)] = True
    return predictions


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = decode_scores(scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
                "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
            }
        )
    best = max(rows, key=lambda row: (row["micro_f1"], -abs(row["threshold"])))
    return best["threshold"], rows


def micro_set_metrics(true_sets: Sequence[set[str]], predicted_sets: Sequence[set[str]]) -> dict[str, float]:
    true_positive = sum(len(truth & prediction) for truth, prediction in zip(true_sets, predicted_sets))
    false_positive = sum(len(prediction - truth) for truth, prediction in zip(true_sets, predicted_sets))
    false_negative = sum(len(truth - prediction) for truth, prediction in zip(true_sets, predicted_sets))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parent_sets(label_sets: Sequence[set[str]], length: int) -> list[set[str]]:
    return [{label[:length] for label in labels} for labels in label_sets]


def hierarchy_sets(label_sets: Sequence[set[str]]) -> list[set[str]]:
    return [
        {
            item
            for label in labels
            for item in (f"section:{label[:1]}", f"class:{label[:3]}", f"subclass:{label[:4]}")
        }
        for labels in label_sets
    ]


def evaluate_predictions(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    true_sets_all: Sequence[set[str]],
    predicted_sets: Sequence[set[str]],
) -> dict[str, float]:
    precision, recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_predicted,
        average="micro",
        zero_division=0,
    )
    metrics = {
        "benchmark_subclass_micro_precision": float(precision),
        "benchmark_subclass_micro_recall": float(recall),
        "benchmark_subclass_micro_f1": float(micro_f1),
        "benchmark_subclass_macro_f1": float(f1_score(y_true, y_predicted, average="macro", zero_division=0)),
        "benchmark_subclass_samples_f1": float(f1_score(y_true, y_predicted, average="samples", zero_division=0)),
        "benchmark_subclass_exact_match": float(np.mean(np.all(y_true == y_predicted, axis=1))),
        "true_label_cardinality": float(y_true.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_predicted.sum(axis=1).mean()),
    }
    for prefix, truth, prediction in [
        ("all_subclass", true_sets_all, predicted_sets),
        ("class", parent_sets(true_sets_all, 3), parent_sets(predicted_sets, 3)),
        ("section", parent_sets(true_sets_all, 1), parent_sets(predicted_sets, 1)),
        ("hierarchy", hierarchy_sets(true_sets_all), hierarchy_sets(predicted_sets)),
    ]:
        for name, value in micro_set_metrics(truth, prediction).items():
            metrics[f"{prefix}_micro_{name}"] = float(value)
    return metrics


def make_model(c: float, seed: int, n_jobs: int) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LinearSVC(C=c, class_weight="balanced", max_iter=5000, random_state=seed),
        n_jobs=n_jobs,
    )


def make_vectorizer(ngram_min: int, ngram_max: int, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fit_and_score(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    target_texts: Sequence[str],
    *,
    c: float,
    seed: int,
    ngram_min: int,
    ngram_max: int,
    max_features: int,
    n_jobs: int,
) -> np.ndarray:
    vectorizer = make_vectorizer(ngram_min, ngram_max, max_features)
    train_features = vectorizer.fit_transform(train_texts)
    target_features = vectorizer.transform(target_texts)
    model = make_model(c, seed, n_jobs)
    model.fit(train_features, train_labels)
    return np.asarray(model.decision_function(target_features))


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_csv(args.data).fillna("")
    frame.insert(0, "_dataset_row", np.arange(len(frame)))
    if args.max_samples and len(frame) > args.max_samples:
        frame = frame.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    if "ipc_subclasses" not in frame:
        raise ValueError("Dataset does not contain ipc_subclasses; rebuild it with build_multilabel_dataset.py")

    true_sets_all = [set(split_labels(value)) for value in frame["ipc_subclasses"]]
    total_documents = len(frame)
    all_subclasses_count = len(set().union(*true_sets_all))
    all_assignment_count = sum(len(labels) for labels in true_sets_all)
    vocabulary = build_vocabulary(true_sets_all, args.min_label_count)
    if len(vocabulary) < 2:
        raise ValueError("Fewer than two labels satisfy --min-label-count")
    vocabulary_set = set(vocabulary)
    true_sets_benchmark = [labels & vocabulary_set for labels in true_sets_all]
    retained_assignment_count = sum(len(labels) for labels in true_sets_benchmark)
    documents_with_any_benchmark_label = sum(bool(labels) for labels in true_sets_benchmark)
    documents_with_all_labels_in_benchmark = sum(
        truth == benchmark for truth, benchmark in zip(true_sets_all, true_sets_benchmark)
    )
    keep = np.array([bool(labels) for labels in true_sets_benchmark])
    frame = frame.loc[keep].reset_index(drop=True)
    true_sets_all = [labels for labels, include in zip(true_sets_all, keep) if include]
    true_sets_benchmark = [labels for labels in true_sets_benchmark if labels]

    texts = [
        weighted_text(title, abstract, args.title_weight)
        for title, abstract in zip(frame["patent_title"], frame["patent_abstract"])
    ]
    label_binarizer = MultiLabelBinarizer(classes=vocabulary)
    y = label_binarizer.fit_transform(true_sets_benchmark).astype(bool)
    splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    oof_predictions = np.zeros_like(y, dtype=bool)
    fold_assignments = np.zeros(len(frame), dtype=int)
    fold_rows: list[dict[str, float | int]] = []
    threshold_rows: list[dict[str, float | int]] = []

    for fold, (outer_train, test) in enumerate(splitter.split(np.zeros(len(frame)), y), start=1):
        inner_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=args.validation_size,
            random_state=args.seed + fold,
        )
        inner_train_relative, validation_relative = next(
            inner_splitter.split(np.zeros(len(outer_train)), y[outer_train])
        )
        inner_train = outer_train[inner_train_relative]
        validation = outer_train[validation_relative]

        validation_scores = fit_and_score(
            [texts[index] for index in inner_train],
            y[inner_train],
            [texts[index] for index in validation],
            c=args.c,
            seed=args.seed + fold,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
        )
        threshold, search_rows = choose_threshold(y[validation], validation_scores, thresholds)
        for row in search_rows:
            threshold_rows.append({"fold": fold, **row})

        test_scores = fit_and_score(
            [texts[index] for index in outer_train],
            y[outer_train],
            [texts[index] for index in test],
            c=args.c,
            seed=args.seed + fold,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
        )
        predictions = decode_scores(test_scores, threshold)
        oof_predictions[test] = predictions
        fold_assignments[test] = fold
        predicted_sets = [set(label_binarizer.classes_[row]) for row in predictions]
        fold_metrics = evaluate_predictions(y[test], predictions, [true_sets_all[i] for i in test], predicted_sets)
        fold_rows.append(
            {
                "fold": fold,
                "train_documents": int(len(outer_train)),
                "test_documents": int(len(test)),
                "threshold": threshold,
                **fold_metrics,
            }
        )
        print(json.dumps(fold_rows[-1], ensure_ascii=False), flush=True)

    predicted_sets_all = [set(label_binarizer.classes_[row]) for row in oof_predictions]
    oof_metrics = evaluate_predictions(y, oof_predictions, true_sets_all, predicted_sets_all)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.output_dir / "threshold_search.csv", index=False)

    prediction_rows = []
    for index, (truth_all, truth_benchmark, prediction) in enumerate(
        zip(true_sets_all, true_sets_benchmark, predicted_sets_all)
    ):
        prediction_rows.append(
            {
                "dataset_row": int(frame.iloc[index]["_dataset_row"]),
                "fold": int(fold_assignments[index]),
                "publication_numbers": frame.iloc[index]["publication_numbers"],
                "patent_title": frame.iloc[index]["patent_title"],
                "true_subclasses_all": "|".join(sorted(truth_all)),
                "true_subclasses_benchmark": "|".join(sorted(truth_benchmark)),
                "predicted_subclasses": "|".join(sorted(prediction)),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    label_precision, label_recall, label_f1, label_support = precision_recall_fscore_support(
        y,
        oof_predictions,
        average=None,
        zero_division=0,
    )
    pd.DataFrame(
        {
            "ipc_subclass": vocabulary,
            "support": label_support.astype(int),
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
        }
    ).sort_values(["support", "ipc_subclass"], ascending=[False, True]).to_csv(
        args.output_dir / "label_metrics.csv",
        index=False,
    )

    fold_frame = pd.DataFrame(fold_rows)
    metric_columns = [column for column in fold_frame if column not in {"fold", "train_documents", "test_documents", "threshold"}]
    summary = {
        "dataset": {
            "documents": total_documents,
            "evaluated_documents": len(frame),
            "benchmark_subclasses": len(vocabulary),
            "all_subclasses": all_subclasses_count,
            "assignment_coverage": retained_assignment_count / all_assignment_count,
            "document_coverage": documents_with_any_benchmark_label / total_documents,
            "documents_with_all_labels_in_benchmark": documents_with_all_labels_in_benchmark / total_documents,
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"data", "output_dir"}
        },
        "environment": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ["numpy", "pandas", "scikit-learn", "iterative-stratification"]
            },
        },
        "oof_metrics": oof_metrics,
        "fold_mean": {column: float(fold_frame[column].mean()) for column in metric_columns},
        "fold_std": {column: float(fold_frame[column].std(ddof=0)) for column in metric_columns},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated hierarchical multilabel IPC baseline.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_ipc_clean.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "results" / "multilabel_tfidf_svm",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--min-label-count", type=int, default=5)
    parser.add_argument("--title-weight", type=int, default=10)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=0.5)
    parser.add_argument("--threshold-steps", type=int, default=31)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["oof_metrics"], ensure_ascii=False, indent=2))
