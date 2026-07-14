from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "data" / "processed" / "patent_ipc_clean.csv"
    if not data.exists():
        raise SystemExit("Missing patent_ipc_clean.csv")

    df = pd.read_csv(data)
    required = {
        "patent_title",
        "patent_abstract",
        "ipc_level_1",
        "ipc_level_2",
        "ipc_level_3",
        "ipc_level_4",
        "publication_numbers",
        "ipc_codes",
        "ipc_sections",
        "ipc_classes",
        "ipc_subclasses",
        "ipc_code_count",
    }
    if not required.issubset(df.columns):
        raise SystemExit("Missing required columns")
    if df.duplicated(["patent_title", "patent_abstract"]).any():
        raise SystemExit("Duplicate title and abstract pairs found")
    if len(df) < 1000:
        raise SystemExit("Dataset is unexpectedly small")
    code_lists = df["ipc_codes"].str.split("|")
    if not (code_lists.str.len() == df["ipc_code_count"]).all():
        raise SystemExit("ipc_code_count does not match ipc_codes")
    if not all(primary in codes for primary, codes in zip(df["ipc_level_4"], code_lists)):
        raise SystemExit("A primary IPC code is absent from its complete code list")
    if int((df["ipc_code_count"] > 1).sum()) < len(df) // 2:
        raise SystemExit("Multilabel coverage is unexpectedly low")

    results = ROOT / "experiments" / "results" / "multilabel_tfidf_svm"
    result_files = {
        "fold_metrics.csv",
        "label_metrics.csv",
        "oof_predictions.csv",
        "summary.json",
        "threshold_search.csv",
    }
    missing_results = [name for name in sorted(result_files) if not (results / name).exists()]
    if missing_results:
        raise SystemExit(f"Missing multilabel results: {', '.join(missing_results)}")

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    if summary["dataset"]["documents"] != len(df):
        raise SystemExit("Benchmark summary does not match the dataset")
    min_label_count = summary["config"]["min_label_count"]
    label_sets = [set(value.split("|")) for value in df["ipc_subclasses"]]
    support = Counter(label for labels in label_sets for label in labels)
    vocabulary = {label for label, count in support.items() if count >= min_label_count}
    expected_rows = [index for index, labels in enumerate(label_sets) if labels & vocabulary]

    predictions = pd.read_csv(results / "oof_predictions.csv").fillna("")
    if predictions["dataset_row"].tolist() != expected_rows:
        raise SystemExit("OOF predictions do not map to the expected dataset rows")
    if not (
        predictions["publication_numbers"].to_numpy()
        == df.loc[expected_rows, "publication_numbers"].to_numpy()
    ).all():
        raise SystemExit("OOF publication numbers do not match the dataset")
    if (predictions["predicted_subclasses"] == "").any():
        raise SystemExit("OOF output contains an empty prediction")
    if set(predictions["fold"]) != set(range(1, summary["config"]["folds"] + 1)):
        raise SystemExit("OOF output has unexpected fold assignments")

    truth = [set(value.split("|")) for value in predictions["true_subclasses_benchmark"]]
    predicted = [set(value.split("|")) for value in predictions["predicted_subclasses"]]
    true_positive = sum(len(actual & guess) for actual, guess in zip(truth, predicted))
    false_positive = sum(len(guess - actual) for actual, guess in zip(truth, predicted))
    false_negative = sum(len(actual - guess) for actual, guess in zip(truth, predicted))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    micro_f1 = 2 * precision * recall / (precision + recall)
    reported_f1 = summary["oof_metrics"]["benchmark_subclass_micro_f1"]
    if abs(micro_f1 - reported_f1) > 1e-12:
        raise SystemExit("Reported subclass micro-F1 does not match the OOF predictions")

    label_metrics = pd.read_csv(results / "label_metrics.csv")
    if set(label_metrics["ipc_subclass"]) != vocabulary:
        raise SystemExit("Per-label metrics do not match the benchmark vocabulary")
    reported_support = dict(zip(label_metrics["ipc_subclass"], label_metrics["support"]))
    if any(reported_support[label] != support[label] for label in vocabulary):
        raise SystemExit("Per-label support does not match the dataset")

    ignored_directories = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*"):
        if ignored_directories.intersection(path.parts):
            continue
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".csv", ".cff"}:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            for marker in ["C:\\Users", "Users\\\\33672", "\u4efb\u52a14", "\u674e\u7fd4\u5b87"]:
                if marker in text:
                    raise SystemExit(f"Forbidden marker {marker!r} found in {path}")

    print(f"Validated {len(df)} patent rows.")


if __name__ == "__main__":
    main()
