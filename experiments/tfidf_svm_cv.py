from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ipc_level_1", "ipc_level_2", "ipc_level_3"]


def weighted_text(row: pd.Series, title_weight: int) -> str:
    title = "" if pd.isna(row["patent_title"]) else str(row["patent_title"])
    abstract = "" if pd.isna(row["patent_abstract"]) else str(row["patent_abstract"])
    return (title * max(title_weight, 1)) + "\n" + abstract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_ipc_clean.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "tfidf_svm_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--title-weight", type=int, default=10)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts = [weighted_text(row, args.title_weight) for _, row in df.iterrows()]
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_rows: list[dict[str, float | int]] = []
    predictions: list[dict[str, str | int]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, df["ipc_level_3"]), start=1):
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, float | int] = {"fold": fold, "test_samples": int(len(test_idx))}
        fold_preds: dict[str, np.ndarray] = {}
        for label in LABELS:
            model = make_pipeline(
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(args.ngram_min, args.ngram_max),
                    min_df=2,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
                LinearSVC(C=args.c, max_iter=5000, random_state=args.seed),
            )
            model.fit([texts[i] for i in train_idx], df.iloc[train_idx][label].astype(str))
            pred = model.predict([texts[i] for i in test_idx])
            fold_preds[label] = pred
            row[f"{label}_accuracy"] = float(accuracy_score(df.iloc[test_idx][label].astype(str), pred))
            joblib.dump(model, fold_dir / f"{label}.joblib")

        fold_rows.append(row)
        for pos, idx in enumerate(test_idx):
            item: dict[str, str | int] = {"fold": fold, "row_index": int(idx)}
            item["patent_title"] = str(df.iloc[idx]["patent_title"])
            for label in LABELS:
                item[f"{label}_true"] = str(df.iloc[idx][label])
                item[f"{label}_pred"] = str(fold_preds[label][pos])
            predictions.append(item)

        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        f"{label}_accuracy_mean": float(np.mean([row[f"{label}_accuracy"] for row in fold_rows]))
        for label in LABELS
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "folds": fold_rows, "config": vars(args)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

