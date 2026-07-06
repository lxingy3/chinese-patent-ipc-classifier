from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_ipc_clean.csv")
    parser.add_argument("--label", default="ipc_level_2")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=3000)
    args = parser.parse_args()

    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42)
    counts = df[args.label].value_counts()
    df = df[df[args.label].isin(counts[counts >= args.min_count].index)].copy()
    counts = df[args.label].value_counts()
    df = df[df[args.label].isin(counts[counts >= 2].index)].copy()

    df["text"] = df["patent_title"].astype(str) + "\n" + df["patent_abstract"].astype(str)
    train_x, test_x, train_y, test_y = train_test_split(
        df["text"],
        df[args.label],
        test_size=0.2,
        random_state=42,
        stratify=df[args.label],
    )

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50_000),
        LinearSVC(C=2.0),
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    print(f"label={args.label}")
    print(f"samples={len(df)} classes={df[args.label].nunique()}")
    print(f"accuracy={accuracy_score(test_y, pred):.4f}")
    print(classification_report(test_y, pred, zero_division=0))


if __name__ == "__main__":
    main()
