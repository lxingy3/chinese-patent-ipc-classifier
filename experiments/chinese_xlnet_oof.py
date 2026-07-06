from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]


class PatentTextDataset(Dataset):
    def __init__(self, encodings, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with: pip install -r requirements-xlnet.txt") from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def stratified_folds(labels: list[str], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        buckets.setdefault(label, []).append(idx)
    split = [[] for _ in range(folds)]
    for indices in sorted(buckets.values(), key=len, reverse=True):
        rng.shuffle(indices)
        for offset, idx in enumerate(indices):
            split[offset % folds].append(idx)
    return [sorted(indices) for indices in split]


def encode_texts(tokenizer, texts: list[str], max_length: int):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        logits = model(**inputs).logits
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_fold(args, fold: int, train_df: pd.DataFrame, valid_df: pd.DataFrame, label_to_id: dict[str, int]):
    AutoModelForSequenceClassification, AutoTokenizer = load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_to_id))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_texts = (train_df["patent_title"].astype(str) + "\n" + train_df["patent_abstract"].astype(str)).tolist()
    valid_texts = (valid_df["patent_title"].astype(str) + "\n" + valid_df["patent_abstract"].astype(str)).tolist()
    train_labels = [label_to_id[str(value)] for value in train_df["ipc_level_3"]]
    valid_labels = [label_to_id[str(value)] for value in valid_df["ipc_level_3"]]

    train_loader = DataLoader(
        PatentTextDataset(encode_texts(tokenizer, train_texts, args.max_length), train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        PatentTextDataset(encode_texts(tokenizer, valid_texts, args.max_length), valid_labels),
        batch_size=args.batch_size,
    )

    class_counts = np.bincount(train_labels, minlength=len(label_to_id)).astype(np.float32)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs).logits
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        pred = predict(model, valid_loader, device)
        accuracy = float((pred == np.array(valid_labels)).mean())
        row = {"fold": fold, "epoch": epoch, "loss": total_loss / max(len(train_loader), 1), "ipc_level_3_accuracy": accuracy}
        history.append(row)
        print(json.dumps(row), flush=True)

    pred = predict(model, valid_loader, device)
    return pred, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_ipc_clean.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "results" / "chinese_xlnet_oof")
    parser.add_argument("--model-name", default="hfl/chinese-xlnet-mid")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    label_to_id = {label: idx for idx, label in enumerate(sorted(df["ipc_level_3"].astype(str).unique()))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    folds = stratified_folds(df["ipc_level_3"].astype(str).tolist(), args.folds, args.seed)

    predictions = []
    history = []
    all_idx = np.arange(len(df))
    for fold, valid_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_idx, np.array(valid_idx))
        pred_ids, fold_history = train_fold(args, fold, df.iloc[train_idx], df.iloc[valid_idx], label_to_id)
        history.extend(fold_history)
        for pos, row_idx in enumerate(valid_idx):
            true_level_3 = str(df.iloc[row_idx]["ipc_level_3"])
            pred_level_3 = id_to_label[int(pred_ids[pos])]
            predictions.append(
                {
                    "fold": fold,
                    "row_index": int(row_idx),
                    "patent_title": str(df.iloc[row_idx]["patent_title"]),
                    "ipc_level_1_true": true_level_3[:1],
                    "ipc_level_1_pred": pred_level_3[:1],
                    "ipc_level_2_true": true_level_3[:3],
                    "ipc_level_2_pred": pred_level_3[:3],
                    "ipc_level_3_true": true_level_3,
                    "ipc_level_3_pred": pred_level_3,
                }
            )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

