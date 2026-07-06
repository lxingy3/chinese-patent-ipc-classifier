from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ipc_level_1", "ipc_level_2", "ipc_level_3"]
PAD = "<PAD>"
UNK = "<UNK>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_terms(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text.lower())
    terms = re.findall(r"[a-z0-9][a-z0-9_\-/\.]{1,}", text)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for n in (2, 3):
            terms.extend(chunk[i : i + n] for i in range(max(len(chunk) - n + 1, 0)))
    return terms


def build_vocab(rows: list[list[str]], max_vocab: int, min_freq: int) -> dict[str, int]:
    counter = Counter(term for row in rows for term in row)
    vocab = {PAD: 0, UNK: 1}
    for term, count in counter.most_common(max_vocab - 2):
        if count < min_freq:
            break
        vocab[term] = len(vocab)
    return vocab


def encode(terms: list[str], vocab: dict[str, int], top_k: int) -> tuple[list[int], list[float]]:
    counts = Counter(terms)
    total = sum(counts.values()) or 1
    pairs = counts.most_common(top_k)
    ids = [vocab.get(term, vocab[UNK]) for term, _ in pairs]
    ratios = [count / total for _, count in pairs]
    while len(ids) < top_k:
        ids.append(vocab[PAD])
        ratios.append(0.0)
    return ids, ratios


class PatentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, terms: list[list[str]], vocab: dict[str, int], maps: dict[str, dict[str, int]], top_k: int) -> None:
        self.df = df.reset_index(drop=True)
        self.terms = terms
        self.vocab = vocab
        self.maps = maps
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, ratios = encode(self.terms[idx], self.vocab, self.top_k)
        labels = [self.maps[col][str(self.df.iloc[idx][col])] for col in LABELS]
        return {
            "term_ids": torch.tensor(ids, dtype=torch.long),
            "ratios": torch.tensor(ratios, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class RatioAttentionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.gate = nn.Linear(1, embed_dim)

    def forward(self, term_ids: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        x = self.embedding(term_ids) + self.gate(ratios.unsqueeze(-1))
        mask = term_ids.eq(0)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        weights = ratios.masked_fill(mask, 0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (encoded * weights.unsqueeze(-1)).sum(dim=1)


class HierarchicalClassifier(nn.Module):
    def __init__(self, vocab_size: int, label_sizes: list[int], embed_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = RatioAttentionEncoder(vocab_size, embed_dim, heads, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head1 = nn.Linear(embed_dim, label_sizes[0])
        self.head2 = nn.Linear(embed_dim + label_sizes[0], label_sizes[1])
        self.head3 = nn.Linear(embed_dim + label_sizes[1], label_sizes[2])

    def forward(self, term_ids: torch.Tensor, ratios: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(self.encoder(term_ids, ratios))
        out1 = self.head1(h)
        out2 = self.head2(torch.cat([h, torch.softmax(out1, dim=1)], dim=1))
        out3 = self.head3(torch.cat([h, torch.softmax(out2, dim=1)], dim=1))
        return [out1, out2, out3]


def label_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {col: {label: i for i, label in enumerate(sorted(df[col].astype(str).unique()))} for col in LABELS}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = np.zeros(len(LABELS), dtype=np.int64)
    total = 0
    for batch in loader:
        outputs = model(batch["term_ids"].to(device), batch["ratios"].to(device))
        labels = batch["labels"].to(device)
        for i, logits in enumerate(outputs):
            correct[i] += int((logits.argmax(dim=1) == labels[:, i]).sum().item())
        total += labels.shape[0]
    return {f"{LABELS[i]}_accuracy": float(correct[i] / max(total, 1)) for i in range(len(LABELS))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_ipc_clean.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "ratio_attention_ipc")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--max-vocab", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    terms = [text_terms(f"{row.patent_title} {row.patent_abstract}") for row in df.itertuples()]
    vocab = build_vocab(terms, args.max_vocab, min_freq=2)
    maps = label_maps(df)
    split = int(len(df) * 0.8)
    train_ds = PatentDataset(df.iloc[:split], terms[:split], vocab, maps, args.top_k)
    valid_ds = PatentDataset(df.iloc[split:], terms[split:], vocab, maps, args.top_k)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalClassifier(len(vocab), [len(maps[col]) for col in LABELS], 128, 4, 1, 0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            outputs = model(batch["term_ids"].to(device), batch["ratios"].to(device))
            labels = batch["labels"].to(device)
            loss = sum(nn.functional.cross_entropy(logits, labels[:, i]) for i, logits in enumerate(outputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        metrics = evaluate(model, valid_loader, device)
        metrics.update({"epoch": epoch, "loss": total_loss / max(len(train_loader), 1)})
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "label_maps.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

