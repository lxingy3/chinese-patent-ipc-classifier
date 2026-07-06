from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "data" / "processed" / "patent_ipc_clean.csv"
    if not data.exists():
        raise SystemExit("Missing patent_ipc_clean.csv")

    df = pd.read_csv(data)
    required = {"patent_title", "patent_abstract", "ipc_level_1", "ipc_level_2", "ipc_level_3"}
    if not required.issubset(df.columns):
        raise SystemExit("Missing required columns")
    if df.duplicated(["patent_title", "patent_abstract"]).any():
        raise SystemExit("Duplicate title and abstract pairs found")
    if len(df) < 1000:
        raise SystemExit("Dataset is unexpectedly small")

    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".csv", ".cff"}:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            for marker in ["C:\\Users", "Users\\\\33672", "\u4efb\u52a14", "\u674e\u7fd4\u5b87"]:
                if marker in text:
                    raise SystemExit(f"Forbidden marker {marker!r} found in {path}")

    print(f"Validated {len(df)} patent rows.")


if __name__ == "__main__":
    main()
