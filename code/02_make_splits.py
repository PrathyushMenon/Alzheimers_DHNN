"""Create the fixed 60/20/20 stratified split files used by every model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", default=str(repo_root / "data" / "bids" / "participants.tsv"))
    parser.add_argument("--out-dir", default=str(repo_root / "data" / "splits"))
    args = parser.parse_args()

    participants = Path(args.participants)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(participants, sep="\t")
    if df.empty:
        raise SystemExit("Cannot create splits: participants.tsv is empty.")
    if "diagnosis" not in df.columns:
        raise SystemExit("Cannot create splits: participants.tsv lacks a diagnosis column.")
    counts = df["diagnosis"].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise SystemExit(f"Cannot create stratified 60/20/20 splits from class counts: {counts.to_dict()}")

    train_df, temp_df = train_test_split(
        df, test_size=0.40, stratify=df["diagnosis"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=42
    )

    train_df.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    val_df.to_csv(out_dir / "val.tsv", sep="\t", index=False)
    test_df.to_csv(out_dir / "test.tsv", sep="\t", index=False)

    for name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{name} ({len(d)} subjects):")
        print(d["diagnosis"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
