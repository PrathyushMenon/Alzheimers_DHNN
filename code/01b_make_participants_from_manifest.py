"""Build participants.tsv from ADNI download manifests.

Use this when the image download folder does not include the full ADNI clinical
package but the search/export manifests include research groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUP_TO_DX = {
    "CN": "CN",
    "MCI": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "AD": "AD",
}


def adni_to_bids_subject(adni_id: str) -> str:
    return "sub-ADNI" + adni_id.replace("_", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"D:\ALZ\results\combined_33_plus_ad_manifest.csv")
    parser.add_argument("--out", default=r"D:\ALZ\data\bids\participants.tsv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    group_col = "research_group" if "research_group" in manifest.columns else "Research Group"
    df = manifest[["Subject ID", group_col]].drop_duplicates("Subject ID").copy()
    df["diagnosis"] = df[group_col].astype(str).str.upper().map(GROUP_TO_DX)
    df = df.dropna(subset=["diagnosis"])
    df["participant_id"] = df["Subject ID"].map(adni_to_bids_subject)
    df["session_id"] = "ses-M000"
    df["adni_id"] = df["Subject ID"]
    df["rid"] = df["Subject ID"].str.extract(r"_S_(\d+)").astype(int)
    df["age"] = pd.NA
    df["sex"] = pd.NA
    df["moca"] = pd.NA
    out_df = df[["participant_id", "session_id", "diagnosis", "age", "sex", "moca", "adni_id", "rid"]].sort_values("participant_id")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path} with {len(out_df)} participants")
    print(out_df["diagnosis"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
