from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_first(patterns: list[Path]) -> pd.DataFrame | None:
    for path in patterns:
        if path.exists():
            return pd.read_csv(path)
    return None


def parse_date(value) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def participant_to_adni_id(pid: str) -> str:
    s = pid.replace("sub-ADNI", "")
    return f"{s[:3]}_S_{s[4:]}"


def load_manifest(root: Path) -> pd.DataFrame:
    candidates = [
        root / "results" / "combined_33_plus_ad_manifest.csv",
        root / "results" / "final_adni_download_manifest.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["rid"] = df["Subject ID"].astype(str).str.extract(r"_(\d+)$")[0].astype(float).astype("Int64")
            df["study_date"] = df.get("dti_study_date", pd.Series([pd.NA] * len(df))).map(parse_date)
            return df[["Subject ID", "rid", "study_date"]].drop_duplicates("rid")
    return pd.DataFrame(columns=["Subject ID", "rid", "study_date"])


def nearest_moca(moca: pd.DataFrame, rid: int, target_date: pd.Timestamp | pd.NaT) -> float | np.nan:
    if moca is None or "RID" not in moca.columns or "MOCA" not in moca.columns:
        return np.nan
    sub = moca[moca["RID"].astype("Int64") == rid].copy()
    sub["MOCA"] = pd.to_numeric(sub["MOCA"], errors="coerce")
    sub = sub[sub["MOCA"].notna()]
    if sub.empty:
        return np.nan
    if pd.notna(target_date) and "VISDATE" in sub.columns:
        sub["date"] = pd.to_datetime(sub["VISDATE"], errors="coerce")
        sub["delta"] = (sub["date"] - target_date).abs()
        sub = sub.sort_values("delta", na_position="last")
    return float(sub.iloc[0]["MOCA"])


def demographics(ptdemog: pd.DataFrame | None, rid: int, target_date: pd.Timestamp | pd.NaT) -> tuple[float | np.nan, str | float]:
    if ptdemog is None or "RID" not in ptdemog.columns:
        return np.nan, np.nan
    sub = ptdemog[ptdemog["RID"].astype("Int64") == rid].copy()
    if sub.empty:
        return np.nan, np.nan
    if pd.notna(target_date) and "VISDATE" in sub.columns:
        sub["date"] = pd.to_datetime(sub["VISDATE"], errors="coerce")
        sub["delta"] = (sub["date"] - target_date).abs()
        sub = sub.sort_values("delta", na_position="last")
    row = sub.iloc[0]
    gender = row.get("PTGENDER", np.nan)
    sex = {1: "M", 2: "F", "1": "M", "2": "F", "Male": "M", "Female": "F"}.get(gender, np.nan)
    raw_birth = row.get("PTDOBYY", np.nan)
    birth_year = pd.to_numeric(raw_birth, errors="coerce")
    if pd.isna(birth_year):
        birth_date = pd.to_datetime(raw_birth, errors="coerce")
        birth_year = birth_date.year if pd.notna(birth_date) else np.nan
    if pd.notna(target_date) and pd.notna(birth_year):
        age = float(target_date.year - birth_year)
    else:
        age = np.nan
    return age, sex


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Join ADNI MOCA/PTDEMOG fields into BIDS participants.tsv.")
    parser.add_argument("--participants", default=root / "data" / "bids" / "participants.tsv", type=Path)
    parser.add_argument("--tabular-dir", default=root / "ADNI DATA" / "tabular", type=Path)
    parser.add_argument("--out", default=root / "data" / "bids" / "participants.tsv", type=Path)
    args = parser.parse_args()

    participants = pd.read_csv(args.participants, sep="\t", dtype={"rid": "Int64"})
    manifest = load_manifest(root)
    moca = read_first(sorted(args.tabular_dir.glob("MOCA*.csv")))
    ptdemog = read_first(sorted(args.tabular_dir.glob("PTDEMOG*.csv")))

    rows = []
    for _, row in participants.iterrows():
        rid = int(row["rid"]) if pd.notna(row.get("rid")) else int(participant_to_adni_id(row["participant_id"]).split("_")[-1])
        mrow = manifest[manifest["rid"] == rid]
        study_date = mrow.iloc[0]["study_date"] if not mrow.empty else pd.NaT
        age, sex = demographics(ptdemog, rid, study_date)
        moca_score = nearest_moca(moca, rid, study_date)
        out = row.copy()
        out["adni_id"] = row.get("adni_id", participant_to_adni_id(row["participant_id"]))
        out["rid"] = rid
        out["age"] = age
        out["sex"] = sex
        out["moca"] = moca_score
        rows.append(out)

    out_df = pd.DataFrame(rows)
    for col in ("age", "moca"):
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
        if out_df[col].isna().any():
            out_df[col] = out_df[col].fillna(out_df[col].median())
    if out_df["sex"].isna().any():
        mode = out_df["sex"].dropna().mode()
        out_df["sex"] = out_df["sex"].fillna(mode.iloc[0] if not mode.empty else "M")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out} with {len(out_df)} rows")
    print(out_df[["diagnosis", "age", "sex", "moca"]].head().to_string(index=False))
    print("Missing after join:", out_df[["age", "sex", "moca"]].isna().sum().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
