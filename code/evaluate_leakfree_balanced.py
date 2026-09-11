from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dhnn_proto_head import DHNNProtoConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fusion_trainer", ROOT / "code" / "05_train_fusion_model.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Unable to load the fusion trainer.")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def choose(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"Need {count} {frame['diagnosis'].iloc[0]} subjects, found {len(frame)}.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[np.sort(selected)].copy()


def main() -> int:
    participants = pd.read_csv(ROOT / "data" / "bids_clinical" / "participants.tsv", sep="\t")
    split_frames = [
        pd.read_csv(ROOT / "data" / "splits" / f"{split}.tsv", sep="\t")
        for split in ("train", "val", "test")
    ]
    subjects = pd.concat(split_frames, ignore_index=True).drop_duplicates("participant_id").reset_index(drop=True)
    modalities = {
        name: {
            split: TRAINER.load_feature_dir(ROOT / "features" / name, split)
            for split in ("train", "val", "test")
        }
        for name in ("fod", "mri_pet")
    }
    clinical = {
        split: TRAINER.align_tabular(
            participants,
            pd.read_csv(ROOT / "data" / "splits" / f"{split}.tsv", sep="\t"),
        )
        for split in ("train", "val", "test")
    }
    all_fod = np.concatenate([modalities["fod"][split] for split in ("train", "val", "test")])
    all_mri_pet = np.concatenate([modalities["mri_pet"][split] for split in ("train", "val", "test")])
    all_clinical = np.concatenate([clinical[split] for split in ("train", "val", "test")])
    all_y = subjects["diagnosis"].map(TRAINER.LABEL_MAP).to_numpy(dtype=np.int64)
    by_class = {label: subjects[subjects["diagnosis"] == label] for label in TRAINER.LABEL_ORDER}

    # Strict disjoint evaluation: 16/8/16 development and 4/2/4 test.
    development_parts = [
        choose(by_class["CN"], 16, 110),
        choose(by_class["MCI"], 8, 111),
        choose(by_class["AD"], 16, 112),
    ]
    development = pd.concat(development_parts, ignore_index=True)
    development_ids = set(development["participant_id"])
    test_parts = [
        choose(by_class["CN"][~by_class["CN"]["participant_id"].isin(development_ids)], 4, 210),
        choose(by_class["MCI"][~by_class["MCI"]["participant_id"].isin(development_ids)], 2, 211),
        choose(by_class["AD"][~by_class["AD"]["participant_id"].isin(development_ids)], 4, 212),
    ]
    test = pd.concat(test_parts, ignore_index=True)
    test_ids = set(test["participant_id"])
    if development_ids & test_ids:
        raise RuntimeError("Strict split construction produced overlapping subjects.")

    id_to_index = {row.participant_id: index for index, row in subjects.iterrows()}
    development_indices = np.array([id_to_index[item] for item in development["participant_id"]])
    test_indices = np.array([id_to_index[item] for item in test["participant_id"]])
    y_dev = all_y[development_indices]
    y_test = all_y[test_indices]

    config = DHNNProtoConfig(
        lr=8e-4,
        weight_decay=5e-3,
        epochs=300,
        patience=300,
        knn_k=4,
        proto_temperature=0.12,
        biomarker_bins=5,
        class_weights=(1.0, 1.5, 1.0),
        mci_margin=1.2,
        mci_margin_weight=0.08,
        seed=42,
    )
    metrics, probabilities, fold_metrics = TRAINER.run_cv_ensemble(
        fod=all_fod[development_indices],
        mri_pet=all_mri_pet[development_indices],
        tab=all_clinical[development_indices],
        y=y_dev,
        fod_test=all_fod[test_indices],
        mri_pet_test=all_mri_pet[test_indices],
        tab_test=all_clinical[test_indices],
        y_test=y_test,
        config=config,
        folds=5,
        device="cpu",
    )
    predictions = probabilities.argmax(axis=1)
    payload = {
        "split": {
            "development_counts": development["diagnosis"].value_counts().to_dict(),
            "test_counts": test["diagnosis"].value_counts().to_dict(),
            "development_samples": len(development),
            "test_samples": len(test),
            "overlap_count": len(development_ids & test_ids),
            "strict_leak_free": True,
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=np.arange(3)).tolist(),
        "fold_metrics": fold_metrics,
        "config": {
            "class_weights": [1.0, 1.5, 1.0],
            "mci_margin": 1.2,
            "mci_margin_weight": 0.08,
            "proto_temperature": 0.12,
            "knn_k": 4,
            "pca_components": {"fod": 20, "mri_pet": 35},
            "folds": 5,
            "train_with_val": False,
        },
        "development_subjects": development["participant_id"].tolist(),
        "test_subjects": test["participant_id"].tolist(),
    }
    output = ROOT / "results" / "leakfree_balanced_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
