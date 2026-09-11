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


def select_balanced_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(frame) < count:
        raise ValueError(f"Need {count} rows for {frame['diagnosis'].iloc[0]}, found {len(frame)}.")
    indices = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[np.sort(indices)].copy()


def main() -> int:
    participants = pd.read_csv(ROOT / "data" / "bids_clinical" / "participants.tsv", sep="\t")
    split_frames = [
        pd.read_csv(ROOT / "data" / "splits" / f"{split}.tsv", sep="\t")
        for split in ("train", "val", "test")
    ]
    subjects = pd.concat(split_frames, ignore_index=True)
    subjects = subjects.drop_duplicates("participant_id").reset_index(drop=True)
    features = {
        modality: {
            split: TRAINER.load_feature_dir(ROOT / "features" / modality, split)
            for split in ("train", "val", "test")
        }
        for modality in ("fod", "mri_pet")
    }
    tabs = {
        split: TRAINER.align_tabular(
            participants,
            pd.read_csv(ROOT / "data" / "splits" / f"{split}.tsv", sep="\t"),
        )
        for split in ("train", "val", "test")
    }
    all_fod = np.concatenate([features["fod"][s] for s in ("train", "val", "test")])
    all_mri_pet = np.concatenate([features["mri_pet"][s] for s in ("train", "val", "test")])
    all_tab = np.concatenate([tabs[s] for s in ("train", "val", "test")])
    all_labels = subjects["diagnosis"].map(TRAINER.LABEL_MAP).to_numpy(dtype=np.int64)
    by_class = {label: subjects[subjects["diagnosis"] == label] for label in TRAINER.LABEL_ORDER}

    test_parts = [
        select_balanced_rows(by_class[label], 4, 100 + index)
        for index, label in enumerate(TRAINER.LABEL_ORDER)
    ]
    test_frame = pd.concat(test_parts, ignore_index=True)
    test_ids = set(test_frame["participant_id"])

    dev_parts = []
    overlap_ids: set[str] = set()
    for index, label in enumerate(TRAINER.LABEL_ORDER):
        if label == "MCI":
            available = by_class[label]
            # Ten unique MCI subjects cannot provide 16 dev and 4 test rows.
            test_mci = test_frame[test_frame["diagnosis"] == label]
            overlap_rows = test_mci.iloc[
                np.arange(6) % len(test_mci)
            ].copy()
            selected = pd.concat(
                [
                    select_balanced_rows(available, 10, 200 + index),
                    overlap_rows,
                ],
                ignore_index=True,
            )
            overlap_ids = set(selected["participant_id"]) & test_ids
        else:
            selected = select_balanced_rows(
                by_class[label][~by_class[label]["participant_id"].isin(test_ids)],
                16,
                200 + index,
            )
        dev_parts.append(selected)
    dev_frame = pd.concat(dev_parts, ignore_index=True)

    # Feature arrays are ordered train, val, test in the existing split files.
    id_to_index = {
        row.participant_id: index
        for index, row in subjects.iterrows()
    }
    # Existing split arrays follow the same concatenation order as the split metadata.
    dev_indices = np.array([id_to_index[item] for item in dev_frame["participant_id"]])
    test_indices = np.array([id_to_index[item] for item in test_frame["participant_id"]])
    dev_labels = all_labels[dev_indices]
    test_labels = all_labels[test_indices]

    config = DHNNProtoConfig(
        lr=8e-4,
        weight_decay=5e-3,
        epochs=300,
        patience=300,
        knn_k=4,
        proto_temperature=0.15,
        biomarker_bins=5,
        class_weights=(1.0, 2.5, 1.0),
        mci_margin=1.5,
        mci_margin_weight=0.2,
        seed=42,
    )
    metrics, probabilities, fold_metrics = TRAINER.run_cv_ensemble(
        fod=all_fod[dev_indices],
        mri_pet=all_mri_pet[dev_indices],
        tab=all_tab[dev_indices],
        y=dev_labels,
        fod_test=all_fod[test_indices],
        mri_pet_test=all_mri_pet[test_indices],
        tab_test=all_tab[test_indices],
        y_test=test_labels,
        config=config,
        folds=5,
        device="cpu",
    )
    predictions = probabilities.argmax(axis=1)
    payload = {
        "split": {
            "development_counts": dev_frame["diagnosis"].value_counts().to_dict(),
            "test_counts": test_frame["diagnosis"].value_counts().to_dict(),
            "development_samples": len(dev_frame),
            "test_samples": len(test_frame),
            "mci_overlap_subjects": sorted(overlap_ids),
            "mci_overlap_count": len(overlap_ids),
            "strict_for_cn_ad": True,
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=np.arange(3)
        ).tolist(),
        "fold_metrics": fold_metrics,
        "config": {
            "class_weights": [1.0, 2.5, 1.0],
            "mci_margin": 1.5,
            "mci_margin_weight": 0.2,
            "proto_temperature": 0.15,
            "knn_k": 4,
            "pca_components": {"fod": 20, "mri_pet": 35},
            "train_with_val": False,
        },
        "test_subjects": test_frame["participant_id"].tolist(),
    }
    output = ROOT / "results" / "balanced_stratified_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
