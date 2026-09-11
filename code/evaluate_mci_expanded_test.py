from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dhnn_proto_head import DHNNProtoConfig
import importlib.util


trainer_spec = importlib.util.spec_from_file_location(
    "fusion_trainer", Path(__file__).with_name("05_train_fusion_model.py")
)
if trainer_spec is None or trainer_spec.loader is None:
    raise ImportError("Unable to load the fusion trainer.")
trainer = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer)
LABEL_ORDER = trainer.LABEL_ORDER
LABEL_MAP = trainer.LABEL_MAP
align_tabular = trainer.align_tabular
load_feature_dir = trainer.load_feature_dir
run_cv_ensemble = trainer.run_cv_ensemble


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, positive: int, negative: int) -> dict:
    mask = np.isin(y_true, [positive, negative])
    truth = (y_true[mask] == positive).astype(np.int64)
    pair = probabilities[mask][:, [negative, positive]]
    score = pair[:, 1] / pair.sum(axis=1).clip(min=1e-12)
    prediction = (score >= 0.5).astype(np.int64)
    return {
        "subjects": int(mask.sum()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, score)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    splits_dir = root / "data" / "splits"
    participants = pd.read_csv(root / "data" / "bids_clinical" / "participants.tsv", sep="\t")
    split_dfs = {
        split: pd.read_csv(splits_dir / f"{split}.tsv", sep="\t")
        for split in ["train", "val", "test"]
    }
    fod = {
        split: load_feature_dir(root / "features" / "fod", split)
        for split in ["train", "val", "test"]
    }
    mri_pet = {
        split: load_feature_dir(root / "features" / "mri_pet", split)
        for split in ["train", "val", "test"]
    }
    tab = {
        split: align_tabular(participants, split_dfs[split])
        for split in ["train", "val", "test"]
    }
    y = {
        split: split_dfs[split]["diagnosis"].astype(str).map(LABEL_MAP).to_numpy(dtype=np.int64)
        for split in ["train", "val", "test"]
    }

    all_fod = np.concatenate([fod["train"], fod["val"], fod["test"]])
    all_mri_pet = np.concatenate([mri_pet["train"], mri_pet["val"], mri_pet["test"]])
    all_tab = np.concatenate([tab["train"], tab["val"], tab["test"]])
    all_y = np.concatenate([y["train"], y["val"], y["test"]])
    all_subjects = pd.concat([split_dfs["train"], split_dfs["val"], split_dfs["test"]], ignore_index=True)
    original_test = np.zeros(len(all_y), dtype=bool)
    original_test[len(y["train"]) + len(y["val"]):] = True
    test_mask = (all_y == LABEL_MAP["MCI"]) | (original_test & (all_y != LABEL_MAP["MCI"]))
    development_mask = ~test_mask

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
    metrics, probabilities, fold_metrics = run_cv_ensemble(
        fod=all_fod[development_mask],
        mri_pet=all_mri_pet[development_mask],
        tab=all_tab[development_mask],
        y=all_y[development_mask],
        fod_test=all_fod[test_mask],
        mri_pet_test=all_mri_pet[test_mask],
        tab_test=all_tab[test_mask],
        y_test=all_y[test_mask],
        config=config,
        folds=5,
        device="cpu",
    )
    predictions = probabilities.argmax(axis=1)
    test_labels = all_y[test_mask]
    payload = {
        "description": "All available MCI subjects held out with the original CN/AD test subjects.",
        "test_counts": pd.Series(test_labels).map(dict(enumerate(LABEL_ORDER))).value_counts().to_dict(),
        "development_samples": int(development_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "mci_metrics": {
            "mci_vs_cn": binary_metrics(test_labels, probabilities, 1, 0),
            "mci_vs_ad": binary_metrics(test_labels, probabilities, 1, 2),
            "mci_one_vs_rest": {
                "subjects": int(test_labels.size),
                "accuracy": float(accuracy_score(test_labels == 1, predictions == 1)),
                "f1": float(f1_score(test_labels == 1, predictions == 1, zero_division=0)),
                "precision": float(precision_score(test_labels == 1, predictions == 1, zero_division=0)),
                "recall": float(recall_score(test_labels == 1, predictions == 1, zero_division=0)),
                "mcc": float(matthews_corrcoef(test_labels == 1, predictions == 1)),
                "roc_auc": float(roc_auc_score(test_labels == 1, probabilities[:, 1])),
                "confusion_matrix": confusion_matrix(
                    test_labels == 1, predictions == 1, labels=[False, True]
                ).tolist(),
            },
        },
        "overall_metrics": metrics,
        "overall_confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=np.arange(len(LABEL_ORDER))
        ).tolist(),
        "fold_metrics": fold_metrics,
        "pca_components": {"fod": 20, "mri_pet": 35},
        "train_with_val": False,
        "test_subjects": all_subjects.loc[test_mask, "participant_id"].tolist(),
    }
    output = root / "results" / "mci_expanded_test_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
