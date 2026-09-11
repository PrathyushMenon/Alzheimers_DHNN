from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dhnn_proto_head import DHNNProtoConfig, train_dhnn_proto_classifier

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trainer", ROOT / "code" / "05_train_fusion_model.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Unable to load the fusion trainer.")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def prepare(raw_fod, raw_pet, tab, train_idx, eval_idx):
    fod_train, fod_eval, _ = TRAINER.pca_fit_transform(
        raw_fod[train_idx], raw_fod[eval_idx], raw_fod[eval_idx], 20
    )
    pet_train, pet_eval, _ = TRAINER.pca_fit_transform(
        raw_pet[train_idx], raw_pet[eval_idx], raw_pet[eval_idx], 35
    )
    x_train = np.concatenate([fod_train, pet_train, tab[train_idx]], axis=1)
    x_eval = np.concatenate([fod_eval, pet_eval, tab[eval_idx]], axis=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_eval = scaler.transform(x_eval).astype(np.float32)
    clinical_scaler = StandardScaler()
    clinical_train = clinical_scaler.fit_transform(tab[train_idx]).astype(np.float32)
    clinical_eval = clinical_scaler.transform(tab[eval_idx]).astype(np.float32)
    return x_train, x_eval, clinical_train, clinical_eval


def score_config(fod, pet, tab, y, config, folds):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, eval_idx) in enumerate(splitter.split(fod, y), 1):
        x_train, x_eval, c_train, c_eval = prepare(fod, pet, tab, train_idx, eval_idx)
        _, proba, _ = train_dhnn_proto_classifier(
            x_train, y[train_idx], c_train, x_eval, y[eval_idx], c_eval,
            x_eval, c_eval, config=type(config)(**{**config.__dict__, "seed": 42 + fold}),
            label_names=TRAINER.LABEL_ORDER, device="cpu",
        )
        scores.append(TRAINER.compute_metrics(y[eval_idx], proba.argmax(1), proba)["accuracy"])
    return float(np.mean(scores)), scores


def main():
    participants = pd.read_csv(ROOT / "data" / "bids_clinical" / "participants.tsv", sep="\t")
    splits = [pd.read_csv(ROOT / "data" / "splits" / f"{s}.tsv", sep="\t") for s in ("train", "val", "test")]
    subjects = pd.concat(splits, ignore_index=True)
    fod = np.concatenate([TRAINER.load_feature_dir(ROOT / "features" / "fod", s) for s in ("train", "val", "test")])
    pet = np.concatenate([TRAINER.load_feature_dir(ROOT / "features" / "mri_pet", s) for s in ("train", "val", "test")])
    tab = np.concatenate([TRAINER.align_tabular(participants, s) for s in splits])
    y_all = subjects.diagnosis.map(TRAINER.LABEL_MAP).to_numpy(np.int64)
    rng = np.random.default_rng(110)
    dev_idx, test_idx = [], []
    for cls, n_dev, n_test in [(0, 16, 4), (1, 8, 2), (2, 16, 4)]:
        indices = np.flatnonzero(y_all == cls)
        rng.shuffle(indices)
        dev_idx.extend(indices[:n_dev])
        test_idx.extend(indices[n_dev:n_dev + n_test])
    dev_idx, test_idx = np.array(dev_idx), np.array(test_idx)
    configs = [
        DHNNProtoConfig(lr=8e-4, weight_decay=5e-3, epochs=180, patience=180, knn_k=4, proto_temperature=.12,
                        biomarker_bins=5, class_weights=(1., 1.25, 1.), mci_margin=1.2, mci_margin_weight=.08),
        DHNNProtoConfig(lr=8e-4, weight_decay=5e-3, epochs=180, patience=180, knn_k=4, proto_temperature=.10,
                        biomarker_bins=5, class_weights=(1., 1.25, 1.), mci_margin=1.2, mci_margin_weight=.08),
        DHNNProtoConfig(lr=6e-4, weight_decay=5e-3, epochs=180, patience=180, knn_k=4, proto_temperature=.12,
                        biomarker_bins=5, class_weights=(1., 1.5, 1.), mci_margin=1.2, mci_margin_weight=.05),
    ]
    tuning = []
    for i, config in enumerate(configs, 1):
        mean_accuracy, fold_accuracy = score_config(fod[dev_idx], pet[dev_idx], tab[dev_idx], y_all[dev_idx], config, 5)
        tuning.append({"index": i, "mean_dev_accuracy": mean_accuracy, "fold_accuracy": fold_accuracy, "config": config.__dict__})
        print(tuning[-1], flush=True)
    best = max(tuning, key=lambda item: item["mean_dev_accuracy"])
    config = DHNNProtoConfig(**best["config"])
    metrics, proba, fold_metrics = TRAINER.run_cv_ensemble(
        fod=fod[dev_idx],
        mri_pet=pet[dev_idx],
        tab=tab[dev_idx],
        y=y_all[dev_idx],
        fod_test=fod[test_idx],
        mri_pet_test=pet[test_idx],
        tab_test=tab[test_idx],
        y_test=y_all[test_idx],
        config=config,
        folds=5,
        device="cpu",
    )
    pred = proba.argmax(1)
    payload = {
        "selection": "best mean accuracy on 5-fold development-only tuning",
        "tuning_results": tuning,
        "selected_config": best["config"],
        "test_metrics": metrics,
        "confusion_matrix": confusion_matrix(y_all[test_idx], pred, labels=np.arange(3)).tolist(),
        "fold_metrics": fold_metrics,
        "test_counts": {label: int((y_all[test_idx] == i).sum()) for i, label in enumerate(TRAINER.LABEL_ORDER)},
        "overlap_count": 0,
    }
    (ROOT / "results" / "leakfree_tuned_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
