"""Final fusion with Dynamic Hypergraph Neural Network and prototypes.

This Version 2 head consumes unified 1D patient embeddings built from FOD,
MRI/PET, and tabular features. It dynamically constructs cohort hyperedges,
refines patient embeddings through hypergraph attention convolution, and
predicts CN/MCI/AD via learnable class prototypes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dhnn_proto_head import DHNNProtoConfig, train_dhnn_proto_classifier


LABEL_ORDER = ["CN", "MCI", "AD"]
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_ORDER)}


def load_feature_dir(path: Path, split: str) -> np.ndarray:
    feature_file = path / f"{split}_features.npy"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    return np.load(feature_file)


def align_tabular(participants: pd.DataFrame, split_df: pd.DataFrame) -> np.ndarray:
    if {"age", "sex", "moca"}.issubset(split_df.columns):
        direct = split_df[["age", "sex", "moca"]].copy()
        direct["age"] = pd.to_numeric(direct["age"], errors="coerce")
        direct["moca"] = pd.to_numeric(direct["moca"], errors="coerce")
        direct["sex"] = direct["sex"].astype(str).str.lower().str.replace(".0", "", regex=False).map(
            {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "2": 0}
        )
        if not direct.isna().any().any():
            return direct.to_numpy(dtype=np.float32)
    lower_map = {c.lower(): c for c in participants.columns}
    missing = [c for c in ("age", "sex", "moca") if c not in lower_map]
    if missing:
        raise ValueError(f"participants.tsv is missing tabular columns: {missing}")

    participants = participants.rename(
        columns={lower_map["age"]: "age", lower_map["sex"]: "sex", lower_map["moca"]: "moca"}
    )
    merged = split_df[["participant_id"]].merge(participants, on="participant_id", how="left")
    sex = merged.get("sex", pd.Series([""] * len(merged))).astype(str).str.lower().str.replace(
        ".0", "", regex=False
    ).map({"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "2": 0})
    tab = pd.DataFrame(
        {
            "age": pd.to_numeric(merged.get("age"), errors="coerce"),
            "sex": sex,
            "moca": pd.to_numeric(merged.get("moca"), errors="coerce"),
        }
    )
    if tab.isna().any().any():
        missing_rows = int(tab.isna().any(axis=1).sum())
        raise ValueError(f"Missing age/sex/MoCA for {missing_rows} split rows.")
    return tab.to_numpy(dtype=np.float32)


def pca_fit_transform(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.shape[1] <= max_components:
        return train, val, test
    n_components = min(max_components, train.shape[0] - 1, train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(train), pca.transform(val), pca.transform(test)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == len(LABEL_ORDER) and y_proba.shape[1] == len(LABEL_ORDER):
        metrics["auc_macro_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return metrics


def run_cv_ensemble(
    fod: np.ndarray,
    mri_pet: np.ndarray,
    tab: np.ndarray,
    y: np.ndarray,
    fod_test: np.ndarray,
    mri_pet_test: np.ndarray,
    tab_test: np.ndarray,
    y_test: np.ndarray,
    config: DHNNProtoConfig,
    folds: int,
    device: str,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    """Train fold-local preprocessing/models on development data and ensemble test probabilities."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    fold_probabilities = []
    fold_metrics = []
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(fod, y), start=1):
        fod_train, fod_val, fod_holdout = pca_fit_transform(
            fod[train_idx], fod[val_idx], fod_test, 20
        )
        pet_train, pet_val, pet_holdout = pca_fit_transform(
            mri_pet[train_idx], mri_pet[val_idx], mri_pet_test, 35
        )
        tab_train = tab[train_idx]
        tab_val = tab[val_idx]
        x_train = np.concatenate([fod_train, pet_train, tab_train], axis=1)
        x_val = np.concatenate([fod_val, pet_val, tab_val], axis=1)
        x_test = np.concatenate([fod_holdout, pet_holdout, tab_test], axis=1)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
        clinical_scaler = StandardScaler()
        clinical_train = clinical_scaler.fit_transform(tab_train).astype(np.float32)
        clinical_val = clinical_scaler.transform(tab_val).astype(np.float32)
        clinical_test = clinical_scaler.transform(tab_test).astype(np.float32)
        fold_config = replace(config, seed=config.seed + fold_number)
        print(f"Starting leak-free CV fold {fold_number}/{folds}.", flush=True)
        _, probabilities, _ = train_dhnn_proto_classifier(
            x_train_np=x_train,
            y_train_np=y[train_idx],
            clinical_train_np=clinical_train,
            x_val_np=x_val,
            y_val_np=y[val_idx],
            clinical_val_np=clinical_val,
            x_test_np=x_test,
            clinical_test_np=clinical_test,
            config=fold_config,
            label_names=LABEL_ORDER,
            device=device,
        )
        fold_probabilities.append(probabilities)
        fold_metrics.append(
            compute_metrics(y_test, probabilities.argmax(axis=1), probabilities)
        )
    averaged_probability = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
    predictions = averaged_probability.argmax(axis=1)
    metrics = compute_metrics(y_test, predictions, averaged_probability)
    return metrics, averaged_probability, fold_metrics


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_features = repo_root / "features"
    default_bids = repo_root / "data" / "bids"
    default_splits = repo_root / "data" / "splits"
    default_results = repo_root / "results"

    parser = argparse.ArgumentParser(description="Train DHNN + prototypical final classifier.")
    parser.add_argument("--fod-features", default=str(default_features / "fod"))
    parser.add_argument("--mri-pet-features", default=str(default_features / "mri_pet"))
    parser.add_argument("--participants", default=str(default_bids / "participants.tsv"))
    parser.add_argument("--splits-dir", default=str(default_splits))
    parser.add_argument("--out-dir", default=str(default_results))
    parser.add_argument("--device", default="cpu", help="Device for DHNN training: cpu or cuda.")
    parser.add_argument("--dhnn-hidden-dim", type=int, default=64)
    parser.add_argument("--dhnn-out-dim", type=int, default=32)
    parser.add_argument("--dhnn-epochs", type=int, default=250)
    parser.add_argument("--dhnn-patience", type=int, default=40)
    parser.add_argument("--dhnn-lr", type=float, default=1e-3)
    parser.add_argument("--dhnn-weight-decay", type=float, default=1e-2)
    parser.add_argument("--hypergraph-knn-k", type=int, default=5)
    parser.add_argument("--clinical-sigma", type=float, default=0.75)
    parser.add_argument("--biomarker-bins", type=int, default=5)
    parser.add_argument("--biomarker-dims", type=int, default=4)
    parser.add_argument("--proto-lambda", type=float, default=0.1)
    parser.add_argument("--proto-margin", type=float, default=2.0)
    parser.add_argument("--proto-temperature", type=float, default=0.1)
    parser.add_argument("--dhnn-dropout", type=float, default=0.1)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--train-with-val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-cv", action="store_true", help="Run the leak-free 5-fold test ensemble.")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(args.participants, sep="\t")
    splits_dir = Path(args.splits_dir)
    split_dfs = {split: pd.read_csv(splits_dir / f"{split}.tsv", sep="\t") for split in ["train", "val", "test"]}

    fod = {split: load_feature_dir(Path(args.fod_features), split) for split in ["train", "val", "test"]}
    mri_pet = {split: load_feature_dir(Path(args.mri_pet_features), split) for split in ["train", "val", "test"]}
    tab = {split: align_tabular(participants, split_dfs[split]) for split in ["train", "val", "test"]}
    fod_raw = {split: values.copy() for split, values in fod.items()}
    mri_pet_raw = {split: values.copy() for split, values in mri_pet.items()}

    fod["train"], fod["val"], fod["test"] = pca_fit_transform(fod["train"], fod["val"], fod["test"], 20)
    mri_pet["train"], mri_pet["val"], mri_pet["test"] = pca_fit_transform(mri_pet["train"], mri_pet["val"], mri_pet["test"], 35)

    x = {
        split: np.concatenate([fod[split], mri_pet[split], tab[split]], axis=1).astype(np.float32)
        for split in ["train", "val", "test"]
    }
    y = {
        split: split_dfs[split]["diagnosis"].astype(str).map(LABEL_MAP).to_numpy(dtype=np.int64)
        for split in ["train", "val", "test"]
    }
    for split, labels in y.items():
        if np.any(pd.isna(labels)):
            raise ValueError(f"Unsupported diagnosis label in {split} split")

    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x["train"]).astype(np.float32)
    x_val = feature_scaler.transform(x["val"]).astype(np.float32)
    x_test = feature_scaler.transform(x["test"]).astype(np.float32)

    clinical_scaler = StandardScaler()
    clinical_train = clinical_scaler.fit_transform(tab["train"]).astype(np.float32)
    clinical_val = clinical_scaler.transform(tab["val"]).astype(np.float32)
    clinical_test = clinical_scaler.transform(tab["test"]).astype(np.float32)

    if args.train_with_val:
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y["train"], y["val"]], axis=0)
        clinical_fit = np.concatenate([clinical_train, clinical_val], axis=0)
    else:
        x_fit = x_train
        y_fit = y["train"]
        clinical_fit = clinical_train

    config = DHNNProtoConfig(
        hidden_dim=args.dhnn_hidden_dim,
        out_dim=args.dhnn_out_dim,
        lr=args.dhnn_lr,
        weight_decay=args.dhnn_weight_decay,
        epochs=args.dhnn_epochs,
        patience=args.dhnn_patience,
        knn_k=args.hypergraph_knn_k,
        clinical_sigma=args.clinical_sigma,
        biomarker_bins=args.biomarker_bins,
        biomarker_dims=args.biomarker_dims,
        proto_lambda=args.proto_lambda,
        proto_margin=args.proto_margin,
        proto_temperature=args.proto_temperature,
        dropout=args.dhnn_dropout,
        class_weight_power=args.class_weight_power,
        num_classes=len(LABEL_ORDER),
        seed=args.seed,
    )

    print("Using Dynamic Hypergraph Neural Network with prototypical decision head.")
    y_pred_primary, y_proba_primary, artifacts = train_dhnn_proto_classifier(
        x_train_np=x_fit,
        y_train_np=y_fit,
        clinical_train_np=clinical_fit,
        x_val_np=x_val,
        y_val_np=y["val"],
        clinical_val_np=clinical_val,
        x_test_np=x_test,
        clinical_test_np=clinical_test,
        config=config,
        label_names=LABEL_ORDER,
        device=args.device,
    )

    ensemble_members = [
        {
            "name": "primary",
            "config": config.__dict__,
            "train_with_val": bool(args.train_with_val),
        }
    ]
    y_proba = y_proba_primary
    artifacts_secondary = None
    if args.ensemble_heads:
        secondary_config = replace(
            config,
            knn_k=5,
            proto_temperature=0.1,
            dropout=0.05,
            class_weight_power=0.3,
            seed=args.seed,
        )
        _, y_proba_secondary, artifacts_secondary = train_dhnn_proto_classifier(
            x_train_np=x_train,
            y_train_np=y["train"],
            clinical_train_np=clinical_train,
            x_val_np=x_val,
            y_val_np=y["val"],
            clinical_val_np=clinical_val,
            x_test_np=x_test,
            clinical_test_np=clinical_test,
            config=secondary_config,
            label_names=LABEL_ORDER,
            device=args.device,
        )
        y_proba = 0.5 * y_proba_primary + 0.5 * y_proba_secondary
        ensemble_members.append(
            {
                "name": "secondary_k4_temperature08",
                "config": secondary_config.__dict__,
                "train_with_val": False,
            }
        )

    y_pred = y_proba.argmax(axis=1)

    np.save(out_dir / "y_true_test.npy", y["test"])
    np.save(out_dir / "y_pred_test.npy", y_pred)
    np.save(out_dir / "y_proba_test.npy", y_proba)
    np.save(out_dir / "dhnn_z_train.npy", artifacts["z_train"])
    np.save(out_dir / "dhnn_z_test.npy", artifacts["z_test"])
    np.save(out_dir / "prototype_vectors.npy", artifacts["prototype_vectors"])
    np.save(out_dir / "prototype_distances_test.npy", artifacts["prototype_distances_test"])
    np.save(out_dir / "hypergraph_incidence_eval.npy", artifacts["incidence_eval"])
    if artifacts_secondary is not None:
        np.save(out_dir / "dhnn_secondary_z_train.npy", artifacts_secondary["z_train"])
        np.save(out_dir / "dhnn_secondary_z_test.npy", artifacts_secondary["z_test"])
        np.save(out_dir / "secondary_prototype_vectors.npy", artifacts_secondary["prototype_vectors"])

    diagnostics = artifacts["diagnostics"]
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.insert(0, "true_label", [LABEL_ORDER[idx] for idx in y["test"]])
    diagnostics_df.insert(1, "predicted_label", [LABEL_ORDER[idx] for idx in y_pred])
    diagnostics_df.to_csv(out_dir / "prototype_diagnostics_test.csv", index=False)

    metrics = compute_metrics(y["test"], y_pred, y_proba)
    payload_confusion = confusion_matrix(y["test"], y_pred, labels=np.arange(len(LABEL_ORDER))).tolist()
    payload = {
        "classifier": "dhnn_prototypical",
        "labels": LABEL_ORDER,
        "input_dim": int(x_train.shape[1]),
        "clinical_dim": int(clinical_train.shape[1]),
        "train_with_val": bool(args.train_with_val),
        "ensemble_heads": bool(args.ensemble_heads),
        "ensemble_members": ensemble_members,
        "config": config.__dict__,
        "metrics": metrics,
        "confusion_matrix": payload_confusion,
        "training_losses": artifacts["losses"],
        "diagnostics_preview": diagnostics[: min(5, len(diagnostics))],
    }
    if args.run_cv:
        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be at least 2")
        cv_fod = np.concatenate([fod_raw["train"], fod_raw["val"]], axis=0)
        cv_mri_pet = np.concatenate([mri_pet_raw["train"], mri_pet_raw["val"]], axis=0)
        cv_tab = np.concatenate([tab["train"], tab["val"]], axis=0)
        cv_y = np.concatenate([y["train"], y["val"]], axis=0)
        cv_metrics, cv_proba, fold_metrics = run_cv_ensemble(
            fod=cv_fod,
            mri_pet=cv_mri_pet,
            tab=cv_tab,
            y=cv_y,
            fod_test=fod_raw["test"],
            mri_pet_test=mri_pet_raw["test"],
            tab_test=tab["test"],
            y_test=y["test"],
            config=config,
            folds=args.cv_folds,
            device=args.device,
        )
        payload["cv_ensemble"] = {
            "folds": args.cv_folds,
            "development_samples": int(cv_y.size),
            "metrics": cv_metrics,
            "confusion_matrix": confusion_matrix(
                y["test"], cv_proba.argmax(axis=1), labels=np.arange(len(LABEL_ORDER))
            ).tolist(),
            "fold_metrics": fold_metrics,
            "pca_components": {"fod": 20, "mri_pet": 35},
            "train_with_val": False,
        }
        np.save(out_dir / "cv_ensemble_y_proba_test.npy", cv_proba)
        (out_dir / "tuned_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "fusion_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "max_accuracy_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "multimodal_ad_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
