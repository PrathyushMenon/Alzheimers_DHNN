"""Single release entrypoint for the leak-free multimodal AD pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from dhnn_proto_head import DHNNProtoConfig, train_dhnn_proto_classifier  # noqa: E402

LABELS = ("CN", "MCI", "AD")
LABEL_MAP = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "val", "test")
MODALITIES = ("dmri", "mri", "pet")


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = YAML(typ="safe").load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping.")
    return config


def fail_missing_inputs() -> None:
    raise RuntimeError(
        "Deep encoder inputs are unavailable. Install the required raw-pipeline "
        "tools, provide compatible checkpoints/model modules, or place "
        "data/processed/{dmri,mri,pet}/{train,val,test}_features.npy."
    )


def run_raw_pipeline() -> None:
    runner = ROOT / "code" / "run_pipeline.py"
    if not runner.exists():
        raise FileNotFoundError(f"Raw pipeline runner is missing: {runner}")
    print("Processed embeddings are absent; running the raw-data pipeline.")
    completed = subprocess.run([sys.executable, str(runner)], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "The raw-data pipeline failed. Install ClinicaDL, ANTs, MRtrix3/FSL, "
            "and provide compatible encoder checkpoints/model modules."
        )


def materialize_legacy_features(processed: Path) -> None:
    """Adapt the existing runner's fused feature files to the release layout."""
    legacy_fod = ROOT / "features" / "fod"
    legacy_mri_pet = ROOT / "features" / "mri_pet"
    if not all((legacy_fod / f"{split}_features.npy").exists() for split in SPLITS):
        return
    if not all((legacy_mri_pet / f"{split}_features.npy").exists() for split in SPLITS):
        return
    for modality in MODALITIES:
        (processed / modality).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        np.save(processed / "dmri" / f"{split}_features.npy", np.load(legacy_fod / f"{split}_features.npy"))
        np.save(processed / "mri" / f"{split}_features.npy", np.load(legacy_mri_pet / f"{split}_features.npy"))
        np.save(processed / "pet" / f"{split}_features.npy", np.zeros((np.load(legacy_mri_pet / f"{split}_features.npy").shape[0], 0), dtype=np.float32))


def load_clinical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing clinical metadata: {path}")
    table = pd.read_csv(path)
    lowered = {str(column).strip().lower(): column for column in table.columns}
    label_column = next((lowered[name] for name in ("diagnosis", "label", "group", "class") if name in lowered), None)
    if label_column is None:
        raise ValueError("clinical.csv must contain a diagnosis, label, group, or class column.")
    labels = table[label_column].astype(str).str.upper().str.strip()
    if not labels.isin(LABELS).all():
        raise ValueError(f"clinical labels must be exactly {LABELS}.")

    def numeric(name: str) -> pd.Series:
        if name not in lowered:
            raise ValueError(f"clinical.csv must contain {name}.")
        return pd.to_numeric(table[lowered[name]], errors="coerce")

    result = pd.DataFrame({
        "label": labels.map(LABEL_MAP).to_numpy(dtype=np.int64),
        "age": numeric("age"),
        "sex": table[lowered["sex"]].astype(str).str.lower().str.replace(".0", "", regex=False).map(
            {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "2": 0}
        ),
        "moca": numeric("moca") if "moca" in lowered else numeric("mmse"),
    })
    if result[["age", "sex", "moca"]].isna().any().any():
        raise ValueError("clinical.csv contains missing or invalid age, sex, or moca values.")
    if "split" in lowered:
        result["split"] = table[lowered["split"]].astype(str).str.lower().to_numpy()
    return result


def load_embeddings(processed: Path, clinical: pd.DataFrame):
    features: dict[str, dict[str, np.ndarray]] = {}
    for modality in MODALITIES:
        features[modality] = {}
        for split in SPLITS:
            path = processed / modality / f"{split}_features.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing embedding file: {path}")
            values = np.asarray(np.load(path), dtype=np.float32)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f"{path} must be a finite 2-D array.")
            features[modality][split] = values
    labels: dict[str, np.ndarray] = {}
    clinical_features: dict[str, np.ndarray] = {}
    if "split" in clinical.columns:
        for split in SPLITS:
            mask = clinical["split"] == split
            labels[split] = clinical.loc[mask, "label"].to_numpy()
            clinical_features[split] = clinical.loc[mask, ["age", "sex", "moca"]].to_numpy(dtype=np.float32)
    else:
        lengths = [features["dmri"][split].shape[0] for split in SPLITS]
        if len(clinical) != sum(lengths):
            raise ValueError("clinical.csv must contain a split column or rows ordered train, val, test.")
        offset = 0
        for split, length in zip(SPLITS, lengths):
            rows = clinical.iloc[offset:offset + length]
            labels[split] = rows["label"].to_numpy()
            clinical_features[split] = rows[["age", "sex", "moca"]].to_numpy(dtype=np.float32)
            offset += length
    for split in SPLITS:
        expected = features["dmri"][split].shape[0]
        if any(features[m][split].shape[0] != expected for m in MODALITIES) or labels[split].shape[0] != expected:
            raise ValueError(f"Modality/clinical row mismatch for {split}.")
    return features, labels, clinical_features


def reduce_modalities(features: dict[str, dict[str, np.ndarray]], config: dict[str, Any]) -> dict[str, np.ndarray]:
    combined = {split: np.concatenate([features["mri"][split], features["pet"][split]], axis=1) for split in SPLITS}
    reduced: dict[str, np.ndarray] = {}
    for name, source, target in (("dmri", features["dmri"], 20), ("mri_pet", combined, 35)):
        train = source["train"]
        components = min(target, train.shape[0] - 1, train.shape[1])
        if components != target:
            raise ValueError(f"Training data cannot support the required {target} {name} components.")
        pca = PCA(n_components=components, random_state=int(config.get("random_state", 42)))
        scaler = StandardScaler()
        scaler.fit(pca.fit_transform(train))
        for split in SPLITS:
            reduced[f"{name}_{split}"] = scaler.transform(pca.transform(source[split])).astype(np.float32)
    return reduced


def save_tsne(values: np.ndarray, labels: np.ndarray, path: Path, seed: int) -> None:
    coords = TSNE(n_components=2, init="pca", random_state=seed, perplexity=min(30, max(1, values.shape[0] - 1))).fit_transform(values)
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": [LABELS[i] for i in labels]}).to_csv(path.with_suffix(".csv"), index=False)
    import matplotlib.pyplot as plt
    for index, label in enumerate(LABELS):
        mask = labels == index
        plt.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    config = load_config()
    processed, results = ROOT / "data" / "processed", ROOT / "results"
    plots = results / "plots"
    results.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    has_embeddings = all((processed / m / f"{s}_features.npy").exists() for m in MODALITIES for s in SPLITS)
    if not has_embeddings:
        if not bool(config.get("auto_raw_pipeline", True)):
            fail_missing_inputs()
        run_raw_pipeline()
        materialize_legacy_features(processed)
        has_embeddings = all((processed / m / f"{s}_features.npy").exists() for m in MODALITIES for s in SPLITS)
        if not has_embeddings:
            raise RuntimeError("Raw pipeline completed without producing all data/processed embedding files.")
    clinical = load_clinical(ROOT / "data" / "raw" / "clinical.csv")
    features, labels, clinical_features = load_embeddings(processed, clinical)
    reduced = reduce_modalities(features, config)
    clinical_scaler = StandardScaler().fit(clinical_features["train"])
    clinical_scaled = {s: clinical_scaler.transform(clinical_features[s]).astype(np.float32) for s in SPLITS}
    x = {s: np.concatenate([reduced[f"dmri_{s}"], reduced[f"mri_pet_{s}"], clinical_scaled[s]], axis=1).astype(np.float32) for s in SPLITS}
    if x["train"].shape[1] != 58:
        raise RuntimeError(f"Expected 58-dimensional fusion input, got {x['train'].shape[1]}.")
    seeds = [int(seed) for seed in config.get("ensemble_seeds", [config.get("random_state", 42)])]
    weights = tuple(float(value) for value in config.get("class_weights", [1.0, 1.25, 1.0]))
    probabilities_by_seed, artifacts = [], None
    for seed in seeds:
        model_config = DHNNProtoConfig(epochs=int(config.get("epochs", 250)), patience=int(config.get("patience", 40)), seed=seed, num_classes=3, class_weights=weights, mci_margin=float(config.get("mci_margin", 1.2)), mci_margin_weight=float(config.get("mci_margin_weight", 0.08)), proto_temperature=float(config.get("proto_temperature", 0.12)), knn_k=int(config.get("hypergraph_knn_k", 4)))
        _, probabilities, seed_artifacts = train_dhnn_proto_classifier(x["train"], labels["train"], clinical_scaled["train"], x["val"], labels["val"], clinical_scaled["val"], x["test"], clinical_scaled["test"], model_config, list(LABELS), str(config.get("device", "cpu")))
        probabilities_by_seed.append(probabilities)
        artifacts = artifacts or seed_artifacts
    predictions = np.mean(np.stack(probabilities_by_seed), axis=0).argmax(axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(labels["test"], predictions, labels=np.arange(3), zero_division=0)
    metrics = pd.DataFrame([{"accuracy": accuracy_score(labels["test"], predictions), "balanced_accuracy": balanced_accuracy_score(labels["test"], predictions), "f1_macro": f1_score(labels["test"], predictions, average="macro", zero_division=0), "mcc": matthews_corrcoef(labels["test"], predictions), "cn_precision": precision[0], "cn_recall": recall[0], "mci_precision": precision[1], "mci_recall": recall[1], "ad_precision": precision[2], "ad_recall": recall[2], "ensemble_members": len(seeds)}])
    metrics.to_csv(results / "metrics.csv", index=False)
    (results / "ensemble_metrics.json").write_text(json.dumps({"metrics": metrics.iloc[0].to_dict(), "confusion_matrix": confusion_matrix(labels["test"], predictions, labels=np.arange(3)).tolist(), "seeds": seeds, "train_with_val": False, "class_weights": list(weights)}, indent=2), encoding="utf-8")
    save_tsne(x["test"], labels["test"], plots / "input_tsne.png", seeds[0])
    save_tsne(artifacts["z_test"], labels["test"], plots / "refined_tsne.png", seeds[0])
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
