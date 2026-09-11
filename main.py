"""Single release entrypoint for the leak-free multimodal AD pipeline."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
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


def fail_missing_inputs(weights_present: bool) -> None:
    if weights_present:
        detail = (
            "Weights were found, but no supported encoder adapter is included in "
            "this release. Add pre-extracted embeddings under data/processed/."
        )
    else:
        detail = "Add the compatible Swin-FOD and ALBEF weights under checkpoints/."
    raise RuntimeError(
        f"Deep encoder inputs are unavailable. {detail} You may also place pre-extracted "
        "embeddings under data/processed/ as "
        "data/processed/{dmri,mri,pet}/{train,val,test}_features.npy. "
        "The release pipeline does not silently substitute raw-volume features."
    )


def load_clinical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing clinical metadata: {path}")
    table = pd.read_csv(path)
    lowered = {str(column).strip().lower(): column for column in table.columns}
    label_column = next(
        (lowered[name] for name in ("diagnosis", "label", "group", "class") if name in lowered),
        None,
    )
    if label_column is None:
        raise ValueError("clinical.csv must contain a diagnosis, label, group, or class column.")
    labels = table[label_column].astype(str).str.upper().str.strip()
    if not labels.isin(LABELS).all():
        raise ValueError(f"clinical labels must be exactly {LABELS}.")
    def numeric(name: str) -> pd.Series:
        if name not in lowered:
            raise ValueError(f"clinical.csv must contain {name}.")
        return pd.to_numeric(table[lowered[name]], errors="coerce")

    result = pd.DataFrame(
        {
            "label": labels.map(LABEL_MAP).to_numpy(dtype=np.int64),
            "age": numeric("age"),
            "sex": table[lowered["sex"]].astype(str).str.lower().str.replace(".0", "", regex=False).map(
                {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "2": 0}
            ),
            "moca": numeric("moca") if "moca" in lowered else numeric("mmse"),
        }
    )
    if result[["age", "sex", "moca"]].isna().any().any():
        raise ValueError("clinical.csv contains missing or invalid age, sex, or moca values.")
    if "split" in lowered:
        result["split"] = table[lowered["split"]].astype(str).str.lower().to_numpy()
    return result


def load_embeddings(
    processed: Path, clinical: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    features: dict[str, np.ndarray] = {}
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
            mask = clinical["split"].astype(str).str.lower() == split
            labels[split] = clinical.loc[mask, "label"].to_numpy()
            clinical_features[split] = clinical.loc[mask, ["age", "sex", "moca"]].to_numpy(dtype=np.float32)
    else:
        lengths = [features[MODALITIES[0]][split].shape[0] for split in SPLITS]
        if len(clinical) != sum(lengths):
            raise ValueError(
                "clinical.csv must contain a split column, or rows ordered as train, val, test "
                "with one row per embedding."
            )
        offset = 0
        for split, length in zip(SPLITS, lengths):
            rows = clinical.iloc[offset : offset + length]
            labels[split] = rows["label"].to_numpy()
            clinical_features[split] = rows[["age", "sex", "moca"]].to_numpy(dtype=np.float32)
            offset += length
    for split in SPLITS:
        expected = features[MODALITIES[0]][split].shape[0]
        if labels[split].shape[0] != expected:
            raise ValueError(f"Label/embedding count mismatch for {split}: {labels[split].shape[0]} != {expected}.")
    return features, labels, clinical_features


def reduce_modalities(
    features: dict[str, dict[str, np.ndarray]], config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Fit the intended 20 + 35 modality PCA fusion on training data only."""
    seed = int(config.get("random_state", 42))
    allocations = {"dmri": 20, "mri_pet": 35}
    combined_mri_pet = {
        split: np.concatenate([features["mri"][split], features["pet"][split]], axis=1)
        for split in SPLITS
    }
    sources = {"dmri": features["dmri"], "mri_pet": combined_mri_pet}
    reduced: dict[str, np.ndarray] = {}
    for name, source in sources.items():
        train = source["train"]
        components = min(allocations[name], train.shape[0] - 1, train.shape[1])
        pca = PCA(n_components=components, random_state=seed)
        scaler = StandardScaler()
        transformed_train = pca.fit_transform(train)
        scaler.fit(transformed_train)
        for split in SPLITS:
            reduced[f"{name}_{split}"] = scaler.transform(
                pca.transform(source[split])
            ).astype(np.float32)
    return reduced


def save_tsne(values: np.ndarray, labels: np.ndarray, path: Path, seed: int) -> None:
    perplexity = min(30, max(1, values.shape[0] - 1))
    coords = TSNE(n_components=2, init="pca", random_state=seed, perplexity=perplexity).fit_transform(values)
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": [LABELS[i] for i in labels]}).to_csv(
        path.with_suffix(".csv"), index=False
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save t-SNE scatter plots.") from exc
    for index, label in enumerate(LABELS):
        mask = labels == index
        plt.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    config = load_config()
    processed = ROOT / "data" / "processed"
    results = ROOT / "results"
    plots = results / "plots"
    results.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)

    checkpoint_files = list((ROOT / "checkpoints").rglob("*")) if (ROOT / "checkpoints").exists() else []
    has_embeddings = (processed / "dmri").exists() and (processed / "mri").exists() and (processed / "pet").exists()
    if not has_embeddings and not any(path.is_file() for path in checkpoint_files):
        fail_missing_inputs(False)
    if not has_embeddings:
        fail_missing_inputs(True)

    clinical = load_clinical(ROOT / "data" / "raw" / "clinical.csv")
    features, labels, clinical_features = load_embeddings(processed, clinical)
    reduced = reduce_modalities(features, config)
    clinical_scaler = StandardScaler().fit(clinical_features["train"])
    clinical_scaled = {
        split: clinical_scaler.transform(clinical_features[split]).astype(np.float32)
        for split in SPLITS
    }
    x = {
        split: np.concatenate(
            [reduced[f"dmri_{split}"], reduced[f"mri_pet_{split}"], clinical_scaled[split]],
            axis=1,
        ).astype(np.float32)
        for split in SPLITS
    }
    if x["train"].shape[1] != 58:
        raise RuntimeError(f"Expected 58-dimensional fusion input, got {x['train'].shape[1]}.")
    seeds = [int(seed) for seed in config.get("ensemble_seeds", [config.get("random_state", 42)])]
    if not seeds:
        raise ValueError("ensemble_seeds must contain at least one seed.")
    class_weights = tuple(float(value) for value in config.get("class_weights", [1.0, 1.25, 1.0]))
    probabilities_by_seed = []
    artifacts = None
    for seed in seeds:
        model_config = DHNNProtoConfig(
            epochs=int(config.get("epochs", 250)),
            patience=int(config.get("patience", 40)),
            seed=seed,
            num_classes=3,
            class_weights=class_weights,
            mci_margin=float(config.get("mci_margin", 1.2)),
            mci_margin_weight=float(config.get("mci_margin_weight", 0.08)),
            proto_temperature=float(config.get("proto_temperature", 0.12)),
            knn_k=int(config.get("hypergraph_knn_k", 4)),
        )
        _, probabilities, seed_artifacts = train_dhnn_proto_classifier(
            x["train"], labels["train"], clinical_scaled["train"],
            x["val"], labels["val"], clinical_scaled["val"],
            x["test"], clinical_scaled["test"],
            model_config, list(LABELS), str(config.get("device", "cpu")),
        )
        probabilities_by_seed.append(probabilities)
        if artifacts is None:
            artifacts = seed_artifacts
    probabilities = np.mean(np.stack(probabilities_by_seed, axis=0), axis=0)
    predictions = probabilities.argmax(axis=1)
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            labels["test"], predictions, labels=np.arange(len(LABELS)), zero_division=0
        )
    )
    metrics = pd.DataFrame([{
        "accuracy": accuracy_score(labels["test"], predictions),
        "balanced_accuracy": balanced_accuracy_score(labels["test"], predictions),
        "f1_macro": f1_score(labels["test"], predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels["test"], predictions),
        "cn_precision": per_class_precision[0],
        "cn_recall": per_class_recall[0],
        "mci_precision": per_class_precision[1],
        "mci_recall": per_class_recall[1],
        "ad_precision": per_class_precision[2],
        "ad_recall": per_class_recall[2],
        "ensemble_members": len(seeds),
    }])
    metrics.to_csv(results / "metrics.csv", index=False)
    (results / "ensemble_metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics.iloc[0].to_dict(),
                "confusion_matrix": confusion_matrix(
                    labels["test"], predictions, labels=np.arange(len(LABELS))
                ).tolist(),
                "seeds": seeds,
                "train_with_val": False,
                "class_weights": list(class_weights),
                "prototype_temperature": float(config.get("proto_temperature", 0.12)),
                "hypergraph_knn_k": int(config.get("hypergraph_knn_k", 4)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_tsne(x["test"], labels["test"], plots / "input_tsne.png", model_config.seed)
    save_tsne(artifacts["z_test"], labels["test"], plots / "refined_tsne.png", model_config.seed)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
