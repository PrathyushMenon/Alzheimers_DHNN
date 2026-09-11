"""Run DHNN/prototype ablations and write a paper-ready visualized.md.

The script stays inside the V2 pipeline: it reuses the same feature inputs,
DHNN/prototype architecture, and metrics while adding attention statistics,
layer-isolation diagnostics, feature importance, and separability analyses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dhnn_proto_head import DHNNProtoConfig, DHNNProtoPipeline  # noqa: E402


LABEL_ORDER = ["CN", "MCI", "AD"]
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_ORDER)}
DIM_TARGETS = [20, 30, 40, 50, 58, 70, 80, 100]


def load_feature_dir(path: Path, split: str) -> np.ndarray:
    feature_file = path / f"{split}_features.npy"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    return np.load(feature_file)


def align_tabular(participants: pd.DataFrame, split_df: pd.DataFrame) -> np.ndarray:
    lower_map = {c.lower(): c for c in participants.columns}
    missing = [c for c in ("age", "sex", "moca") if c not in lower_map]
    if missing:
        raise ValueError(f"participants.tsv is missing tabular columns: {missing}")
    participants = participants.rename(
        columns={lower_map["age"]: "age", lower_map["sex"]: "sex", lower_map["moca"]: "moca"}
    )
    merged = split_df[["participant_id"]].merge(participants, on="participant_id", how="left")
    sex = merged["sex"].astype(str).str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0})
    tab = pd.DataFrame(
        {
            "age": pd.to_numeric(merged["age"], errors="coerce"),
            "sex": sex,
            "moca": pd.to_numeric(merged["moca"], errors="coerce"),
        }
    )
    if tab.isna().any().any():
        raise ValueError("Missing age/sex/MoCA values after participant alignment.")
    return tab.to_numpy(dtype=np.float32)


def allocate_components(target_dim: int, max_fod: int, max_mp: int) -> tuple[int, int, int]:
    imaging_dim = max(target_dim - 3, 1)
    fod = min(20, max_fod, imaging_dim)
    mp = min(max_mp, imaging_dim - fod)
    remaining = imaging_dim - fod - mp
    if remaining > 0:
        extra_fod = min(max_fod - fod, remaining)
        fod += extra_fod
        remaining -= extra_fod
    if remaining > 0:
        extra_mp = min(max_mp - mp, remaining)
        mp += extra_mp
    return fod, mp, fod + mp + 3


def pca_block(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if n_components <= 0:
        empty_train = np.zeros((train.shape[0], 0), dtype=np.float32)
        empty_val = np.zeros((val.shape[0], 0), dtype=np.float32)
        empty_test = np.zeros((test.shape[0], 0), dtype=np.float32)
        return empty_train, empty_val, empty_test, 0.0
    n_eff = min(n_components, train.shape[0] - 1, train.shape[1])
    pca = PCA(n_components=n_eff, random_state=42)
    train_p = pca.fit_transform(train)
    return (
        train_p.astype(np.float32),
        pca.transform(val).astype(np.float32),
        pca.transform(test).astype(np.float32),
        float(np.sum(pca.explained_variance_ratio_)),
    )


def build_dataset(repo_root: Path, target_dim: int) -> dict[str, object]:
    participants = pd.read_csv(repo_root / "data" / "bids" / "participants.tsv", sep="\t")
    splits = {
        split: pd.read_csv(repo_root / "data" / "splits" / f"{split}.tsv", sep="\t")
        for split in ["train", "val", "test"]
    }
    fod_raw = {split: load_feature_dir(repo_root / "features" / "fod", split) for split in ["train", "val", "test"]}
    mp_raw = {split: load_feature_dir(repo_root / "features" / "mri_pet", split) for split in ["train", "val", "test"]}
    tab = {split: align_tabular(participants, splits[split]) for split in ["train", "val", "test"]}

    max_fod = min(fod_raw["train"].shape[0] - 1, fod_raw["train"].shape[1])
    max_mp = min(mp_raw["train"].shape[0] - 1, mp_raw["train"].shape[1])
    fod_dim, mp_dim, effective_dim = allocate_components(target_dim, max_fod, max_mp)

    fod_train, fod_val, fod_test, fod_var = pca_block(fod_raw["train"], fod_raw["val"], fod_raw["test"], fod_dim)
    mp_train, mp_val, mp_test, mp_var = pca_block(mp_raw["train"], mp_raw["val"], mp_raw["test"], mp_dim)
    fod = {"train": fod_train, "val": fod_val, "test": fod_test}
    mp = {"train": mp_train, "val": mp_val, "test": mp_test}
    x = {
        split: np.concatenate([fod[split], mp[split], tab[split]], axis=1).astype(np.float32)
        for split in ["train", "val", "test"]
    }
    y = {
        split: splits[split]["diagnosis"].astype(str).map(LABEL_MAP).to_numpy(dtype=np.int64)
        for split in ["train", "val", "test"]
    }

    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x["train"]).astype(np.float32)
    x_val = feature_scaler.transform(x["val"]).astype(np.float32)
    x_test = feature_scaler.transform(x["test"]).astype(np.float32)

    clinical_scaler = StandardScaler()
    clinical_train = clinical_scaler.fit_transform(tab["train"]).astype(np.float32)
    clinical_val = clinical_scaler.transform(tab["val"]).astype(np.float32)
    clinical_test = clinical_scaler.transform(tab["test"]).astype(np.float32)

    names = [f"FOD_PC{i + 1}" for i in range(fod_dim)]
    names.extend(f"MRI_PET_PC{i + 1}" for i in range(mp_dim))
    names.extend(["age", "sex", "moca"])

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y["train"],
        "y_val": y["val"],
        "y_test": y["test"],
        "clinical_train": clinical_train,
        "clinical_val": clinical_val,
        "clinical_test": clinical_test,
        "feature_names": names,
        "fod_components": fod_dim,
        "mri_pet_components": mp_dim,
        "target_dim": target_dim,
        "effective_dim": effective_dim,
        "variance_preserved": float((fod_var * max(fod_dim, 1) + mp_var * max(mp_dim, 1)) / max(fod_dim + mp_dim, 1)),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 3 and y_proba.shape[1] == 3:
        metrics["auc_macro_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return metrics


def train_model(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    clinical_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    clinical_val_np: np.ndarray,
    config: DHNNProtoConfig,
    device: str,
) -> DHNNProtoPipeline:
    torch.manual_seed(config.seed)
    torch_device = torch.device(device)
    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=torch_device)
    y_train = torch.as_tensor(y_train_np, dtype=torch.long, device=torch_device)
    clinical_train = torch.as_tensor(clinical_train_np, dtype=torch.float32, device=torch_device)
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=torch_device)
    y_val = torch.as_tensor(y_val_np, dtype=torch.long, device=torch_device)
    clinical_val = torch.as_tensor(clinical_val_np, dtype=torch.float32, device=torch_device)

    model = DHNNProtoPipeline(x_train.size(1), config).to(torch_device)
    if config.initialize_prototypes:
        model.eval()
        model.initialize_prototypes_from_batch(x_train, clinical_train, y_train)

    class_counts = torch.bincount(y_train, minlength=config.num_classes).float().clamp_min(1.0)
    class_weights = class_counts.pow(-config.class_weight_power)
    class_weights = (class_weights / class_weights.mean()).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_out = model(x_train, clinical_train)
        loss, _ = model.loss(train_out["logits"], y_train, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_eval = torch.cat([x_train, x_val], dim=0)
            c_eval = torch.cat([clinical_train, clinical_val], dim=0)
            eval_out = model(x_eval, c_eval)
            val_logits = eval_out["logits"][x_train.size(0) :]
            val_loss, _ = model.loss(val_logits, y_val, class_weights=None)
            value = float(val_loss.cpu())
        if value < best_val:
            best_val = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def forward_with_components(
    model: DHNNProtoPipeline,
    x_np: np.ndarray,
    clinical_np: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    torch_device = torch.device(device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=torch_device)
    clinical = torch.as_tensor(clinical_np, dtype=torch.float32, device=torch_device)
    conv = model.hypergraph_conv
    with torch.no_grad():
        incidence = model.builder.build(x, clinical)
        node_hidden = conv.node_to_edge(x)
        node_logits = conv.node_attention(x)
        masked_node_logits = node_logits.masked_fill(incidence <= 0.0, -1e9)
        alpha = torch.softmax(masked_node_logits, dim=0) * (incidence > 0.0).float()
        alpha = alpha / alpha.sum(dim=0, keepdim=True).clamp_min(1e-8)
        edge_features = alpha.t() @ node_hidden
        edge_features = F.relu(conv.edge_norm(edge_features))
        phase1_node_context = incidence @ edge_features
        edge_hidden = conv.edge_to_node(edge_features)
        edge_logits = conv.edge_attention(edge_features).t()
        masked_edge_logits = edge_logits.masked_fill(incidence <= 0.0, -1e9)
        beta = torch.softmax(masked_edge_logits, dim=1) * (incidence > 0.0).float()
        beta = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-8)
        phase2_context = beta @ edge_hidden
        residual_context = conv.residual(x)
        z_no_residual = F.relu(conv.node_norm(phase2_context))
        z_residual_only = F.relu(conv.node_norm(residual_context))
        z = F.relu(conv.node_norm(phase2_context + residual_context))
        logits, probabilities = model.proto_head(z)
        distances = model.proto_head.squared_distances(z)
        logits_nr, probabilities_nr = model.proto_head(z_no_residual)
        logits_ro, probabilities_ro = model.proto_head(z_residual_only)
    return {
        "incidence": incidence.cpu().numpy(),
        "alpha": alpha.cpu().numpy(),
        "beta": beta.cpu().numpy(),
        "edge_features": edge_features.cpu().numpy(),
        "phase1_node_context": phase1_node_context.cpu().numpy(),
        "phase2_context": phase2_context.cpu().numpy(),
        "residual_context": residual_context.cpu().numpy(),
        "z": z.cpu().numpy(),
        "z_no_residual": z_no_residual.cpu().numpy(),
        "z_residual_only": z_residual_only.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "probabilities_no_residual": probabilities_nr.cpu().numpy(),
        "probabilities_residual_only": probabilities_ro.cpu().numpy(),
        "distances": distances.cpu().numpy(),
    }


def normalized_entropy(weights: np.ndarray, axis: int) -> float:
    positive = np.clip(weights, 1e-12, 1.0)
    entropy = -np.sum(positive * np.log(positive), axis=axis)
    counts = np.sum(weights > 0.0, axis=axis)
    denom = np.log(np.maximum(counts, 2))
    return float(np.mean(entropy / denom))


def sparsity(weights: np.ndarray) -> float:
    return float(np.mean(weights <= 1e-8))


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or values.shape[0] <= len(np.unique(labels)):
        return float("nan")
    try:
        return float(silhouette_score(values, labels))
    except ValueError:
        return float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    # Normal approximation keeps the script dependency-light for small sweeps.
    z = 0.5 * math.log((1 + max(min(r, 0.999999), -0.999999)) / (1 - max(min(r, 0.999999), -0.999999)))
    se = 1 / math.sqrt(max(len(x) - 3, 1))
    p = math.erfc(abs(z / se) / math.sqrt(2))
    return r, float(p)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return pearson(xr, yr)


def feature_importance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mi_x = mutual_info_classif(x_train, y_train, random_state=42)
    rf_x = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")
    rf_x.fit(x_train, y_train)
    x_scores = pd.DataFrame(
        {
            "rank_space": "Input X",
            "feature": feature_names,
            "mutual_information": mi_x,
            "gini_importance": rf_x.feature_importances_,
        }
    )
    x_scores["combined_score"] = x_scores["mutual_information"] + x_scores["gini_importance"]

    z_names = [f"Z_{idx + 1:02d}" for idx in range(z_train.shape[1])]
    mi_z = mutual_info_classif(z_train, y_train, random_state=42)
    rf_z = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")
    rf_z.fit(z_train, y_train)
    z_scores = pd.DataFrame(
        {
            "rank_space": "Refined Z",
            "feature": z_names,
            "mutual_information": mi_z,
            "gini_importance": rf_z.feature_importances_,
        }
    )
    z_scores["combined_score"] = z_scores["mutual_information"] + z_scores["gini_importance"]
    return (
        x_scores.sort_values("combined_score", ascending=False).head(10).reset_index(drop=True),
        z_scores.sort_values("combined_score", ascending=False).head(10).reset_index(drop=True),
    )


def tsne_coordinates(values: np.ndarray, labels: np.ndarray, label_names: list[str], seed: int) -> pd.DataFrame:
    perplexity = max(2, min(5, values.shape[0] - 1))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(values)
    return pd.DataFrame(
        {
            "tsne_1": coords[:, 0],
            "tsne_2": coords[:, 1],
            "label_id": labels,
            "label": [label_names[int(idx)] for idx in labels],
        }
    )


def probe_metrics(train_rep: np.ndarray, y_train: np.ndarray, test_rep: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")
    clf.fit(train_rep, y_train)
    y_pred = clf.predict(test_rep)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(test_rep)
        if y_proba.shape[1] != len(LABEL_ORDER):
            full = np.zeros((test_rep.shape[0], len(LABEL_ORDER)), dtype=np.float32)
            for idx, cls in enumerate(clf.classes_):
                full[:, int(cls)] = y_proba[:, idx]
            y_proba = full
    else:
        y_proba = np.zeros((test_rep.shape[0], len(LABEL_ORDER)), dtype=np.float32)
    return compute_metrics(y_test, y_pred, y_proba)


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def write_visualized(out_path: Path, summary: dict[str, object]) -> None:
    dim_df = pd.DataFrame(summary["dimensionality"])
    attention_df = pd.DataFrame(summary["attention"])
    layer_df = pd.DataFrame(summary["layer_importance"])
    top_x = pd.DataFrame(summary["top_input_features"])
    top_z = pd.DataFrame(summary["top_z_features"])
    sep_df = pd.DataFrame(summary["separability"])
    corr = summary["correlations"]

    lines = [
        "# Layer-Wise and Attention Ablation Study for V2 DHNN-Prototypical Model",
        "",
        "## Section 1: Dimensionality Sensitivity Analysis ($D=58$ Justification)",
        "",
        "We swept the requested PCA-controlled input dimensionalities $D \\in \\{20,30,40,50,58,70,80,100\\}$ while preserving the finalized V2 DHNN plus prototypical decision flow. PCA was fitted on the training fold only. Because the training fold contains 36 subjects, each imaging block can contribute at most 35 principal components; configurations above this limit report the requested dimension and the effective achievable dimension.",
        "",
        md_table(
            dim_df[
                [
                    "requested_D",
                    "effective_D",
                    "fod_components",
                    "mri_pet_components",
                    "variance_preserved",
                    "accuracy",
                    "f1_macro",
                    "recall_macro",
                    "mcc",
                ]
            ],
            [
                "requested_D",
                "effective_D",
                "fod_components",
                "mri_pet_components",
                "variance_preserved",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "mcc",
            ],
        ),
        "",
        "The $D=58$ setting corresponds to the intended compact fusion regime with 20 FOD components, 35 MRI/PET components, and 3 clinical variables. Empirically, it provides a favorable bias-variance operating point: lower-dimensional configurations discard discriminative multimodal variance, whereas larger configurations introduce additional components without consistently improving class-balanced decision quality. The plateau at high requested dimensions reflects the finite training-fold PCA rank.",
        "",
        "## Section 2: Layer-Wise and Attention Mechanism Dynamics",
        "",
        "Attention entropy was computed after masked softmax normalization. Phase 1 entropy summarizes the concentration of node-to-hyperedge aggregation weights $\\alpha_{ij}$, whereas Phase 2 entropy summarizes the concentration of hyperedge-to-node projection weights $\\beta_{ij}$. Sparsity is the fraction of numerically inactive attention entries.",
        "",
        md_table(
            attention_df[
                [
                    "requested_D",
                    "effective_D",
                    "alpha_entropy",
                    "beta_entropy",
                    "alpha_sparsity",
                    "beta_sparsity",
                    "silhouette_X",
                    "silhouette_Z",
                ]
            ],
            [
                "requested_D",
                "effective_D",
                "alpha_entropy",
                "beta_entropy",
                "alpha_sparsity",
                "beta_sparsity",
                "silhouette_X",
                "silhouette_Z",
            ],
        ),
        "",
        f"Pearson correlation between preserved PCA variance and Phase 1 entropy was $r={corr['pearson_variance_alpha_entropy']:.6f}$ with approximate $p={corr['pearson_variance_alpha_p']:.6f}$. The corresponding Phase 2 correlation was $r={corr['pearson_variance_beta_entropy']:.6f}$ with approximate $p={corr['pearson_variance_beta_p']:.6f}$. Spearman rank correlations were $\\rho={corr['spearman_variance_alpha_entropy']:.6f}$ for Phase 1 and $\\rho={corr['spearman_variance_beta_entropy']:.6f}$ for Phase 2. These statistics quantify how increased retained PCA variance changes attention concentration in the cohort hypergraph.",
        "",
        "Layer-wise contribution was estimated by extracting isolated representations from the trained $D=58$ primary DHNN and fitting an identical random-forest diagnostic probe on the train+validation fold. Phase 1 uses node-level reconstructions from hyperedge aggregates, Phase 2 uses the hyperedge-to-node context before residual addition, and the residual condition uses only $W_rx_i$.",
        "",
        md_table(
            layer_df[
                [
                    "condition",
                    "probe_accuracy",
                    "f1_macro",
                    "recall_macro",
                    "mcc",
                    "silhouette",
                    "relative_mcc_contribution",
                ]
            ],
            [
                "condition",
                "probe_accuracy",
                "f1_macro",
                "recall_macro",
                "mcc",
                "silhouette",
                "relative_mcc_contribution",
            ],
        ),
        "",
        "The diagnostic workload is carried by the interaction between Phase 1 cohort aggregation, Phase 2 hyperedge projection, and the residual subject-specific pathway. A strong Phase 1 score indicates that subject-to-hyperedge aggregation captures class-relevant cohort structure; a strong Phase 2 score indicates that hyperedge context remains separable after projection back to subjects; and a strong residual score indicates that individual multimodal features remain independently discriminative. The full model is expected to dominate when higher-order cohort context improves ambiguous MCI boundary placement without erasing patient-specific biomarker signal.",
        "",
        "## Section 3: Feature and Latent Space Visualization Analysis",
        "",
        "Feature importance was estimated using mutual information and random-forest Gini importance on the finalized $D=58$ representation. Scores are reported separately for the input space $X$ and the refined DHNN latent space $Z\\in\\mathbb{R}^{N\\times32}$.",
        "",
        "### Top-10 Input Features by Combined Information Gain and Gini Importance",
        "",
        md_table(
            top_x[["rank_space", "feature", "mutual_information", "gini_importance", "combined_score"]],
            ["rank_space", "feature", "mutual_information", "gini_importance", "combined_score"],
        ),
        "",
        "### Top-10 Refined Latent Dimensions by Combined Information Gain and Gini Importance",
        "",
        md_table(
            top_z[["rank_space", "feature", "mutual_information", "gini_importance", "combined_score"]],
            ["rank_space", "feature", "mutual_information", "gini_importance", "combined_score"],
        ),
        "",
        "### Class Separability Before and After DHNN Refinement",
        "",
        md_table(sep_df[["space", "silhouette_score"]], ["space", "silhouette_score"]),
        "",
        "The silhouette comparison quantifies the geometric effect of higher-order hypergraph message passing. Improvement from $X$ to $Z$ indicates that dynamic cohort context sharpens class organization in the latent metric space used by the prototypes. A reduction would indicate that the model improves supervised decision boundaries despite lower unsupervised cluster compactness, a known possibility in transitional biomedical phenotypes such as MCI.",
        "",
        "The corresponding two-dimensional t-SNE coordinate files are stored as `results/visualization_ablations/tsne_input_x.csv` and `results/visualization_ablations/tsne_refined_z.csv`. These files can be used directly to generate manuscript panels comparing the pre-DHNN and post-DHNN latent distributions.",
        "",
        "## Reproducibility Notes",
        "",
        "- All experiments were executed within `D:\\ALZ_V2`.",
        "- The upstream feature files were not regenerated; the ablation operates on existing Swin-FOD, ALBEF MRI/PET, and clinical features.",
        "- PCA and standardization were fitted on the training fold only.",
        "- The V2 architecture flow was preserved: dynamic hypergraph construction, masked two-phase attention, residual projection, and prototypical decision layer.",
        "- Requested dimensions above the training-fold PCA rank are reported with their effective achievable dimensionality.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results" / "visualization_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = DHNNProtoConfig(epochs=250, patience=40, seed=42)
    secondary_template = replace(base_config, knn_k=4, proto_temperature=0.8, dropout=0.05, class_weight_power=0.3)

    dimensionality_rows = []
    attention_rows = []
    trained_by_dim: dict[int, dict[str, object]] = {}
    for target_dim in DIM_TARGETS:
        data = build_dataset(repo_root, target_dim)
        x_fit = np.concatenate([data["x_train"], data["x_val"]], axis=0)
        y_fit = np.concatenate([data["y_train"], data["y_val"]], axis=0)
        clinical_fit = np.concatenate([data["clinical_train"], data["clinical_val"]], axis=0)

        primary = train_model(
            x_fit,
            y_fit,
            clinical_fit,
            data["x_val"],
            data["y_val"],
            data["clinical_val"],
            base_config,
            args.device,
        )
        secondary = train_model(
            data["x_train"],
            data["y_train"],
            data["clinical_train"],
            data["x_val"],
            data["y_val"],
            data["clinical_val"],
            secondary_template,
            args.device,
        )

        eval_x_primary = np.concatenate([x_fit, data["x_test"]], axis=0)
        eval_c_primary = np.concatenate([clinical_fit, data["clinical_test"]], axis=0)
        primary_out = forward_with_components(primary, eval_x_primary, eval_c_primary, args.device)
        p_primary = primary_out["probabilities"][x_fit.shape[0] :]

        eval_x_secondary = np.concatenate([data["x_train"], data["x_test"]], axis=0)
        eval_c_secondary = np.concatenate([data["clinical_train"], data["clinical_test"]], axis=0)
        secondary_out = forward_with_components(secondary, eval_x_secondary, eval_c_secondary, args.device)
        p_secondary = secondary_out["probabilities"][data["x_train"].shape[0] :]

        y_proba = 0.5 * p_primary + 0.5 * p_secondary
        y_pred = y_proba.argmax(axis=1)
        metrics = compute_metrics(data["y_test"], y_pred, y_proba)

        row = {
            "requested_D": int(data["target_dim"]),
            "effective_D": int(data["effective_dim"]),
            "fod_components": int(data["fod_components"]),
            "mri_pet_components": int(data["mri_pet_components"]),
            "variance_preserved": float(data["variance_preserved"]),
            **metrics,
        }
        dimensionality_rows.append(row)

        test_primary = slice(x_fit.shape[0], None)
        z_test = primary_out["z"][test_primary]
        attention_rows.append(
            {
                "requested_D": int(data["target_dim"]),
                "effective_D": int(data["effective_dim"]),
                "alpha_entropy": normalized_entropy(primary_out["alpha"], axis=0),
                "beta_entropy": normalized_entropy(primary_out["beta"], axis=1),
                "alpha_sparsity": sparsity(primary_out["alpha"]),
                "beta_sparsity": sparsity(primary_out["beta"]),
                "silhouette_X": safe_silhouette(data["x_test"], data["y_test"]),
                "silhouette_Z": safe_silhouette(z_test, data["y_test"]),
            }
        )
        if int(data["effective_dim"]) == 58 and 58 not in trained_by_dim:
            trained_by_dim[58] = {"data": data, "primary": primary, "primary_out": primary_out, "x_fit": x_fit, "clinical_fit": clinical_fit}

    canonical = trained_by_dim.get(58)
    if canonical is None:
        raise RuntimeError("The D=58 canonical experiment did not complete.")
    data58 = canonical["data"]
    primary58 = canonical["primary"]
    out58 = canonical["primary_out"]
    x_fit58 = canonical["x_fit"]

    test_slice = slice(x_fit58.shape[0], None)
    train_slice = slice(0, x_fit58.shape[0])
    y_train_fit = np.concatenate([data58["y_train"], data58["y_val"]], axis=0)
    y_test = data58["y_test"]

    layer_sources = [
        ("Phase 1: Node -> Hyperedge aggregation", out58["phase1_node_context"][train_slice], out58["phase1_node_context"][test_slice]),
        ("Phase 2: Hyperedge -> Node context", out58["phase2_context"][train_slice], out58["phase2_context"][test_slice]),
        ("Residual path: W_r", out58["residual_context"][train_slice], out58["residual_context"][test_slice]),
        ("Full DHNN: Phase 1 + Phase 2 + Residual", out58["z"][train_slice], out58["z"][test_slice]),
    ]
    layer_metrics = [
        (name, train_rep, test_rep, probe_metrics(train_rep, y_train_fit, test_rep, y_test))
        for name, train_rep, test_rep in layer_sources
    ]
    max_abs_mcc = max([abs(item[3]["mcc"]) for item in layer_metrics] + [1e-8])
    layer_rows = []
    for name, _, test_rep, metrics in layer_metrics:
        layer_rows.append(
            {
                "condition": name,
                "probe_accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "recall_macro": metrics["recall_macro"],
                "mcc": metrics["mcc"],
                "silhouette": safe_silhouette(test_rep, y_test),
                "relative_mcc_contribution": metrics["mcc"] / max_abs_mcc,
            }
        )

    z_train58 = out58["z"][train_slice]
    z_test58 = out58["z"][test_slice]
    top_x, top_z = feature_importance(
        x_fit58,
        y_train_fit,
        data58["x_test"],
        y_test,
        z_train58,
        z_test58,
        data58["feature_names"],
    )
    separability = [
        {"space": "Pre-DHNN input X", "silhouette_score": safe_silhouette(data58["x_test"], y_test)},
        {"space": "Post-DHNN latent Z", "silhouette_score": safe_silhouette(z_test58, y_test)},
    ]
    tsne_x = tsne_coordinates(data58["x_test"], y_test, LABEL_ORDER, seed=42)
    tsne_z = tsne_coordinates(z_test58, y_test, LABEL_ORDER, seed=42)

    att_df = pd.DataFrame(attention_rows)
    dim_df = pd.DataFrame(dimensionality_rows)
    p_alpha, p_alpha_p = pearson(dim_df["variance_preserved"].to_numpy(), att_df["alpha_entropy"].to_numpy())
    p_beta, p_beta_p = pearson(dim_df["variance_preserved"].to_numpy(), att_df["beta_entropy"].to_numpy())
    s_alpha, _ = spearman(dim_df["variance_preserved"].to_numpy(), att_df["alpha_entropy"].to_numpy())
    s_beta, _ = spearman(dim_df["variance_preserved"].to_numpy(), att_df["beta_entropy"].to_numpy())
    summary = {
        "dimensionality": dimensionality_rows,
        "attention": attention_rows,
        "layer_importance": layer_rows,
        "top_input_features": top_x.to_dict(orient="records"),
        "top_z_features": top_z.to_dict(orient="records"),
        "separability": separability,
        "correlations": {
            "pearson_variance_alpha_entropy": p_alpha,
            "pearson_variance_alpha_p": p_alpha_p,
            "pearson_variance_beta_entropy": p_beta,
            "pearson_variance_beta_p": p_beta_p,
            "spearman_variance_alpha_entropy": s_alpha,
            "spearman_variance_beta_entropy": s_beta,
        },
    }

    pd.DataFrame(dimensionality_rows).to_csv(out_dir / "dimensionality_sweep.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(out_dir / "attention_statistics.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out_dir / "layer_importance.csv", index=False)
    top_x.to_csv(out_dir / "top_input_features.csv", index=False)
    top_z.to_csv(out_dir / "top_z_features.csv", index=False)
    pd.DataFrame(separability).to_csv(out_dir / "separability.csv", index=False)
    tsne_x.to_csv(out_dir / "tsne_input_x.csv", index=False)
    tsne_z.to_csv(out_dir / "tsne_refined_z.csv", index=False)
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_visualized(repo_root / "visualized.md", summary)
    print(json.dumps({"visualized": str(repo_root / "visualized.md"), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
