from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DHNNProtoConfig:
    hidden_dim: int = 64
    out_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    patience: int = 40
    knn_k: int = 6
    clinical_sigma: float = 0.75
    biomarker_bins: int = 3
    biomarker_dims: int = 4
    proto_lambda: float = 0.1
    proto_margin: float = 2.0
    proto_temperature: float = 1.0
    dropout: float = 0.1
    class_weight_power: float = 0.5
    initialize_prototypes: bool = True
    num_classes: int = 3
    seed: int = 42
    class_weights: tuple[float, float, float] = (1.0, 2.5, 1.0)
    mci_margin: float = 1.5
    mci_margin_weight: float = 0.2
    modality_dims: tuple[int, int] = (20, 35)


class DynamicHypergraphBuilder:
    """Builds dynamic soft incidence matrices from node and clinical features."""

    def __init__(
        self,
        knn_k: int = 6,
        clinical_sigma: float = 0.75,
        biomarker_bins: int = 3,
        biomarker_dims: int = 4,
        modality_dims: tuple[int, int] = (20, 35),
    ):
        self.knn_k = knn_k
        self.clinical_sigma = clinical_sigma
        self.biomarker_bins = biomarker_bins
        self.biomarker_dims = biomarker_dims
        self.modality_dims = modality_dims

    def _soft_knn_incidence(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D_in] patient embeddings -> H_knn: [N, N] patient-centered hyperedges.
        n_nodes = x.size(0)
        x_norm = F.normalize(x, p=2, dim=1)  # [N, D_in]
        sim = x_norm @ x_norm.t()  # [N, N]
        k_eff = min(max(self.knn_k, 1), n_nodes)
        values, indices = torch.topk(sim, k=k_eff, dim=1)  # each: [N, k_eff]
        weights = F.softmax(values, dim=1)  # [N, k_eff]
        incidence = torch.zeros(n_nodes, n_nodes, dtype=x.dtype, device=x.device)  # [N, M_knn=N]
        incidence.scatter_(0, indices.t(), weights.t())
        return incidence.clamp_min(0.0)

    def _soft_bin_incidence(self, values: torch.Tensor, n_bins: int) -> torch.Tensor:
        # values: [N, 1] -> H_bins: [N, n_bins].
        if values.numel() == 0:
            return values.new_zeros(values.size(0), 0)
        v_min = values.min(dim=0, keepdim=True).values  # [1, 1]
        v_max = values.max(dim=0, keepdim=True).values  # [1, 1]
        centers = torch.linspace(0.0, 1.0, n_bins, device=values.device, dtype=values.dtype).view(1, n_bins)
        scaled = (values - v_min) / (v_max - v_min).clamp_min(1e-6)  # [N, 1]
        logits = -((scaled - centers) ** 2) / (2.0 * self.clinical_sigma * self.clinical_sigma)  # [N, n_bins]
        return F.softmax(logits, dim=1)

    def _clinical_incidence(self, clinical: torch.Tensor | None) -> torch.Tensor:
        # clinical: [N, C_clinical] where columns are age, sex, MoCA after scaling.
        if clinical is None or clinical.numel() == 0:
            return torch.empty(0, 0, device=clinical.device if clinical is not None else None)
        columns = [clinical[:, idx : idx + 1] for idx in range(clinical.size(1))]
        return torch.cat([self._soft_bin_incidence(col, self.biomarker_bins) for col in columns], dim=1)

    def _biomarker_proxy_incidence(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D_in]. Early projected components act as final-stage proxies for PET/atrophy patterns.
        dims = min(self.biomarker_dims, x.size(1))
        if dims <= 0:
            return x.new_zeros(x.size(0), 0)
        columns = [x[:, idx : idx + 1] for idx in range(dims)]
        return torch.cat([self._soft_bin_incidence(col, self.biomarker_bins) for col in columns], dim=1)

    def _modality_incidence(self, x: torch.Tensor) -> torch.Tensor:
        """Build separate k-NN and biomarker edges for each imaging modality."""
        parts = []
        offset = 0
        for width in self.modality_dims:
            end = min(offset + width, x.size(1))
            if end > offset:
                modality = x[:, offset:end]
                parts.extend(
                    [
                        self._soft_knn_incidence(modality),
                        self._biomarker_proxy_incidence(modality),
                    ]
                )
            offset += width
        if not parts:
            return x.new_zeros(x.size(0), 0)
        return torch.cat(parts, dim=1)

    def build(self, x: torch.Tensor, clinical: torch.Tensor | None) -> torch.Tensor:
        # x: [N, D_in], clinical: [N, C_clinical] -> incidence H: [N, M].
        incidence_parts = [self._modality_incidence(x)]
        if clinical is not None and clinical.numel() > 0:
            incidence_parts.append(self._clinical_incidence(clinical))
        incidence = torch.cat(incidence_parts, dim=1)  # [N, M]
        column_mass = incidence.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, M]
        return incidence / column_mass


class DynamicHypergraphConv(nn.Module):
    """Two-phase attention hypergraph convolution."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_to_edge = nn.Linear(in_dim, hidden_dim, bias=False)
        self.edge_to_node = nn.Linear(hidden_dim, out_dim, bias=False)
        self.residual = nn.Linear(in_dim, out_dim, bias=False)
        self.node_attention = nn.Linear(in_dim, 1)
        self.edge_attention = nn.Linear(hidden_dim, 1)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, D_in], incidence: [N, M].
        node_hidden = self.node_to_edge(x)  # W1*x_i: [N, D_hidden]
        node_logits = self.node_attention(x)  # [N, 1]
        masked_node_logits = node_logits.masked_fill(incidence <= 0.0, -1e9)  # [N, M] by broadcast
        alpha = torch.softmax(masked_node_logits, dim=0) * (incidence > 0.0).float()  # [N, M]
        alpha = alpha / alpha.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [N, M]
        edge_features = alpha.t() @ node_hidden  # f_e: [M, D_hidden]
        edge_features = self.dropout(F.relu(self.edge_norm(edge_features)))  # [M, D_hidden]

        edge_hidden = self.edge_to_node(edge_features)  # W2*f_e: [M, D_out]
        edge_logits = self.edge_attention(edge_features).t()  # [1, M]
        masked_edge_logits = edge_logits.masked_fill(incidence <= 0.0, -1e9)  # [N, M] by broadcast
        beta = torch.softmax(masked_edge_logits, dim=1) * (incidence > 0.0).float()  # [N, M]
        beta = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [N, M]
        z = beta @ edge_hidden  # refined node features: [N, D_out]
        z = z + self.residual(x)  # residual patient-specific signal: [N, D_out]
        z = self.dropout(F.relu(self.node_norm(z)))  # [N, D_out]
        return z, incidence


class PrototypicalHead(nn.Module):
    """Learnable 3-class prototype decision layer."""

    def __init__(self, out_dim: int, num_classes: int = 3, temperature: float = 1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, out_dim) * 0.02)
        self.temperature = max(float(temperature), 1e-3)

    def squared_distances(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, D_out], prototypes: [C, D_out] -> distances: [N, C].
        return torch.cdist(z, self.prototypes, p=2).pow(2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self.squared_distances(z)  # [N, C]
        logits = -distances / self.temperature  # [N, C]
        probabilities = torch.softmax(logits, dim=1)  # [N, C]
        return logits, probabilities

    def separation_loss(self, margin: float) -> torch.Tensor:
        proto_dist = torch.cdist(self.prototypes, self.prototypes, p=2)  # [C, C]
        mask = ~torch.eye(proto_dist.size(0), dtype=torch.bool, device=proto_dist.device)  # [C, C]
        pairwise = proto_dist[mask]  # [C * (C - 1)]
        return F.relu(margin - pairwise).pow(2).mean()

    def mci_separation_loss(self, margin: float) -> torch.Tensor:
        """Keep the MCI prototype separated from both neighboring diagnoses."""
        prototypes = self.prototypes
        cn_mci = torch.linalg.vector_norm(prototypes[1] - prototypes[0])
        mci_ad = torch.linalg.vector_norm(prototypes[1] - prototypes[2])
        return F.relu(margin - cn_mci) + F.relu(margin - mci_ad)


class DHNNProtoPipeline(nn.Module):
    """Dynamic hypergraph aggregation followed by learnable class prototypes."""

    def __init__(self, in_dim: int, config: DHNNProtoConfig):
        super().__init__()
        self.config = config
        self.builder = DynamicHypergraphBuilder(
            knn_k=config.knn_k,
            clinical_sigma=config.clinical_sigma,
            biomarker_bins=config.biomarker_bins,
            biomarker_dims=config.biomarker_dims,
            modality_dims=config.modality_dims,
        )
        self.hypergraph_conv = DynamicHypergraphConv(in_dim, config.hidden_dim, config.out_dim, config.dropout)
        self.proto_head = PrototypicalHead(config.out_dim, config.num_classes, config.proto_temperature)

    def forward(self, x: torch.Tensor, clinical: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # x: [N, D_in], clinical: [N, C_clinical].
        incidence = self.builder.build(x, clinical)  # [N, M]
        z, incidence = self.hypergraph_conv(x, incidence)  # z: [N, D_out], incidence: [N, M]
        logits, probabilities = self.proto_head(z)  # each: [N, C]
        distances = self.proto_head.squared_distances(z)  # [N, C]
        return {
            "z": z,
            "incidence": incidence,
            "logits": logits,
            "probabilities": probabilities,
            "distances": distances,
        }

    @torch.no_grad()
    def initialize_prototypes_from_batch(self, x: torch.Tensor, clinical: torch.Tensor, labels: torch.Tensor) -> None:
        out = self.forward(x, clinical)
        z = out["z"]  # [N, D_out]
        for cls_idx in range(self.config.num_classes):
            cls_mask = labels == cls_idx  # [N]
            if cls_mask.any():
                self.proto_head.prototypes[cls_idx].copy_(z[cls_mask].mean(dim=0))

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
        mci_margin_loss = self.proto_head.mci_separation_loss(self.config.mci_margin)
        total = ce_loss + self.config.mci_margin_weight * mci_margin_loss
        return total, {
            "loss_total": float(total.detach().cpu()),
            "loss_ce": float(ce_loss.detach().cpu()),
            "loss_mci_margin": float(mci_margin_loss.detach().cpu()),
        }


def prototype_diagnostics(probabilities: np.ndarray, distances: np.ndarray, labels: list[str]) -> list[dict[str, float | str]]:
    diagnostics: list[dict[str, float | str]] = []
    for prob_row, dist_row in zip(probabilities, distances):
        row: dict[str, float | str] = {}
        for label, prob, dist in zip(labels, prob_row, dist_row):
            row[f"similarity_to_{label}"] = float(prob)
            row[f"distance_to_{label}"] = float(dist)
        row["nearest_prototype"] = labels[int(np.argmin(dist_row))]
        diagnostics.append(row)
    return diagnostics


def train_dhnn_proto_classifier(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    clinical_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    clinical_val_np: np.ndarray,
    x_test_np: np.ndarray,
    clinical_test_np: np.ndarray,
    config: DHNNProtoConfig,
    label_names: list[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | list[dict[str, float | str]] | dict[str, float]]]:
    torch.manual_seed(config.seed)
    torch_device = torch.device(device)

    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=torch_device)  # [N_train, D_in]
    y_train = torch.as_tensor(y_train_np, dtype=torch.long, device=torch_device)  # [N_train]
    clinical_train = torch.as_tensor(clinical_train_np, dtype=torch.float32, device=torch_device)  # [N_train, C]
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=torch_device)  # [N_val, D_in]
    y_val = torch.as_tensor(y_val_np, dtype=torch.long, device=torch_device)  # [N_val]
    clinical_val = torch.as_tensor(clinical_val_np, dtype=torch.float32, device=torch_device)  # [N_val, C]
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=torch_device)  # [N_test, D_in]
    clinical_test = torch.as_tensor(clinical_test_np, dtype=torch.float32, device=torch_device)  # [N_test, C]

    model = DHNNProtoPipeline(x_train.size(1), config).to(torch_device)
    if config.initialize_prototypes:
        model.eval()
        model.initialize_prototypes_from_batch(x_train, clinical_train, y_train)

    if len(config.class_weights) != config.num_classes:
        raise ValueError("class_weights must contain one value per class.")
    class_weights = torch.as_tensor(config.class_weights, dtype=torch.float32, device=torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_losses: dict[str, float] = {}
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_out = model(x_train, clinical_train)
        loss, losses = model.loss(train_out["logits"], y_train, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        print(
            f"epoch={epoch + 1}/{config.epochs} "
            f"loss={losses['loss_total']:.6f} ce={losses['loss_ce']:.6f} "
            f"mci_margin={losses['loss_mci_margin']:.6f}",
            flush=True,
        )

        model.eval()
        with torch.no_grad():
            eval_out = model(x_val, clinical_val)
            val_logits = eval_out["logits"]
            val_loss, _ = model.loss(val_logits, y_val, class_weights=None)
            val_value = float(val_loss.cpu())

        if val_value < best_val:
            best_val = val_value
            best_losses = losses
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train, clinical_train)
        out = model(x_test, clinical_test)
        z_train = train_out["z"]  # [N_train, D_out]
        z_test = out["z"]  # [N_test, D_out]
        probabilities = out["probabilities"]  # [N_test, C]
        distances = out["distances"]  # [N_test, C]
        predictions = probabilities.argmax(dim=1)  # [N_test]

    y_pred = predictions.cpu().numpy()
    y_proba = probabilities.cpu().numpy()
    dist_np = distances.cpu().numpy()
    artifacts: dict[str, np.ndarray | list[dict[str, float | str]] | dict[str, float]] = {
        "z_train": z_train.cpu().numpy(),
        "z_test": z_test.cpu().numpy(),
        "prototype_vectors": model.proto_head.prototypes.detach().cpu().numpy(),
        "prototype_distances_test": dist_np,
        "incidence_eval": out["incidence"].detach().cpu().numpy(),
        "diagnostics": prototype_diagnostics(y_proba, dist_np, label_names),
        "losses": {"best_val_loss": best_val, **best_losses},
    }
    return y_pred, y_proba, artifacts
