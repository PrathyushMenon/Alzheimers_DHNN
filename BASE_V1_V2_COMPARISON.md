# Base vs V1 vs V2 Comparison

## Purpose

This document compares the original baseline pipeline, Version 1, and the current Version 2 implementation. It is intended as a handover file so another engineer can quickly understand what changed, what stayed fixed, and which results were produced.

## Scope Boundary

All current V2 implementation work is in:

- `D:\ALZ_V2`

The base model directory is not part of the V2 implementation target:

- `D:\ALZ_base_model`

## Shared Upstream Pipeline

All three variants use the same conceptual upstream multimodal feature pipeline:

| Stage | Description | Changed in V1/V2? |
|---|---|---|
| dMRI branch | Swin-FOD extracts white-matter fiber representations `v_FOD` | No |
| MRI/PET branch | 3D ALBEF extracts/fuses T1 MRI and Tau PET representations `v_fused` | No |
| Tabular branch | Age, sex, and MoCA are used as clinical tabular features | No |
| Final embedding preparation | FOD + MRI/PET + tabular features are concatenated, PCA-reduced where needed, and standardized into patient embeddings `X` | No |
| Final classifier/head | Converts patient embeddings into CN/MCI/AD predictions | Yes |

The architectural experiments are isolated to the final classification head.

## Architecture Summary

| Version | Final Head | Cohort Modeling | Decision Rule | Training Signal |
|---|---|---|---|---|
| Base | Baseline fusion classifier / MLP-style final classifier | No explicit graph cohort structure | Learned classifier logits | Cross-entropy-style supervised classification |
| V1 | 2-layer GCN + k-NN evaluation head | Population graph over patient embeddings | Distance-weighted k-NN over GCN-refined training embeddings | Temporary linear head with cross-entropy during training |
| V2 current | Dynamic Hypergraph Neural Network + Prototypical Network | Dynamic soft hypergraph over patient embeddings, biomarker proxies, and clinical similarity | Softmax over negative squared distances to CN/MCI/AD prototypes; default ensemble averages two DHNN/prototype heads | Cross-entropy over prototype logits plus prototype separation loss |

## Base Architecture

The base pipeline keeps the final decision stage comparatively simple:

```text
FOD features + MRI/PET features + age/sex/MoCA
-> PCA / scaling
-> unified patient embedding X
-> baseline final classifier
-> CN/MCI/AD probabilities
```

Key properties:

- Treats patients mostly independently at the final decision step.
- Does not explicitly construct patient-to-patient cohort relationships.
- Does not use graph, hypergraph, k-NN, or prototype geometry in the final head.

## V1 Architecture

V1 replaces the baseline final classifier with a graph-refinement and k-NN decision head:

```text
X in R^(N x D)
-> population graph A from cosine/RBF similarity
-> self-loops and symmetric normalization
-> 2-layer GCN
-> Z in R^(N x D_out)
-> train-time temporary linear classifier
-> eval-time distance-weighted k-NN
-> CN/MCI/AD probabilities
```

Main V1 changes:

- Adds a cohort population graph where each subject is a node.
- Computes adjacency using embedding similarity.
- Uses graph smoothing to refine each patient embedding with cohort context.
- Uses a temporary linear head only for training gradients.
- Uses k-NN at evaluation time rather than the temporary linear head.

Confirmed V1 result from `D:\ALZ_V1\results\multimodal_ad_results.json`:

| Metric | V1 GCN+kNN |
|---|---:|
| Accuracy | 0.6667 |
| AUC macro OvR | 0.8410 |
| F1 macro | 0.6222 |
| Precision macro | 0.6444 |
| Recall macro | 0.6389 |
| MCC | 0.5058 |

## V2 Current Architecture

V2 replaces the final classifier with a dynamic hypergraph aggregation network and prototype decision layer:

```text
X in R^(N x D), clinical C in R^(N x 3)
-> dynamic hyperedge construction
-> incidence matrix H in R^(N x M)
-> node-to-hyperedge attention aggregation
-> hyperedge-to-node attention projection
-> refined patient embeddings Z in R^(N x D_out)
-> learnable CN/MCI/AD prototypes
-> squared Euclidean distances to prototypes
-> softmax over negative distances
-> optional two-head DHNN/prototype probability averaging
-> CN/MCI/AD probabilities
```

Main V2 changes:

- Uses hyperedges instead of only pairwise graph edges.
- Dynamically rebuilds hypergraph topology on every forward pass.
- Builds hyperedges from soft k-NN embedding similarity, biomarker proxy bins, and clinical bins.
- Uses attention for both node-to-hyperedge and hyperedge-to-node message passing.
- Uses learnable disease-stage prototypes instead of an MLP classifier.
- Adds prototype separation loss to keep CN/MCI/AD prototypes geometrically distinct.
- Adds prototype proximity diagnostics for interpretability.
- Uses a default two-member DHNN/prototype ensemble to improve accuracy while preserving the same DHNN/prototype flow.

Confirmed V2 result from `D:\ALZ_V2\results\multimodal_ad_results.json`:

| Metric | V2 DHNN+Prototype |
|---|---:|
| Accuracy | 0.9167 |
| AUC macro OvR | 0.8250 |
| F1 macro | 0.8519 |
| Precision macro | 0.9333 |
| Recall macro | 0.8333 |
| MCC | 0.8711 |

## Metric Comparison

| Metric | Base | V1 GCN+kNN | V2 DHNN+Prototype current |
|---|---:|---:|---:|
| Accuracy | 0.8333 | 0.6667 | 0.9167 |
| AUC macro OvR | Not re-read in this handover | 0.8410 | 0.8250 |
| F1 macro | Not re-read in this handover | 0.6222 | 0.8519 |
| Precision macro | Not re-read in this handover | 0.6444 | 0.9333 |
| Recall macro | Not re-read in this handover | 0.6389 | 0.8333 |
| MCC | Not re-read in this handover | 0.5058 | 0.8711 |

Accuracy change:

| Comparison | Accuracy Delta |
|---|---:|
| V2 current minus Base | +0.0834 |
| V2 current minus V1 | +0.2500 |

Important interpretation:

- V2 currently improves accuracy over both Base and V1.
- V2 also improves F1, precision, recall, and MCC over V1.
- V2 AUC is lower than V1 AUC in the latest recorded runs, so the current V2 tuning is accuracy-oriented rather than AUC-oriented.
- The test set is small, so these values should be treated as reproducibility-run results, not as final clinical validation.

## Implementation Files

Base:

- `D:\ALZ_base_model\code\05_train_fusion_model.py`

V1:

- `D:\ALZ_V1\code\gcn_knn_head.py`
- `D:\ALZ_V1\code\05_train_fusion_model.py`
- `D:\ALZ_V1\V1_BASE_COMPARISON.md`

V2 current:

- `D:\ALZ_V2\code\dhnn_proto_head.py`
- `D:\ALZ_V2\code\05_train_fusion_model.py`
- `D:\ALZ_V2\scripts\run_pipeline.sh`
- `D:\ALZ_V2\V2_DHNN_PROTO_IMPLEMENTATION.md`
- `D:\ALZ_V2\BASE_V1_V2_COMPARISON.md`

## V2 Output Artifacts

Primary result files:

- `D:\ALZ_V2\results\multimodal_ad_results.json`
- `D:\ALZ_V2\results\fusion_metrics.json`
- `D:\ALZ_V2\results\prototype_diagnostics_test.csv`
- `D:\ALZ_V2\results\y_true_test.npy`
- `D:\ALZ_V2\results\y_pred_test.npy`
- `D:\ALZ_V2\results\y_proba_test.npy`

Representation and geometry files:

- `D:\ALZ_V2\results\dhnn_z_train.npy`
- `D:\ALZ_V2\results\dhnn_z_test.npy`
- `D:\ALZ_V2\results\prototype_vectors.npy`
- `D:\ALZ_V2\results\prototype_distances_test.npy`
- `D:\ALZ_V2\results\hypergraph_incidence_eval.npy`

Ensemble secondary-head files:

- `D:\ALZ_V2\results\dhnn_secondary_z_train.npy`
- `D:\ALZ_V2\results\dhnn_secondary_z_test.npy`
- `D:\ALZ_V2\results\secondary_prototype_vectors.npy`

## Current Recommendation

Use V2 as the current best accuracy version.

For reporting, be transparent that:

- The architecture remains DHNN plus prototypical decision head.
- The accuracy improvement comes from a two-member DHNN/prototype ensemble using slightly different hypergraph/prototype settings.
- More robust comparison should be done with repeated seeds, cross-validation, and the final ADNI split intended for the paper reproduction.
