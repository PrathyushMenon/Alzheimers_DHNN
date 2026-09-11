# V2 DHNN + Prototypical Head Comprehensive Handover

## Executive Summary

V2 is the current best-accuracy variant of the final fusion classifier. It keeps the upstream multimodal ADNI feature extraction pipeline intact and replaces only the final classification head with a Dynamic Hypergraph Neural Network followed by a Prototypical Network decision layer.

The practical reason for this design is that CN, MCI, and AD are not always cleanly separable with a simple independent-patient MLP classifier. Disease staging can have overlapping biomarkers, especially around MCI. V2 therefore lets each patient representation borrow structured context from clinically and embedding-similar subjects before making a final disease-stage decision by geometric distance to learned class prototypes.

Current recorded V2 result:

| Metric | Value |
|---|---:|
| Accuracy | 0.9167 |
| AUC macro OvR | 0.8250 |
| F1 macro | 0.8519 |
| Precision macro | 0.9333 |
| Recall macro | 0.8333 |
| MCC | 0.8711 |

## Scope

This version changes only the final classification head in `D:\ALZ_V2`.

The upstream pipeline remains unchanged:

- Swin-FOD feature extraction
- ALBEF MRI/PET feature extraction
- age, sex, and MoCA tabular features
- PCA and standardization before final classification

The exact files changed/added for V2 are:

| File | Role |
|---|---|
| `D:\ALZ_V2\code\dhnn_proto_head.py` | Modular DHNN/prototypical PyTorch implementation |
| `D:\ALZ_V2\code\05_train_fusion_model.py` | Final-stage training script using V2 head |
| `D:\ALZ_V2\scripts\run_pipeline.sh` | Pipeline launcher updated to call the V2 final head |
| `D:\ALZ_V2\V2_DHNN_PROTO_IMPLEMENTATION.md` | This comprehensive V2 architecture and handover document |
| `D:\ALZ_V2\BASE_V1_V2_COMPARISON.md` | Comparison across base, V1, and V2 |

## High-Level Objective

The goal of V2 is to keep the upstream ADNI multimodal representation pipeline intact while replacing the final MLP-style classifier with a cohort-aware decision system:

- A Dynamic Hypergraph Neural Network models higher-order relationships among subjects.
- A Prototypical Network head predicts CN/MCI/AD by distance to learnable disease-stage prototypes.
- Diagnostic outputs expose how close each test subject is to each prototype.
- The current default uses a two-member DHNN/prototype ensemble to improve accuracy while preserving the same architecture family and prediction flow.

Why hypergraphs instead of a standard graph:

- A standard graph edge models mostly pairwise patient similarity.
- A hyperedge can connect multiple patients at once, which is a closer match for cohort-level clinical grouping.
- AD progression groups are often overlapping, not binary. Hyperedges let a patient softly participate in multiple neighborhoods.
- Hyperedges can represent different relationship types: embedding similarity, biomarker-like patterns, and clinical similarity.

Why prototypes instead of a normal MLP head:

- A prototype gives each class an explicit location in the learned disease-stage embedding space.
- Predictions become interpretable as distances to CN, MCI, and AD centers.
- Prototype distances can be exported as diagnostics, which is useful for checking borderline MCI/AD or CN/MCI cases.
- The separation loss encourages class centers to stay meaningfully apart.

## Architecture Overview

```text
FOD features + MRI/PET features + age/sex/MoCA
-> PCA
-> StandardScaler
-> unified patient embedding matrix X in R^(N x D)
-> dynamic hypergraph construction
-> hypergraph attention convolution
-> refined patient embeddings Z in R^(N x D_out)
-> learnable CN/MCI/AD prototypes
-> softmax over negative squared Euclidean distances
-> probability averaging across two DHNN/prototype heads
```

Tensor flow:

| Symbol | Shape | Meaning |
|---|---|---|
| `X_fod` | `N x D_fod` | Swin-FOD feature matrix |
| `X_mri_pet` | `N x D_mri_pet` | ALBEF MRI/PET feature matrix |
| `C` | `N x 3` | Clinical matrix: age, sex, MoCA |
| `X` | `N x D` | Concatenated, PCA-reduced, standardized patient embeddings |
| `H` | `N x M` | Dynamic hypergraph incidence matrix |
| `F_e` | `M x D_hidden` | Hyperedge representations after node-to-hyperedge aggregation |
| `Z` | `N x D_out` | Refined cohort-aware patient embeddings |
| `P` | `3 x D_out` | Learnable CN/MCI/AD prototype matrix |
| `distances` | `N x 3` | Squared Euclidean distance from each patient to each prototype |
| `probabilities` | `N x 3` | Class probabilities for CN/MCI/AD |

Implementation map:

| Pipeline step | Code location | Output |
|---|---|---|
| Load FOD, MRI/PET, tabular features | `05_train_fusion_model.py` | Raw split feature matrices |
| PCA and scaling | `05_train_fusion_model.py` | `x_train`, `x_val`, `x_test` |
| Build dynamic hypergraph | `DynamicHypergraphBuilder.build` | Incidence matrix `H` |
| Hypergraph message passing | `DynamicHypergraphConv.forward` | Refined embeddings `Z` |
| Prototype classification | `PrototypicalHead.forward` | Logits and probabilities |
| Loss calculation | `DHNNProtoPipeline.loss` | CE plus prototype separation |
| Training/evaluation wrapper | `train_dhnn_proto_classifier` | Predictions, probabilities, artifacts |

End-to-end model flow in code:

```text
05_train_fusion_model.py
-> train_dhnn_proto_classifier(...)
-> DHNNProtoPipeline.forward(...)
-> DynamicHypergraphBuilder.build(...)
-> DynamicHypergraphConv.forward(...)
-> PrototypicalHead.forward(...)
```

## Data Preparation Details

V2 consumes split-level feature files already created by earlier stages:

| Input directory | Expected files |
|---|---|
| `D:\ALZ_V2\features\fod` | `train_features.npy`, `val_features.npy`, `test_features.npy` |
| `D:\ALZ_V2\features\mri_pet` | `train_features.npy`, `val_features.npy`, `test_features.npy` |
| `D:\ALZ_V2\data\bids` | `participants.tsv` |
| `D:\ALZ_V2\data\splits` | `train.tsv`, `val.tsv`, `test.tsv` |

Feature preparation steps:

- FOD features are loaded by split.
- MRI/PET features are loaded by split.
- Age, sex, and MoCA are aligned to split participant IDs.
- FOD features are PCA-reduced when their dimension exceeds the configured maximum.
- MRI/PET features are PCA-reduced when their dimension exceeds the configured maximum.
- FOD, MRI/PET, and tabular features are concatenated.
- A `StandardScaler` is fitted on the training split and applied to validation/test.
- A separate clinical scaler is fitted for age/sex/MoCA so clinical hyperedges use normalized clinical values.

Current recorded final input size:

| Quantity | Value |
|---|---:|
| Final embedding dimension `D` | 58 |
| Clinical dimension | 3 |
| Number of classes | 3 |

Label order is fixed as:

```text
0 = CN
1 = MCI
2 = AD
```

## Dynamic Hypergraph Module

Main module:

- `D:\ALZ_V2\code\dhnn_proto_head.py`

Classes:

- `DynamicHypergraphBuilder`
- `DynamicHypergraphConv`
- `PrototypicalHead`
- `DHNNProtoPipeline`

Hyperedge construction:

- soft k-NN hyperedges from patient embeddings
- biomarker proxy hyperedges from low-dimensional fused feature components
- clinical hyperedges from age, sex, and MoCA
- incidence matrix `H` has shape `N x M`
- topology is rebuilt every forward pass from the current patient embeddings

The implementation uses three hyperedge families:

| Hyperedge family | Source | Purpose |
|---|---|---|
| Soft k-NN hyperedges | Cosine similarity over current patient embeddings | Groups subjects that are close in the learned multimodal feature space |
| Biomarker proxy hyperedges | First low-dimensional fused embedding components binned softly | Approximates PET/atrophy-style cohort structure available at the final feature stage |
| Clinical hyperedges | Age, sex, and MoCA after scaling, binned softly | Adds clinically meaningful grouping pressure |

The raw final-stage feature matrix does not expose explicit Tau uptake maps or atrophy maps. Because of that, V2 uses low-dimensional fused feature components as biomarker proxies while still using real MoCA/age/sex clinical inputs.

### Incidence Matrix Definition

The hypergraph is represented by an incidence matrix:

```text
H in R^(N x M)
```

Where:

- `N` is the number of patient nodes in the current cohort/batch.
- `M` is the total number of dynamic hyperedges.
- `H[i, j]` is the soft membership weight of patient `i` in hyperedge `j`.

Unlike a binary incidence matrix, V2 uses soft memberships. This means a patient can partially belong to multiple hyperedges, which is useful for transitional disease stages such as MCI.

### Soft k-NN Hyperedges

For patient embeddings:

```text
X in R^(N x D)
```

The builder computes normalized cosine similarity:

```text
X_norm = normalize(X)
S = X_norm X_norm^T
S in R^(N x N)
```

For each patient-centered hyperedge, the top `k` most similar patients are selected. Their similarities are converted into soft weights with a softmax:

```text
weights = softmax(topk(S))
```

This creates `N` patient-centered hyperedges:

```text
H_knn in R^(N x N)
```

Each column acts like a local cohort neighborhood around one patient.

### Biomarker Proxy Hyperedges

The original V2 requirement mentions clinical attribute thresholding based on Tau PET uptake, structural atrophy, and MoCA. At this final training stage, raw Tau uptake maps and raw atrophy maps are not directly exposed. The available signal is the fused embedding produced from earlier FOD and MRI/PET feature extraction.

V2 therefore creates biomarker proxy hyperedges from the first low-dimensional components of `X`:

```text
X[:, 0:biomarker_dims]
```

Each selected dimension is softly assigned into `biomarker_bins` bins. With defaults:

```text
biomarker_dims = 4
biomarker_bins = 3
```

This contributes:

```text
4 * 3 = 12 biomarker proxy hyperedges
```

These hyperedges approximate latent biomarker regimes learned by the upstream multimodal representation stack.

### Clinical Hyperedges

Clinical hyperedges are built from:

```text
C = [age, sex, MoCA]
C in R^(N x 3)
```

Each clinical column is softly binned into `biomarker_bins` bins. With defaults:

```text
3 clinical variables * 3 bins = 9 clinical hyperedges
```

MoCA is especially important because it gives the hypergraph a direct cognitive severity axis.

### Total Hyperedge Count

With default settings, the approximate number of hyperedges is:

```text
M = N + (biomarker_dims * biomarker_bins) + (clinical_dim * biomarker_bins)
M = N + (4 * 3) + (3 * 3)
M = N + 21
```

The exact `M` depends on the cohort size `N` and whether clinical features are provided.

### Dynamic Rewiring

The hypergraph is rebuilt inside every forward pass:

```text
incidence = self.builder.build(x, clinical)
```

This means the topology can change as the model is trained and embeddings/projections evolve. In the current implementation, the builder constructs topology from the current final-stage patient embeddings and clinical features passed to the head. This keeps the design differentiable through the message-passing layers while updating cohort structure each pass.

## Hypergraph Attention Convolution

Message passing:

- node-to-hyperedge aggregation with attention weights `alpha`
- hyperedge-to-node projection with attention weights `beta`
- output refined embeddings `Z` have shape `N x D_out`

The implemented attention convolution follows two phases:

```text
Node to hyperedge:
X: N x D
W1(X): N x D_hidden
alpha: N x M
F_e = alpha^T W1(X): M x D_hidden

Hyperedge to node:
W2(F_e): M x D_out
beta: N x M
Z = beta W2(F_e): N x D_out
```

V2 also includes:

- residual projection from `X` to `Z`, preserving patient-specific signal
- LayerNorm after hyperedge and node projections
- dropout for regularization
- gradient clipping during training

### Attention Mechanics

The attention module learns two attention directions:

| Attention | Shape | Meaning |
|---|---|---|
| `alpha` | `N x M` | How much each node contributes to each hyperedge |
| `beta` | `N x M` | How much each hyperedge contributes back to each node |

The incidence matrix masks invalid node-hyperedge memberships. Attention is only normalized over valid memberships:

```text
alpha = softmax(masked_node_logits over nodes)
beta = softmax(masked_edge_logits over hyperedges)
```

This avoids message leakage through hyperedges a patient does not belong to.

### Residual Connection

The hypergraph convolution adds a residual projection:

```text
Z = hypergraph_context + residual_projection(X)
```

This is important because cohort smoothing can blur individual patient-specific information. The residual path lets the model preserve subject-specific biomarker signal while still benefiting from cohort context.

### Normalization and Dropout

LayerNorm is applied after hyperedge and node projections to stabilize optimization. Dropout is used after activation to reduce overfitting on the small cohort.

## Prototypical Decision Head

The model learns three prototypes:

- `c_CN`
- `c_MCI`
- `c_AD`

For each patient embedding `z_i`, the head computes:

```text
d_k(z_i) = ||z_i - c_k||_2^2
P(y_i = k | z_i) = softmax(-d_k)
```

Training loss:

```text
L_total = L_CE + lambda * L_sep
```

Where:

```text
L_sep = mean(max(0, margin - ||c_i - c_j||_2)^2), for i != j
```

This keeps prototypes separated without encouraging unbounded magnitude drift.

### Prototype Geometry

The prototype matrix is:

```text
P in R^(3 x D_out)
```

Each row is a learnable class center:

```text
P[0] = c_CN
P[1] = c_MCI
P[2] = c_AD
```

For one patient:

```text
z_i in R^(D_out)
```

The squared distance vector is:

```text
d_i = [
  ||z_i - c_CN||_2^2,
  ||z_i - c_MCI||_2^2,
  ||z_i - c_AD||_2^2
]
```

Logits are the negative distances divided by temperature:

```text
logits_i = -d_i / temperature
```

The model predicts the class with the smallest distance / highest probability:

```text
y_pred = argmax(softmax(logits_i))
```

Temperature controls confidence sharpness:

- Lower temperature makes the softmax sharper and more decisive.
- Higher temperature makes probabilities smoother and less overconfident.

### Prototype Initialization

Prototypes start from data-driven class centroids when possible:

```text
c_k = mean(Z[y == k])
```

This gives the prototype head a clinically meaningful starting geometry instead of relying only on random prototype vectors.

## Default Hyperparameters

| Parameter | Value |
|---|---:|
| DHNN hidden dim | 64 |
| DHNN output dim | 32 |
| hypergraph k-NN k | 6 |
| clinical sigma | 0.75 |
| biomarker bins | 3 |
| biomarker proxy dims | 4 |
| prototype lambda | 0.1 |
| prototype margin | 2.0 |
| prototype temperature | 1.0 |
| dropout | 0.1 |
| class weight power | 0.5 |
| epochs | 250 |
| patience | 40 |
| learning rate | 0.001 |
| weight decay | 0.0001 |

## Training Strategy

Primary training path:

```text
train + validation embeddings
-> DHNN/prototype model
-> prototype logits
-> cross-entropy + prototype separation loss
-> early stopping monitored on validation loss
```

The current default sets `--train-with-val` to true for the primary head. This lets the final model use both train and validation embeddings for the final fit while still using validation loss during early stopping inside the helper routine.

Training loop details:

1. Convert NumPy features and labels into PyTorch tensors.
2. Instantiate `DHNNProtoPipeline`.
3. Optionally initialize prototypes from class centroids.
4. Compute class weights from label counts.
5. Run AdamW optimization.
6. Rebuild the hypergraph on every forward pass.
7. Compute prototype logits and probabilities.
8. Optimize CE plus prototype separation loss.
9. Clip gradients to avoid unstable updates.
10. Track best validation loss.
11. Restore the best model state before inference.

Class imbalance handling:

- Class counts are computed from the training labels.
- Weights use `count^(-class_weight_power)`.
- Weights are normalized to mean 1.
- Default `class_weight_power` is `0.5` for the primary head.

Prototype initialization:

- The model first computes `Z` for the training cohort.
- Each prototype is initialized from the mean `Z` of its class when class labels are available.
- The prototypes remain learnable after initialization.

Loss terms:

| Loss | Purpose |
|---|---|
| Cross-entropy | Makes each patient close to the correct class prototype relative to other prototypes |
| Prototype separation | Prevents CN/MCI/AD prototypes from collapsing into the same region |

Optimization defaults:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Max gradient norm | 5.0 |
| Epochs | 250 |
| Patience | 40 |

## Inference and Evaluation Flow

At inference/evaluation time:

```text
train embeddings + test embeddings
-> shared dynamic hypergraph
-> DHNN refinement
-> slice out test Z
-> prototype distances
-> prototype probabilities
-> predicted class
```

The train and test nodes are evaluated together in the same cohort graph so that test representations are refined using cohort structure. This matches the cohort-based design but should be considered transductive evaluation. If strict inductive deployment is needed later, evaluation should construct hyperedges from training reference nodes plus each incoming patient or from a stored population reference graph.

Metrics retained for 3-class CN/MCI/AD evaluation:

- Accuracy
- AUC macro one-vs-rest
- F1 macro
- Precision macro
- Recall macro
- Matthews correlation coefficient

## Accuracy-Oriented Ensemble

Default ensemble:

| Member | Train set | k | Temperature | Dropout | Class weight power |
|---|---|---:|---:|---:|---:|
| Primary | train + val | 6 | 1.0 | 0.1 | 0.5 |
| Secondary | train only | 4 | 0.8 | 0.05 | 0.3 |

The ensemble does not change the architecture flow. Both members are DHNN plus prototypical-head models. The final probability is:

```text
P_final = 0.5 * P_primary + 0.5 * P_secondary
```

Why this ensemble was used:

- The primary model uses train+val data and has stronger access to available labeled examples.
- The secondary model uses train only, smaller `k`, lower dropout, and sharper prototype temperature.
- Averaging the two probability outputs reduced brittle single-head decisions on the current split.
- The flow is still DHNN plus prototype decision making; no MLP classifier was reintroduced.

Accuracy-oriented tradeoff:

- The ensemble improved top-1 accuracy.
- The current V2 AUC is below the V1 recorded AUC.
- If calibration/ranking becomes more important than accuracy, tune temperature, class weights, and validation selection around AUC instead.

## Clinical Diagnostics

For every test subject, V2 writes a diagnostic row with:

- true label
- predicted label
- similarity/probability to CN prototype
- distance to CN prototype
- similarity/probability to MCI prototype
- distance to MCI prototype
- similarity/probability to AD prototype
- distance to AD prototype
- nearest prototype by distance

The diagnostic CSV is:

- `D:\ALZ_V2\results\prototype_diagnostics_test.csv`

How to read one diagnostic row:

| Column pattern | Meaning |
|---|---|
| `similarity_to_CN` | Prototype softmax probability assigned to CN |
| `distance_to_CN` | Squared Euclidean distance to CN prototype |
| `similarity_to_MCI` | Prototype softmax probability assigned to MCI |
| `distance_to_MCI` | Squared Euclidean distance to MCI prototype |
| `similarity_to_AD` | Prototype softmax probability assigned to AD |
| `distance_to_AD` | Squared Euclidean distance to AD prototype |
| `nearest_prototype` | Prototype with the smallest geometric distance |

Useful interpretation patterns:

- High `similarity_to_AD` and low `distance_to_AD` indicate the refined embedding is geometrically close to the AD prototype.
- Similar probabilities between MCI and AD indicate a borderline case.
- A mismatch between `predicted_label` and `nearest_prototype` should be investigated, especially when ensemble averaging is enabled.
- Very large distances to all prototypes can indicate out-of-distribution behavior or poor representation support.

## Outputs

V2 writes:

- `D:\ALZ_V2\results\y_true_test.npy`
- `D:\ALZ_V2\results\y_pred_test.npy`
- `D:\ALZ_V2\results\y_proba_test.npy`
- `D:\ALZ_V2\results\dhnn_z_train.npy`
- `D:\ALZ_V2\results\dhnn_z_test.npy`
- `D:\ALZ_V2\results\prototype_vectors.npy`
- `D:\ALZ_V2\results\prototype_distances_test.npy`
- `D:\ALZ_V2\results\hypergraph_incidence_eval.npy`
- `D:\ALZ_V2\results\prototype_diagnostics_test.csv`
- `D:\ALZ_V2\results\fusion_metrics.json`
- `D:\ALZ_V2\results\multimodal_ad_results.json`
- `D:\ALZ_V2\results\dhnn_secondary_z_train.npy`
- `D:\ALZ_V2\results\dhnn_secondary_z_test.npy`
- `D:\ALZ_V2\results\secondary_prototype_vectors.npy`

Artifact descriptions:

| Artifact | Description |
|---|---|
| `y_true_test.npy` | Ground-truth numeric labels for the test split |
| `y_pred_test.npy` | Predicted numeric labels for the test split |
| `y_proba_test.npy` | Final class probability matrix after ensemble averaging |
| `dhnn_z_train.npy` | Primary DHNN refined train embeddings |
| `dhnn_z_test.npy` | Primary DHNN refined test embeddings |
| `prototype_vectors.npy` | Primary learned CN/MCI/AD prototype vectors |
| `prototype_distances_test.npy` | Primary squared distances from test patients to prototypes |
| `hypergraph_incidence_eval.npy` | Incidence matrix from final train+test evaluation pass |
| `prototype_diagnostics_test.csv` | Human-readable prototype proximity breakdown |
| `fusion_metrics.json` | Full metrics/config/loss payload |
| `multimodal_ad_results.json` | Duplicate final results payload for pipeline compatibility |
| `dhnn_secondary_z_train.npy` | Secondary DHNN refined train embeddings |
| `dhnn_secondary_z_test.npy` | Secondary DHNN refined test embeddings |
| `secondary_prototype_vectors.npy` | Secondary learned prototype vectors |

## Current CPU Run

Command:

```powershell
python D:\ALZ_V2\code\05_train_fusion_model.py `
  --fod-features D:\ALZ_V2\features\fod `
  --mri-pet-features D:\ALZ_V2\features\mri_pet `
  --participants D:\ALZ_V2\data\bids\participants.tsv `
  --splits-dir D:\ALZ_V2\data\splits `
  --out-dir D:\ALZ_V2\results `
  --device cpu
```

Metrics:

| Metric | V2 DHNN/prototype |
|---|---:|
| Accuracy | 0.9167 |
| AUC macro OvR | 0.8250 |
| F1 macro | 0.8519 |
| Precision macro | 0.9333 |
| Recall macro | 0.8333 |
| MCC | 0.8711 |

## Assumptions and Design Constraints

Assumptions:

- The available split files are the intended ADNI train/val/test splits.
- The upstream feature files are already aligned to the split rows.
- The label vocabulary is exactly CN, MCI, and AD.
- Age, sex, and MoCA are present in `participants.tsv`.
- The final fused features preserve useful PET/atrophy-related information even if raw PET/atrophy scalar maps are unavailable at this stage.

Design constraints:

- V2 must not alter preprocessing hooks.
- V2 must not modify the base model folder.
- V2 must keep the DHNN plus prototype flow.
- V2 must remain runnable on CPU.
- V2 must log standard 3-class metrics.

Clinical/research caution:

- These outputs are not clinical-grade validation.
- Small test sets can make accuracy unstable.
- Repeated split/seed evaluation is needed before claiming generalization.
- Site/scanner/domain effects should be checked before any serious biomedical interpretation.

## Comparison Against Earlier Versions

| Version | Final head | Accuracy |
|---|---|---:|
| Base | Baseline final classifier | 0.8333 |
| V1 | GCN + k-NN | 0.6667 |
| V2 current | DHNN + prototypes, two-head probability average | 0.9167 |

V2 is the current best-accuracy version in the available runs.

Important caveat:

- V2 was tuned for accuracy as requested.
- V2 AUC macro OvR is `0.8250`, while V1 recorded `0.8410`.
- If AUC becomes the main objective, the V2 hyperparameters should be retuned for ranking/calibration instead of top-1 accuracy.

## Hyperparameter Guide

| Parameter | Effect | When to increase | When to decrease |
|---|---|---|---|
| `--hypergraph-knn-k` | Size of embedding-similarity neighborhoods | If graph is too fragmented or predictions are noisy | If classes are over-smoothed |
| `--clinical-sigma` | Soft bin width for clinical/biomarker hyperedges | If memberships are too sharp | If clinical bins mix unrelated patients |
| `--biomarker-bins` | Number of soft bins per proxy dimension | If biomarker regimes need finer grouping | If small cohort makes bins sparse |
| `--biomarker-dims` | Number of fused dimensions used as biomarker proxies | If more latent biomarker axes are useful | If proxy hyperedges add noise |
| `--proto-temperature` | Softmax sharpness over prototype distances | For smoother probabilities / better calibration | For sharper class decisions |
| `--proto-lambda` | Strength of prototype separation | If prototypes collapse | If separation hurts classification |
| `--proto-margin` | Minimum desired prototype distance | If prototypes remain too close | If prototypes are forced too far apart |
| `--dhnn-dropout` | Regularization strength | If overfitting | If underfitting |
| `--class-weight-power` | Class imbalance correction | If minority classes are missed | If minority classes are over-predicted |

Suggested next experiments:

1. Run repeated seeds for V2 and report mean/std accuracy, F1, MCC, and AUC.
2. Tune for AUC separately from accuracy.
3. Compare single-head vs ensemble on the same repeated-seed protocol.
4. Add calibration metrics such as expected calibration error and Brier score.
5. If raw Tau/atrophy scalars become available, replace proxy hyperedges with explicit biomarker hyperedges.

## Reproduction Command

```powershell
python D:\ALZ_V2\code\05_train_fusion_model.py `
  --fod-features D:\ALZ_V2\features\fod `
  --mri-pet-features D:\ALZ_V2\features\mri_pet `
  --participants D:\ALZ_V2\data\bids\participants.tsv `
  --splits-dir D:\ALZ_V2\data\splits `
  --out-dir D:\ALZ_V2\results `
  --device cpu
```

To disable the accuracy-oriented ensemble and run only one DHNN/prototype head:

```powershell
python D:\ALZ_V2\code\05_train_fusion_model.py `
  --fod-features D:\ALZ_V2\features\fod `
  --mri-pet-features D:\ALZ_V2\features\mri_pet `
  --participants D:\ALZ_V2\data\bids\participants.tsv `
  --splits-dir D:\ALZ_V2\data\splits `
  --out-dir D:\ALZ_V2\results `
  --device cpu `
  --no-ensemble-heads
```

## Engineering Status

Implemented:

- `DynamicHypergraphBuilder`
- `DynamicHypergraphConv`
- `PrototypicalHead`
- `DHNNProtoPipeline`
- joint CE plus prototype separation loss
- prototype centroid initialization
- class weighting
- early stopping
- gradient clipping
- primary plus secondary DHNN/prototype probability averaging
- metrics logging for accuracy, AUC, F1, precision, recall, and MCC
- diagnostic prototype proximity CSV

Not changed:

- Swin-FOD preprocessing/extraction hooks
- ALBEF MRI/PET extraction hooks
- tabular age/sex/MoCA input handling
- existing train/validation/test split file format

## Verification Checklist

Use this checklist before handing V2 to someone else:

- Confirm `D:\ALZ_V2\code\dhnn_proto_head.py` exists.
- Confirm `D:\ALZ_V2\code\05_train_fusion_model.py` imports `DHNNProtoConfig` and `train_dhnn_proto_classifier`.
- Confirm `D:\ALZ_V2\results\multimodal_ad_results.json` reports `"classifier": "dhnn_prototypical"`.
- Confirm `ensemble_heads` is true if reproducing the current `0.9167` accuracy run.
- Confirm `prototype_diagnostics_test.csv` exists.
- Confirm no V2 implementation files were written into `D:\ALZ_base_model`.

## Minimal Mental Model

The shortest correct way to explain V2 is:

```
Each patient is a node.
Similar patients and clinically similar patients form soft hyperedges.
The DHNN passes information through those hyperedges to refine each patient embedding.
The prototype head learns one geometric center for CN, one for MCI, and one for AD.
Prediction is whichever prototype the refined patient embedding is closest to, converted into probabilities.
```
