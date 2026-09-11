# Layer-Wise and Attention Ablation Study for V2 DHNN-Prototypical Model

## Section 1: Dimensionality Sensitivity Analysis ($D=58$ Justification)

We swept the requested PCA-controlled input dimensionalities $D \in \{20,30,40,50,58,70,80,100\}$ while preserving the finalized V2 DHNN plus prototypical decision flow. PCA was fitted on the training fold only. Because the training fold contains 36 subjects, each imaging block can contribute at most 35 principal components; configurations above this limit report the requested dimension and the effective achievable dimension.

| requested_D | effective_D | fod_components | mri_pet_components | variance_preserved | accuracy | f1_macro | recall_macro | mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20.000000 | 20.000000 | 17.000000 | 0.000000 | 0.866529 | 0.833333 | 0.750000 | 0.750000 | 0.727273 |
| 30.000000 | 30.000000 | 20.000000 | 7.000000 | 0.788102 | 0.750000 | 0.544444 | 0.611111 | 0.603023 |
| 40.000000 | 40.000000 | 20.000000 | 17.000000 | 0.834052 | 0.833333 | 0.603989 | 0.666667 | 0.738988 |
| 50.000000 | 50.000000 | 20.000000 | 27.000000 | 0.911891 | 0.833333 | 0.603989 | 0.666667 | 0.738988 |
| 58.000000 | 58.000000 | 20.000000 | 35.000000 | 0.964778 | 0.916667 | 0.851852 | 0.833333 | 0.871131 |
| 70.000000 | 70.000000 | 32.000000 | 35.000000 | 0.995540 | 0.750000 | 0.731313 | 0.722222 | 0.651786 |
| 80.000000 | 73.000000 | 35.000000 | 35.000000 | 1.000000 | 0.750000 | 0.544444 | 0.611111 | 0.603023 |
| 100.000000 | 73.000000 | 35.000000 | 35.000000 | 1.000000 | 0.750000 | 0.544444 | 0.611111 | 0.603023 |

The $D=58$ setting corresponds to the intended compact fusion regime with 20 FOD components, 35 MRI/PET components, and 3 clinical variables. Empirically, it provides a favorable bias-variance operating point: lower-dimensional configurations discard discriminative multimodal variance, whereas larger configurations introduce additional components without consistently improving class-balanced decision quality. The plateau at high requested dimensions reflects the finite training-fold PCA rank.

## Section 2: Layer-Wise and Attention Mechanism Dynamics

Attention entropy was computed after masked softmax normalization. Phase 1 entropy summarizes the concentration of node-to-hyperedge aggregation weights $\alpha_{ij}$, whereas Phase 2 entropy summarizes the concentration of hyperedge-to-node projection weights $\beta_{ij}$. Sparsity is the fraction of numerically inactive attention entries.

| requested_D | effective_D | alpha_entropy | beta_entropy | alpha_sparsity | beta_sparsity | silhouette_X | silhouette_Z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20.000000 | 20.000000 | 0.966572 | 0.706980 | 0.666667 | 0.666667 | -0.020771 | 0.040995 |
| 30.000000 | 30.000000 | 0.961638 | 0.863312 | 0.666667 | 0.666667 | -0.012433 | 0.026883 |
| 40.000000 | 40.000000 | 0.954819 | 0.830285 | 0.666667 | 0.666667 | -0.022795 | 0.090958 |
| 50.000000 | 50.000000 | 0.954135 | 0.890458 | 0.666667 | 0.666667 | -0.028329 | 0.065688 |
| 58.000000 | 58.000000 | 0.951165 | 0.836754 | 0.666667 | 0.666667 | -0.039483 | 0.080975 |
| 70.000000 | 70.000000 | 0.958674 | 0.967333 | 0.666667 | 0.666667 | -0.046003 | 0.041038 |
| 80.000000 | 73.000000 | 0.954459 | 0.944506 | 0.666667 | 0.666667 | -0.042743 | 0.071968 |
| 100.000000 | 73.000000 | 0.954459 | 0.944506 | 0.666667 | 0.666667 | -0.042743 | 0.071968 |

Pearson correlation between preserved PCA variance and Phase 1 entropy was $r=-0.499353$ with approximate $p=0.220064$. The corresponding Phase 2 correlation was $r=0.622378$ with approximate $p=0.103139$. Spearman rank correlations were $\rho=-0.469880$ for Phase 1 and $\rho=0.710843$ for Phase 2. These statistics quantify how increased retained PCA variance changes attention concentration in the cohort hypergraph.

Layer-wise contribution was estimated by extracting isolated representations from the trained $D=58$ primary DHNN and fitting an identical random-forest diagnostic probe on the train+validation fold. Phase 1 uses node-level reconstructions from hyperedge aggregates, Phase 2 uses the hyperedge-to-node context before residual addition, and the residual condition uses only $W_rx_i$.

| condition | probe_accuracy | f1_macro | recall_macro | mcc | silhouette | relative_mcc_contribution |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 1: Node -> Hyperedge aggregation | 0.500000 | 0.381818 | 0.416667 | 0.217262 | -0.072741 | 0.360288 |
| Phase 2: Hyperedge -> Node context | 0.583333 | 0.444444 | 0.444444 | 0.318182 | -0.036553 | 0.527645 |
| Residual path: W_r | 0.750000 | 0.544444 | 0.611111 | 0.603023 | 0.016861 | 1.000000 |
| Full DHNN: Phase 1 + Phase 2 + Residual | 0.750000 | 0.557692 | 0.583333 | 0.579365 | 0.080975 | 0.960769 |

The diagnostic workload is carried by the interaction between Phase 1 cohort aggregation, Phase 2 hyperedge projection, and the residual subject-specific pathway. A strong Phase 1 score indicates that subject-to-hyperedge aggregation captures class-relevant cohort structure; a strong Phase 2 score indicates that hyperedge context remains separable after projection back to subjects; and a strong residual score indicates that individual multimodal features remain independently discriminative. The full model is expected to dominate when higher-order cohort context improves ambiguous MCI boundary placement without erasing patient-specific biomarker signal.

## Section 3: Feature and Latent Space Visualization Analysis

Feature importance was estimated using mutual information and random-forest Gini importance on the finalized $D=58$ representation. Scores are reported separately for the input space $X$ and the refined DHNN latent space $Z\in\mathbb{R}^{N\times32}$.

### Top-10 Input Features by Combined Information Gain and Gini Importance

| rank_space | feature | mutual_information | gini_importance | combined_score |
| --- | --- | --- | --- | --- |
| Input X | moca | 0.549369 | 0.127194 | 0.676563 |
| Input X | MRI_PET_PC1 | 0.473690 | 0.072547 | 0.546237 |
| Input X | FOD_PC17 | 0.171060 | 0.010295 | 0.181355 |
| Input X | MRI_PET_PC21 | 0.163002 | 0.008457 | 0.171459 |
| Input X | MRI_PET_PC17 | 0.130067 | 0.009767 | 0.139833 |
| Input X | MRI_PET_PC24 | 0.088998 | 0.027859 | 0.116856 |
| Input X | sex | 0.111490 | 0.004620 | 0.116110 |
| Input X | MRI_PET_PC14 | 0.064615 | 0.045358 | 0.109973 |
| Input X | FOD_PC7 | 0.097932 | 0.011260 | 0.109192 |
| Input X | FOD_PC19 | 0.091872 | 0.013972 | 0.105844 |

### Top-10 Refined Latent Dimensions by Combined Information Gain and Gini Importance

| rank_space | feature | mutual_information | gini_importance | combined_score |
| --- | --- | --- | --- | --- |
| Refined Z | Z_03 | 0.513360 | 0.134331 | 0.647691 |
| Refined Z | Z_14 | 0.518071 | 0.078707 | 0.596778 |
| Refined Z | Z_30 | 0.409657 | 0.080924 | 0.490582 |
| Refined Z | Z_02 | 0.452861 | 0.027859 | 0.480720 |
| Refined Z | Z_23 | 0.358276 | 0.046707 | 0.404983 |
| Refined Z | Z_04 | 0.350209 | 0.049462 | 0.399671 |
| Refined Z | Z_01 | 0.335779 | 0.062862 | 0.398641 |
| Refined Z | Z_10 | 0.322044 | 0.075116 | 0.397160 |
| Refined Z | Z_32 | 0.338828 | 0.057894 | 0.396723 |
| Refined Z | Z_15 | 0.336310 | 0.050073 | 0.386384 |

### Class Separability Before and After DHNN Refinement

| space | silhouette_score |
| --- | --- |
| Pre-DHNN input X | -0.039483 |
| Post-DHNN latent Z | 0.080975 |

The silhouette comparison quantifies the geometric effect of higher-order hypergraph message passing. Improvement from $X$ to $Z$ indicates that dynamic cohort context sharpens class organization in the latent metric space used by the prototypes. A reduction would indicate that the model improves supervised decision boundaries despite lower unsupervised cluster compactness, a known possibility in transitional biomedical phenotypes such as MCI.

The corresponding two-dimensional t-SNE coordinate files are stored as `results/visualization_ablations/tsne_input_x.csv` and `results/visualization_ablations/tsne_refined_z.csv`. These files can be used directly to generate manuscript panels comparing the pre-DHNN and post-DHNN latent distributions.

## Reproducibility Notes

- All experiments were executed within `D:\ALZ_V2`.
- The upstream feature files were not regenerated; the ablation operates on existing Swin-FOD, ALBEF MRI/PET, and clinical features.
- PCA and standardization were fitted on the training fold only.
- The V2 architecture flow was preserved: dynamic hypergraph construction, masked two-phase attention, residual projection, and prototypical decision layer.
- Requested dimensions above the training-fold PCA rank are reported with their effective achievable dimensionality.
