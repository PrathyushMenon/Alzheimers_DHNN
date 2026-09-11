# 3. Methodology

## 3.1 Overview of System Architecture

We developed a multimodal Alzheimer's disease (AD) staging framework for three-way classification of cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer's disease (AD) subjects. The model targets the central difficulty of AD staging: diagnostic labels are discrete, whereas neurodegeneration evolves along a continuous and heterogeneous biomarker continuum. MCI is especially ambiguous, because subjects may express partially normal cognition, emerging tau pathology, early atrophy, or white-matter disruption at different rates. We therefore replace an independent subject-level multilayer perceptron classifier with a cohort-aware Dynamic Hypergraph Neural Network (DHNN) and a metric prototypical decision head.

For a cohort of $N$ subjects, modality-specific encoders produce diffusion MRI, MRI/PET, and clinical descriptors that are concatenated, reduced, and standardized into a unified matrix $X \in \mathbb{R}^{N \times D}$, where $x_i \in \mathbb{R}^{D}$ denotes the representation of subject $i$. The DHNN constructs a dynamic soft incidence matrix $H \in \mathbb{R}^{N \times M}$, with $M$ hyperedges, and refines $X$ into cohort-contextualized embeddings $Z \in \mathbb{R}^{N \times D_{\mathrm{out}}}$. A prototypical classifier then learns class archetypes $c_{\mathrm{CN}}, c_{\mathrm{MCI}}, c_{\mathrm{AD}} \in \mathbb{R}^{D_{\mathrm{out}}}$ and predicts each subject by its temperature-scaled metric proximity to these archetypes. This design explicitly couples multimodal biomedical representation learning, higher-order topological manifold modeling, and interpretable latent metric-space staging.

## 3.2 Multimodal Representation and Preprocessing Subsystems

Each subject is represented by three complementary information streams. First, the diffusion MRI subsystem converts preprocessed dMRI into fiber orientation distributions (FODs) and uses a Swin-FOD encoder to extract white-matter structural connectivity descriptors $v_{\mathrm{FOD},i} \in \mathbb{R}^{D_{\mathrm{FOD}}}$. This stream captures tract-level degeneration and disconnection patterns that may precede or accompany cortical atrophy. Second, the co-registered MRI/PET subsystem uses a 3D ALBEF-style visual transformer to jointly encode T1-weighted structural MRI and $^{18}$F-AV-1451 Tau PET, producing the fused representation $v_{\mathrm{fused},i}=f_{\mathrm{ALBEF}}(I_i^{\mathrm{MRI}},I_i^{\mathrm{PET}})\in\mathbb{R}^{D_{\mathrm{MP}}}$. This branch integrates neurodegenerative atrophy and molecular tau burden, which provide complementary evidence for disease stage. Third, the tabular clinical encoder uses age, sex, and Montreal Cognitive Assessment (MoCA), represented as $a_i=[\mathrm{age}_i,\mathrm{sex}_i,\mathrm{MoCA}_i]\in\mathbb{R}^{3}$ after numerical coding and training-fold standardization.

The final subject vector concatenates PCA-reduced FOD features, PCA-reduced MRI/PET features, and standardized clinical variables. PCA is fitted only on the training fold, with a maximum of $20$ FOD components and $50$ MRI/PET components. Standard scaling is also fitted on training data and applied unchanged to validation and test folds. The resulting unified matrix is $X\in\mathbb{R}^{N\times D}$, with the implemented experiments yielding $D=58$. This preprocessing preserves multimodal complementarity while controlling dimensionality for stable cohort-level learning.

## 3.3 Dynamic Hypergraph Neural Network Topology Construction

The cohort is modeled as a dynamic hypergraph $\mathcal{G}=(\mathcal{V},\mathcal{E},H)$, where $\mathcal{V}$ is the set of $N$ subject nodes, $\mathcal{E}$ is the set of $M$ hyperedges, and $H=[h_{ij}]\in\mathbb{R}^{N\times M}$ is a soft incidence matrix. The scalar $h_{ij}$ denotes the membership strength of subject $i$ in hyperedge $j$. Unlike pairwise graphs, hypergraphs encode higher-order cohort structure by allowing multiple subjects to share a common clinical or latent biomarker relation. This is biologically appropriate for AD, where subjects can occupy overlapping positions along tau, atrophy, white-matter, and cognitive axes. The incidence topology is reconstructed on every forward pass from current subject representations and clinical attributes, allowing dynamic rewiring rather than imposing a fixed population graph.

The first hyperedge family captures local embedding-space neighborhoods using cosine similarity $s_{ij}$ between standardized subject vectors. For each anchor subject $j$, the model identifies the $k$ most similar subjects $\mathcal{N}_k(j)$ and forms a soft patient-centered hyperedge:

$$
h^{\mathrm{knn}}_{ij}
=
\begin{cases}
\displaystyle
\frac{\exp(s_{ij})}
{\sum_{\ell \in \mathcal{N}_k(j)} \exp(s_{\ell j})},
& i \in \mathcal{N}_k(j),\\
0, & \text{otherwise}.
\end{cases}
\tag{11}
$$

The second hyperedge family captures latent biomarker regimes by softly binning leading fused representation dimensions. Let $b_{ir}$ denote subject $i$'s value on proxy dimension $r$, with $r\in\{1,\ldots,D_b\}$, and let $\mu_q$ denote bin center $q\in\{1,\ldots,Q\}$. The Gaussian soft-binning assignment is

$$
h^{\mathrm{bio}}_{i,r,q}
=
\frac{
\exp\left(
-\frac{(b_{ir} - \mu_q)^2}{2\sigma^2}
\right)
}{
\sum_{q'=1}^{Q}
\exp\left(
-\frac{(b_{ir} - \mu_{q'})^2}{2\sigma^2}
\right)
}.
\tag{12}
$$

Clinical axis hyperedges for age, sex, and MoCA use the same Gaussian assignment mechanism as Eq. (12), replacing $b_{ir}$ with the corresponding standardized clinical variable and using clinical bin centers. This preserves explicit diagnostic axes, particularly MoCA-defined cognitive impairment, within the cohort topology. The full incidence matrix concatenates soft k-NN, biomarker-proxy, and clinical-axis hyperedges; under the default configuration, $M=N+D_bQ+3Q$, with $k=6$, $D_b=4$, and $Q=3$. Each hyperedge column is normalized to prevent high-mass hyperedges from dominating message passing:

$$
\hat{h}_{ij}
=
\frac{h_{ij}}{\sum_{n=1}^{N} h_{nj} + \epsilon},
\tag{14}
$$

where $\epsilon>0$ ensures numerical stability.

## 3.4 Two-Phase Masked Attention Message Passing

Given $X\in\mathbb{R}^{N\times D}$ and normalized incidence $\hat{H}\in\mathbb{R}^{N\times M}$, the DHNN performs two-phase attention-weighted message passing. The first phase aggregates subject-node features into hyperedge representations. Each node is projected into a hidden space using $W_1\in\mathbb{R}^{D_h\times D}$, and a masked softmax normalizes node-to-hyperedge attention strictly over active memberships:

$$
\alpha_{ij}
=
\frac{\exp(\ell^{\alpha}_{ij}) \mathbf{1}[\hat{h}_{ij}>0]}
{\sum_{n=1}^{N}\exp(\ell^{\alpha}_{nj}) \mathbf{1}[\hat{h}_{nj}>0]},
\qquad
f_j
=
\operatorname{LN}
\left(
\rho
\left(
\sum_{i=1}^{N}
\alpha_{ij} W_1x_i
\right)
\right)
\in \mathbb{R}^{D_h}.
\tag{17}
$$

Here, $\ell^{\alpha}_{ij}$ is the learnable node-to-hyperedge attention logit, $\rho(\cdot)$ is the activation function, and $\operatorname{LN}(\cdot)$ denotes layer normalization. Masking prevents absent subject-hyperedge pairs from contributing to the softmax denominator, preserving the semantics of the dynamically constructed topology.

The second phase returns hyperedge context to subject nodes. Hyperedge embeddings are projected through $W_2\in\mathbb{R}^{D_{\mathrm{out}}\times D_h}$, while reverse attention coefficients $\beta_{ij}$ are normalized only over hyperedges that contain subject $i$. The final node representation includes a residual projection $W_rx_i$, with $W_r\in\mathbb{R}^{D_{\mathrm{out}}\times D}$, to preserve individual biomarker information that could otherwise be diluted by cohort smoothing:

$$
z_i =
\operatorname{LN}
\left(
\rho
\left(
\sum_{j=1}^{M}
\beta_{ij} W_2 f_j
+ W_r x_i
\right)
\right),
\qquad
z_i \in \mathbb{R}^{D_{\mathrm{out}}}.
\tag{21}
$$

This two-phase formulation enables information exchange across clinically and biologically coherent subject groups rather than only along pairwise graph edges. The residual path, layer normalization, dropout, and gradient clipping jointly stabilize training on small biomedical cohorts.

## 3.5 Geometric Prototypical Decision Head and Diagnostics

The refined embeddings $Z$ are classified using a prototypical metric head. Three learnable disease archetypes $c_{\mathrm{CN}}$, $c_{\mathrm{MCI}}$, and $c_{\mathrm{AD}}$ are instantiated in $\mathbb{R}^{D_{\mathrm{out}}}$ and initialized from training-fold class centroids in the refined latent space. For subject $i$ and class $k$, the model computes the squared Euclidean distance $d_{ik}=\lVert z_i-c_k\rVert_2^2$. Distances are converted into posterior probabilities by temperature-scaled softmax:

$$
P(y_i=k \mid z_i)
=
\frac{\exp(-d_{ik}/T)}
{\sum_{j \in \{\mathrm{CN},\mathrm{MCI},\mathrm{AD}\}}
\exp(-d_{ij}/T)}.
\tag{24}
$$

The predicted label is the class with maximal posterior probability, equivalently the nearest prototype under squared Euclidean distance. This metric geometry provides clinically meaningful diagnostics: each prediction can be expressed as proximity to CN, MCI, and AD archetypes. Borderline MCI subjects can therefore be inspected by comparing their MCI-versus-AD or CN-versus-MCI distances, rather than relying only on opaque classifier logits.

## 3.6 Objective Function and End-to-End Optimization

Training minimizes a joint differentiable objective containing class-weighted cross-entropy and bounded prototype separation. Let $Y\in\{0,1\}^{N\times3}$ be the one-hot label matrix, $P\in\mathbb{R}^{N\times3}$ the predicted posterior matrix, $w_k$ the class weight for class $k$, and $\epsilon$ a numerical stability constant. Class weights are computed from training-fold frequencies as $w_k\propto n_k^{-\gamma}$ with imbalance exponent $\gamma=0.5$. The supervised term is

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{3}
w_k Y_{ik}
\log(P_{ik} + \epsilon).
\tag{25}
$$

To prevent prototype collapse without encouraging unbounded prototype norms, the model uses a bounded margin separation penalty:

$$
\mathcal{L}_{\mathrm{sep}}
=
\frac{1}{K(K-1)}
\sum_{\substack{p,q=1 \\ p \ne q}}^{K}
\left[
\max
\left(
0,
m - \|c_p - c_q\|_2
\right)
\right]^2.
\tag{26}
$$

The total objective is $\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\lambda\mathcal{L}_{\mathrm{sep}}$, with $\lambda=0.1$, margin $m=2.0$, and $K=3$ classes. The implemented optimizer is AdamW with learning rate $10^{-3}$, weight decay $10^{-4}$, maximum gradient norm $5.0$, $250$ training epochs, and early-stopping patience of $40$ epochs. All learnable DHNN projections, attention modules, and prototype vectors are optimized jointly. During evaluation, the trained model computes prototype distances and posterior probabilities for the held-out test subjects, and performance is reported using accuracy, macro F1-score, macro precision, macro recall, Matthews correlation coefficient (MCC), and macro one-vs-rest area under the receiver operating characteristic curve (AUC).

# 4. Experimental Setup and Implementation Details

## 4.1 Dataset and Multimodal Cohort Description

Experiments used an ADNI multimodal cohort containing CN, MCI, and AD diagnostic groups. Each subject was represented by diffusion MRI-derived FOD features, co-registered T1-weighted MRI and $^{18}$F-AV-1451 Tau PET features, and demographic/clinical variables comprising age, sex, and MoCA. Subject-level split files defined non-overlapping training, validation, and test partitions, and all diagnostic labels were mapped consistently as CN, MCI, and AD. Demographic and MoCA variables were aligned by participant identifier before model fitting, ensuring that each imaging-derived representation was paired with the correct clinical vector. Descriptive statistics for age, sex, and MoCA should be reported from the finalized cohort table as mean $\pm$ standard deviation for continuous variables and counts with percentages for categorical variables.

## 4.2 Preprocessing and Feature Extraction Parameters

The dMRI branch used Swin-FOD to encode white-matter FOD volumes into $v_{\mathrm{FOD}}$. The MRI/PET branch used a 3D ALBEF subsystem to fuse structural atrophy information from T1 MRI with tau deposition information from $^{18}$F-AV-1451 PET, yielding $v_{\mathrm{fused}}$. Age, sex, and MoCA were numerically encoded and standardized. Imaging feature dimensionality was controlled by PCA fitted exclusively on the training fold: FOD features were capped at $20$ principal components, and MRI/PET features were capped at $50$ principal components. The final concatenated matrix was standardized with training-fold `StandardScaler` statistics, yielding $X\in\mathbb{R}^{N\times58}$ for the evaluated experiments.

## 4.3 Baseline Model Configurations

Three final-stage classifiers were evaluated under the same upstream feature representation protocol. The base model used TabPFN directly on the concatenated, PCA-reduced, standardized subject embeddings. Version 1 replaced the final classifier with a population graph model: pairwise subject similarity was computed using cosine similarity, the representations were refined by a two-layer GCN with hidden dimension $64$ and output dimension $32$, and inference used distance-weighted k-nearest neighbors with $k=5$. Version 2 used the proposed DHNN with heterogeneous dynamic hyperedges, soft k-NN neighborhood size $k=6$, biomarker and clinical soft-bin count $Q=3$, output dimension $D_{\mathrm{out}}=32$, learnable CN/MCI/AD prototypes, temperature $T=1.0$, prototype margin $m=2.0$, separation coefficient $\lambda=0.1$, and a two-head probability-averaging ensemble.

\begin{table}[t]
\centering
\caption{Final-stage model configurations evaluated on the ADNI multimodal cohort.}
\label{tab:model_configs}
\begin{tabular}{llll}
\hline
\textbf{Model} & \textbf{Cohort structure} & \textbf{Classifier} & \textbf{Key parameters} \\
\hline
Base & None & TabPFN & $X\in\mathbb{R}^{N\times58}$ \\
V1 & Pairwise population graph & GCN + k-NN & $D_h=64$, $D_{\mathrm{out}}=32$, $k_{\mathrm{kNN}}=5$ \\
V2 & Dynamic heterogeneous hypergraph & DHNN + prototypes & $k=6$, $Q=3$, $T=1.0$, $m=2.0$, $\lambda=0.1$ \\
\hline
\end{tabular}
\end{table}

## 4.4 Optimization Protocols and Environment

V2 was trained in PyTorch using AdamW with learning rate $10^{-3}$, weight decay $10^{-4}$, $250$ maximum epochs, early-stopping patience $40$, and gradient clipping with maximum norm $5.0$. Class-weighted cross-entropy used frequency-derived weights with exponent $\gamma=0.5$. The current recorded run used CPU execution. The primary DHNN/prototype head was trained with train-plus-validation embeddings, while a secondary head was trained on the training split only using $k=4$, temperature $T=0.8$, dropout $0.05$, and class-weight exponent $0.3$; final class probabilities were computed by averaging the two heads. This ensemble preserves the DHNN-prototypical architecture while reducing variance in small-cohort decision boundaries.

# 5. Results and Comparative Analysis

## 5.1 Quantitative Benchmarking Across Models

Table \ref{tab:benchmark_results} reports the empirical performance of the base TabPFN classifier, V1 GCN+k-NN model, and V2 DHNN+prototypical model. V2 achieved the highest top-1 accuracy, macro F1-score, macro precision, macro recall, and MCC among the evaluated systems.

\begin{table}[t]
\centering
\caption{Comparative performance across final-stage classifiers. Macro AUC denotes one-vs-rest macro-averaged ROC-AUC. Values not available from the recorded base output are marked as not reported.}
\label{tab:benchmark_results}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Macro AUC} & \textbf{Macro F1} & \textbf{Precision} & \textbf{Recall} & \textbf{MCC} \\
\hline
Base TabPFN & $0.8333$ & N/R & N/R & N/R & N/R & N/R \\
V1 GCN + k-NN & $0.6667$ & $0.8410$ & $0.6222$ & $0.6444$ & $0.6389$ & $0.5058$ \\
V2 DHNN + Proto & $\mathbf{0.9167}$ & $0.8250$ & $\mathbf{0.8519}$ & $\mathbf{0.9333}$ & $\mathbf{0.8333}$ & $\mathbf{0.8711}$ \\
\hline
\end{tabular}
\end{table}

Relative to the base TabPFN model, V2 improved accuracy from $0.8333$ to $0.9167$, an absolute gain of $0.0834$. Relative to V1, V2 improved accuracy by $0.2500$ and MCC by $0.3653$, indicating substantially stronger agreement between predicted and ground-truth diagnostic labels.

## 5.2 Performance Analysis: TabPFN vs. GCN vs. DHNN

The base TabPFN classifier performed strongly because it operates directly on the compact multimodal representation $X\in\mathbb{R}^{N\times58}$ without imposing a potentially noisy cohort topology. In contrast, the V1 pairwise GCN degraded accuracy to $0.6667$. This drop is consistent with graph over-smoothing: pairwise edges constructed from global embedding similarity can blur local CN/MCI/AD boundaries, especially when MCI subjects lie between normal aging and dementia phenotypes. Although V1 retained a relatively high macro AUC of $0.8410$, its lower F1, recall, and MCC show that ranking signal did not translate into balanced top-1 diagnostic assignments.

V2 recovered and exceeded base performance by replacing pairwise graph smoothing with heterogeneous high-order hyperedges. Soft k-NN hyperedges captured local manifold neighborhoods, biomarker-proxy hyperedges grouped subjects along latent imaging axes, and clinical hyperedges preserved age, sex, and MoCA structure. This higher-order topology better matches the neurobiological organization of AD progression, where subjects are not simply connected by binary pairwise similarity but share multiple overlapping pathological and clinical contexts. The residual DHNN update further preserved individual subject signal while injecting transductive cohort context, enabling V2 to reach $0.9167$ accuracy and $0.8711$ MCC.

## 5.3 Metric Prototypical Staging vs. Hard Classification

The prototypical decision head reframes classification as disease-stage localization in a latent metric space. Instead of learning an unconstrained MLP boundary, V2 learns explicit CN, MCI, and AD archetypes and assigns subjects according to temperature-scaled distance. This geometry improves interpretability and class balance: the high macro precision of $0.9333$, macro recall of $0.8333$, and MCC of $0.8711$ indicate that V2 produced more coherent three-class decisions than the pairwise GCN model.

The main trade-off is between top-1 decision accuracy and ranking-oriented AUC. V2 achieved superior accuracy, F1, precision, recall, and MCC, whereas V1 produced a higher macro AUC. This suggests that V1 retained some probability-ranking information but failed to convert it into robust class assignments. V2's prototype geometry sharpened diagnostic decisions and improved agreement with labels, while its probability calibration may require additional tuning if AUC becomes the primary endpoint. Temperature scaling, prototype margin, and class-weight exponent are the most direct calibration levers.

## 5.4 Diagnostic Proximity and Case Analysis

V2 exports subject-level prototype diagnostics, including predicted label, nearest prototype, class probabilities, and squared Euclidean distances to $c_{\mathrm{CN}}$, $c_{\mathrm{MCI}}$, and $c_{\mathrm{AD}}$. These outputs convert model predictions into clinically interpretable geometric evidence. Subjects with small AD distance and high AD similarity represent embeddings near the AD archetype; subjects with comparable MCI and AD distances represent transitional or borderline phenotypes; and subjects far from all prototypes may indicate atypical multimodal patterns or weak support from the training cohort.

This diagnostic layer is particularly important for MCI, where biological heterogeneity is expected. A patient may show MoCA impairment closer to MCI while retaining structural or tau features nearer to CN, or may show imaging patterns approaching AD before severe cognitive decline. Prototype proximity therefore provides a mechanism for inspecting the continuum between diagnostic stages rather than treating the output as a hard class label alone. In the recorded V2 run, the improved MCC and macro recall support the interpretation that dynamic hypergraph refinement and prototype-based staging better preserve clinically relevant decision boundaries than a pairwise GCN while maintaining a transparent subject-to-archetype explanation.
