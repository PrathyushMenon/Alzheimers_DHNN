# Multimodal Alzheimer's Disease Pipeline

This repository contains a leak-aware multimodal CN/MCI/AD classification
pipeline using pre-extracted dMRI, MRI/PET, and clinical features followed by
PCA, dynamic hypergraph attention, and a prototypical classifier.

> **Important limitation:** the repository contains the Swin-FOD and ALBEF
> source/configuration material that was available to us, but it does not
> contain the exact trained author checkpoints, the missing custom 3-D ALBEF
> model module, or a verified copy of the authors' preprocessing/mask. Therefore
> a fresh clone cannot currently turn raw scans into the paper's exact
> embeddings by itself. The public `main.py` contract requires compatible
> pre-extracted embeddings. If only raw scans are supplied, it stops with an
> actionable error rather than silently using an unrelated fallback.

## What is included

The pipeline performs:

1. Input and shape validation.
2. Training-only PCA and scaling.
3. A 58-dimensional fusion:
   - 20 dMRI/Swin-FOD components.
   - 35 combined MRI/PET/ALBEF components.
   - 3 standardized clinical variables: age, sex, and MoCA/MMSE.
4. Modality-specific soft hyperedges for dMRI and MRI/PET, plus clinical
   hyperedges.
5. Two-phase attention with a residual skip path.
6. Prototypical CN/MCI/AD classification.
7. Five-seed probability ensembling.
8. Accuracy, balanced accuracy, macro F1, MCC, per-class metrics, confusion
   matrix, predictions, and t-SNE plots.

The final test set is never used to fit PCA or scalers. Model selection and
threshold tuning must also be performed only on training/development data.

## Installation

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS, activate the environment with:

```bash
source .venv/bin/activate
```

Python 3.10 or newer is recommended. A CPU run is supported; CUDA-enabled
PyTorch can be installed separately when GPU training is available.

## Data layout

### Raw source data (not committed)

Keep raw data outside Git and never commit it. The intended raw-data locations
are:

```text
data/raw/
├── clinical.csv
├── dmri/
├── mri/
└── pet/
```

For an ADNI-style archive, the modality directories may contain nested
subject/session DICOM or NIfTI trees, for example:

```text
data/raw/mri/<subject>/<session>/...
data/raw/pet/<subject>/<session>/...
data/raw/dmri/<subject>/<session>/...
```

`clinical.csv` must contain one row per subject and these columns:

```text
diagnosis, age, sex, moca
```

`mmse` may be used instead of `moca`. Diagnoses must be exactly `CN`, `MCI`, or
`AD`. Sex may be encoded as `male`/`female`, `M`/`F`, or `1`/`2`.

If a `split` column is present, it must contain `train`, `val`, or `test`.
Otherwise rows must be ordered to match the embedding arrays in train, val,
test order. Subject IDs should also be retained in the clinical file so that
external preprocessing can verify identity alignment.

The same subject must have all required modalities and must never be split
randomly by scan. Keep every visit/session for a subject in exactly one of
train, validation, or test. Do not mix scans from one subject across
partitions. The pipeline expects one aligned subject row per modality; missing
or mismatched subjects must be resolved before feature extraction.

### Embeddings required by `main.py`

Before running the release entrypoint, place finite two-dimensional NumPy
arrays in:

```text
data/processed/
├── dmri/
│   ├── train_features.npy
│   ├── val_features.npy
│   └── test_features.npy
├── mri/
│   ├── train_features.npy
│   ├── val_features.npy
│   └── test_features.npy
└── pet/
    ├── train_features.npy
    ├── val_features.npy
    └── test_features.npy
```

The row order in each modality must match `clinical.csv`. MRI and PET are
concatenated and reduced jointly to 35 components. dMRI is reduced to 20
components. The resulting model input is exactly 58 dimensions.

The repository also contains the legacy experiment layout:

```text
features/fod/{train,val,test}_features.npy
features/mri_pet/{train,val,test}_features.npy
```

Those files are used by the experimental scripts under `code/`; the public
single-entrypoint contract uses `data/processed/` and `data/raw/clinical.csv`.

## Running the pipeline

After installing dependencies, adding the raw archive and clinical metadata,
and supplying compatible pre-extracted embeddings:

```powershell
python main.py
```

The command reads [config.yaml](config.yaml). No Python edits are required for
normal configuration changes.

If embeddings are missing, the command fails clearly and explains which files
are required. It does not resize raw volumes or invent replacement features.
The Swin-FOD and ALBEF research scripts under `code/multimodalAD/` are retained
for users who have the missing compatible model/checkpoint assets, but they are
not a guaranteed raw-only one-command extractor in this release.

## Outputs

Results are written to:

```text
results/
├── metrics.csv
├── ensemble_metrics.json
└── plots/
    ├── input_tsne.csv
    ├── input_tsne.png
    ├── refined_tsne.csv
    └── refined_tsne.png
```

The JSON output includes the ensemble seeds, class weights, prototype
temperature, hypergraph configuration, confusion matrix, and per-class
precision/recall. The CSV is convenient for downstream analysis.

## Configuration

The default release configuration is in [config.yaml](config.yaml):

```yaml
ensemble_seeds: [42, 52, 62, 72, 82]
class_weights: [1.0, 1.25, 1.0]
mci_margin: 1.2
mci_margin_weight: 0.08
proto_temperature: 0.12
hypergraph_knn_k: 4
```

The ensemble averages all configured seeds. It does **not** run several
experiments and report only the highest score. Selecting the highest test score
would be test-set cherry-picking and would make the result unreliable.

## Reproducibility and leakage safeguards

- Split by subject, not by individual scan.
- Keep all visits from one subject in one partition.
- Fit PCA and scalers on training data only.
- Do not tune hyperparameters against the final test set.
- Check that clinical rows and modality arrays have identical subject order.
- Report accuracy together with balanced accuracy, macro F1, MCC, per-class
  recall, and the confusion matrix.
- Keep the final test set untouched until the experiment is complete.

## Checkpoints, masks, and compatibility disclosure

The repository includes the available source code and configuration related to
Swin-FOD and ALBEF, but it does not include the exact trained author
checkpoints or the complete exact ALBEF 3-D implementation required to
reproduce the paper's encoder output. The online ALBEF code included here is
not, by itself, an exact compatible checkpoint for this ADNI 3-D pipeline.
The exact author-provided cerebellar mask/provenance was also not confirmed.

The repository contains `data/MNI_cerebellum_mask.nii.gz` as an available
synthetic/substitute mask asset. Users must supply and validate the actual
study mask required by their preprocessing, and must not describe the included
file as the authors' exact mask unless its provenance has been independently
verified.

Accordingly, results generated with substitute or pre-extracted embeddings
must be described as using the closest available compatible components, not as
an exact reproduction of the authors' model. If the original checkpoints,
missing 3-D ALBEF module, actual mask, and preprocessing are obtained later,
they can be used to regenerate the arrays in `data/processed/`. Those private
assets are intentionally not committed.

## What is and is not uploaded

The GitHub release contains source code, configuration, documentation, the
available mask substitute, and empty input-directory markers. It intentionally
does not contain raw ADNI data, clinical records, derived feature arrays,
predictions, results, local environments, checkpoints, or private archives.
After cloning, users must provide their own raw data, actual mask, compatible
encoder checkpoints/model files, and generated embeddings.

## Expected performance

The included development cohort is small, so its accuracy is not a reliable
predictor of performance on a future 450-subject cohort. An 80% result cannot
be guaranteed. A larger cohort should be evaluated with a subject-level
holdout, development-only tuning, confidence intervals, and per-class metrics.
In particular, MCI recall must be reported because overall accuracy can hide a
model that mostly predicts CN or AD.

## Encoder scripts

The available Swin-FOD source and extraction wrapper are under
`code/multimodalAD/Swin_FOD/`. With a compatible Swin-FOD checkpoint and a
subject-aligned manifest, extraction can be run with:

```powershell
python code/multimodalAD/Swin_FOD/extract_features_local.py `
  --checkpoint checkpoints/swin_fod/model.pt `
  --manifest data/official_inputs/swin_fod_local.csv `
  --output-dir data/processed/dmri
```

The available ALBEF source and training scripts are under
`code/multimodalAD/ALBEF/`. The custom 3-D model module
`models/model_pretrain3D.py` and its compatible checkpoint are not present in
the public source, so `scripts/train_albef.sh` stops with a precise message
until those private/custom assets are supplied. Official 2-D Salesforce ALBEF
weights are not compatible with this 3-D MRI/PET model.

These limitations are intentional and documented: the project does not claim
that an unrelated checkpoint is an exact encoder. Once compatible encoders
produce the three split arrays under `data/processed/`, the public classifier
is run with:

```powershell
python main.py
```
