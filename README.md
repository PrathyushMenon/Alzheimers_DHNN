# Multimodal Alzheimer's Disease Pipeline

## **Important limitation:** the repository contains the Swin-FOD and ALBEF
source/configuration material that was available to us, but it does not
contain the exact trained author checkpoints, the missing custom 3-D ALBEF
model module, or a verified copy of the authors' preprocessing/mask. Therefore
a fresh clone cannot currently turn raw scans into the paper's exact
embeddings by itself. The public `main.py` contract requires compatible
pre-extracted embeddings. If only raw scans are supplied, it stops with an
actionable error rather than silently using an unrelated fallback.

## **Important runtime requirement:** the Swin-FOD and ALBEF implementations are
included in the repository, but implementations alone are not runnable
encoders. To go directly from raw scans to final output, users must also
provide compatible trained checkpoints, the complete matching model modules,
the preprocessing/registration configuration, a valid study mask, and a
subject-aligned extraction manifest. If any one of those required assets is
missing or incompatible, raw-only execution cannot produce valid embeddings.
The current public `main.py` invokes the existing raw-data runner when
pre-extracted embeddings are absent. That runner still requires compatible
checkpoints, matching model modules, preprocessing tools, and manifests. It
stops with an actionable error rather than silently using an unrelated
fallback.

## What is present

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

### Required raw-pipeline software

The Python packages are listed in [requirements.txt](requirements.txt). The raw
ADNI stages also call command-line neuroimaging software that pip cannot
reliably install:

| Tool | Used for | Required command(s) |
|---|---|---|
| `dcm2niix` | DICOM-to-NIfTI conversion | `dcm2niix` |
| ClinicaDL/Clinica | T1-linear preprocessing and CAPS output | `clinicadl`, `clinica` |
| ANTs | PET registration and normalization | `antsRegistration` |
| MRtrix3 | dMRI denoising, preprocessing, response estimation, and FOD | `dwidenoise`, `dwifslpreproc`, `dwi2response`, `dwi2fod`, `mrtransform` |
| FSL | Diffusion preprocessing used by MRtrix3 | FSL environment and `eddy` |

On Ubuntu/WSL, install the Python dependencies first and then use the supplied
setup helper:

```bash
bash scripts/setup_exact_environment.sh
```

That helper installs the available system packages and the pinned ClinicaDL
and Clinica versions. Install FSL separately from the official FSL
documentation, because its license, environment variables, and distribution
are managed outside pip. Confirm the tools are visible before running:

```bash
command -v dcm2niix clinicadl antsRegistration dwidenoise dwifslpreproc dwi2fod mrtransform
```

On Windows, run the pipeline inside WSL2 Ubuntu. The Python entrypoint detects
Windows and forwards the raw stages to WSL; native Windows installations of
MRtrix3, FSL, ANTs, and ClinicaDL are not assumed.

## Data layout

### Raw source data

The intended raw-data locations are:

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

### Optional pre-extracted embeddings

The raw pipeline creates these arrays automatically. If you already have
compatible embeddings, you can skip raw extraction and place finite
two-dimensional NumPy arrays in:

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

The repository also contains the previous experiment layout produced by the raw
runner:

```text
features/fod/{train,val,test}_features.npy
features/mri_pet/{train,val,test}_features.npy
```

Those files are used by the experimental scripts under `code/`; the public
single-entrypoint contract uses `data/processed/` and `data/raw/clinical.csv`.

If these arrays are absent, `main.py` invokes `code/run_pipeline.py`, which
runs the existing organization, split, ClinicaDL, PET, FOD, Swin-FOD, and
ALBEF stages. Its legacy feature outputs are adapted to this release layout
before DHNN runs.

## Running the pipeline

After installing dependencies, installing the required system tools, adding
the raw archive and clinical metadata, and supplying the compatible encoder
assets:

```powershell
python main.py
```

The command reads [config.yaml](config.yaml). No Python edits are required for
normal configuration changes.

If embeddings are missing, the command runs the raw stages and reports any
missing external tool, checkpoint, or model module clearly. It does not resize
raw volumes or invent replacement features.

### Complete runtime checklist

Before `python main.py`, verify that all of the following are available:

1. `data/raw/clinical.csv` with diagnosis, age, sex, and MoCA/MMSE.
2. Subject- and visit-matched MRI, PET, and dMRI data under `data/raw/`.
3. A valid MNI cerebellar reference mask at
   `data/MNI_cerebellum_mask.nii.gz`, or a replacement selected with
   `CEREBELLUM_MASK`.
4. The compatible Swin-FOD checkpoint at
   `checkpoints/swin_fod/model.pt` or `model_final.pt`.
5. The complete compatible custom 3-D ALBEF model module under
   `code/multimodalAD/ALBEF/models/model_pretrain3D.py`.
6. The compatible 3-D ALBEF checkpoint at
   `checkpoints/albef/hable_pretrain_checkpoint.pth`, or a replacement
   selected with `ALBEF_PRETRAIN_CHECKPOINT`.
7. The external commands listed above on the WSL/Linux `PATH`.
8. Enough disk space for BIDS, CAPS, PET SUVR, FOD, feature, log, and result
   outputs. These derived files can require many gigabytes.

When all eight requirements are met, a clean run is:

```bash
python main.py
```

The entrypoint reuses existing outputs where possible, runs missing raw stages,
creates subject-level splits and manifests, extracts encoder features, adapts
them to `data/processed/`, and then runs the DHNN classifier. If any required
asset is absent or incompatible, it stops rather than producing misleading
results.

### Important expectation about the two checkpoints

The complete input is **not** just raw data plus two checkpoint files.
Checkpoints contain learned weights, but a successful raw run also needs the
matching model definitions, preprocessing behavior, mask, manifests, and
system tools listed above. In particular, the 3-D ALBEF checkpoint must match
the custom 3-D ALBEF module, and the Swin-FOD checkpoint must match the
included Swin-FOD architecture and input preprocessing.

The supported workflow is:

```text
Install requirements.txt
Install/configure WSL2/Linux neuroimaging tools
Provide raw data, clinical.csv, mask, compatible checkpoints, and matching model files
Run: python main.py
```

When those prerequisites are present and compatible, `main.py` runs the
complete raw-to-results pipeline without manually creating
`data/processed/*.npy`. The stages have been wired together and syntax-checked,
but a fresh clone with private/large checkpoints and every external tool has
not been independently validated here. The README therefore documents the
required environment rather than promising that arbitrary checkpoints or
arbitrary ADNI folder layouts will work.

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

The repository includes the available Swin-FOD and ALBEF implementations and
their configuration/source material. The implementation is not the same thing
as a trained checkpoint. A raw-only run requires **all** of the following:

| Required item | Why it is required |
|---|---|
| Raw dMRI, MRI, PET, and clinical data | The subject-level inputs |
| Compatible trained Swin-FOD checkpoint | Learned dMRI feature extraction weights |
| Compatible trained 3-D ALBEF checkpoint | Learned MRI/PET feature extraction weights |
| Complete matching model modules | Checkpoints contain weights, not necessarily the Python architecture |
| Matching preprocessing and registration rules | The encoders require the same tensor layout, spacing, normalization, and input size used during training |
| Valid study/cerebellar mask | Required by the relevant preprocessing; the included substitute is not verified as the authors' exact mask |
| Subject-aligned manifest and split definition | Prevents MRI, PET, dMRI, and clinical rows from being paired incorrectly or leaking across splits |
| Raw-extraction wiring | `main.py` invokes `code/run_pipeline.py`; its legacy outputs are adapted to `data/processed/` before classification |

If any required item is unavailable, the raw-to-embedding stage cannot be
trusted and the final classifier cannot run from raw data alone. When all
required tools and model assets are available, `main.py` invokes the runner
and performs the complete raw extraction stage automatically.

The exact trained author checkpoints, exact preprocessing provenance, and
complete custom 3-D ALBEF module were not available to verify in this release.
The included encoder implementations can be used as a starting point, but
they do not automatically make arbitrary checkpoints compatible.

The repository contains `data/MNI_cerebellum_mask.nii.gz` as an available
synthetic/substitute mask asset. Users must supply and validate the actual
study mask required by their preprocessing, and must not describe the included
file as the authors' exact mask unless its provenance has been independently
verified.

Accordingly, results generated with substitute or pre-extracted embeddings
must be described as using the closest available compatible components, not as
an exact reproduction of the authors' model. If the original checkpoints,
complete 3-D ALBEF module, actual mask, and preprocessing are obtained later,
they can be used to regenerate the arrays in `data/processed/`. Those private
assets are intentionally not committed.

If the authors do not reply, the following can be recreated locally:

| Missing item | Local recreation approach | Faithfulness level |
|---|---|---|
| Swin-FOD checkpoint | Train or fine-tune the included Swin-FOD implementation on appropriately preprocessed dMRI data | Approximate unless the original training recipe and weights are recovered |
| ALBEF checkpoint | Train or pretrain the included 3-D ALBEF implementation on compatible MRI/PET data | Approximate unless the original training recipe and weights are recovered |
| Exact preprocessing and mask | Reconstruct the documented preprocessing, validate it on the local ADNI-format data, and use a verified study mask | Approximate unless the authors' files are recovered |

Recreating these items produces a working substitute pipeline only after the
raw-extraction runner is also configured and connected to `main.py`. It must
not be described as the authors' exact checkpoint or exact reproduction.

## What is and is not uploaded

The GitHub release contains source code, configuration, documentation, the
available mask substitute, and empty input-directory markers. It intentionally
does not contain raw ADNI data, clinical records, derived feature arrays,
predictions, results, local environments, checkpoints, or private archives.
After cloning, users must provide their own raw data, compatible encoder
checkpoints/model files, and either generated embeddings or a complete
configured raw-extraction stage.

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
