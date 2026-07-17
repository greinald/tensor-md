# Reproducibility

This repository contains the reusable location-aware tensor Mahalanobis
detector and its dependency/environment definitions. The thesis repository
contains the notebooks, official evaluator, and recorded experiment metrics.

## Environment

The recommended environment is Python 3.11 with the dependencies in
`requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name tensor-md --display-name "Python (tensor-md)"
```

After installation, the reusable tensor components can be imported as
`tensor_md`.  The package API contains `PatchExtractionConfig`,
`load_patch_datasets`, `LocationAwareTensorMahalanobisDetector`,
`NeighborhoodScoreLocationAwareTensorMahalanobisDetector`, and
`TensorGaussianState`.  The original files under `Code/Scripts` remain usable
for existing notebooks and command-line experiments.

Alternatively, use the supplied Conda definition:

```bash
conda env create -f environment.yml
conda activate tensor-md
```

On Apple Silicon, install the platform-supported TensorFlow wheel if the
generic `tensorflow` wheel is unavailable, then install the remaining
requirements.  The tensor detector itself uses NumPy and scikit-learn; the
TensorFlow and PyTorch packages are needed for the CNN feature extractors.
YOLO is optional and is only needed by the YOLO experiment notebooks.

## Dataset layout

Download MVTec AD separately and either place it at `Code/Data` or `Data`, or
set `MVTEC_DATA_ROOT` to its absolute path.  The loader expects the standard
layout:

```text
<mvtec_root>/<category>/train/good/*.png
<mvtec_root>/<category>/test/<defect_type>/*.png
<mvtec_root>/<category>/ground_truth/<defect_type>/*.png
```

The dataset is not committed to this repository.

## Main experiment

The thesis repository contains the integrated all-category notebook and the
durable JSON/CSV experiment records. This package intentionally does not ship
the MVTec data, notebooks, anomaly maps, or thesis build outputs.

## Evaluation from anomaly maps

The official MVTec evaluator is kept in the thesis repository. Once anomaly
maps have been generated, run it there with:

```bash
python <thesis-repository>/Code/EVAL/evaluate_experiment.py \
  --dataset_base_dir "$MVTEC_DATA_ROOT" \
  --anomaly_maps_dir <anomaly-map-root> \
  --output_dir <metrics-output>
```

The evaluator expects one TIFF map per test image at
`<maps>/<category>/test/<defect>/<image>.tiff`.  AU-PRO and image AUROC are
computed by the official evaluator; image scores use the maximum anomaly-map
value.  Do not substitute a different evaluator when comparing runs.

## Reproducibility limits

CNN pretrained weights are downloaded by the corresponding framework on the
first run and must be kept fixed.  Hardware, framework versions, and BLAS
threading can change runtime and may cause small floating-point differences.
The experiment ledger and JSON summaries are therefore the authoritative
records of the reported runs.
