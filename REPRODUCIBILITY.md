# Reproducibility guide

This guide separates software reproducibility from reproduction of a reported
benchmark. The repository contains source code and compact inputs; it does not
contain the large ESMFold structure sets or complete result archives underlying
a publication.

## 1. Check out a fixed version

```bash
git clone https://github.com/ahiriadil-lab/NucleoScan.git
cd NucleoScan
git checkout v1.1.0
```

Record the exact revision in every analysis:

```bash
git rev-parse HEAD
```

## 2. Create the environment

Use Python 3.10 or 3.12. Install a PyTorch build matching the system CUDA
runtime, followed by ESMFold and the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "fair-esm[esmfold]"
pip install -r requirements.txt
```

Capture the environment and hardware before a publication run:

```bash
python --version
python -m pip freeze
nvidia-smi
```

## 3. Verify inputs and software

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

The metadata tests validate `CITATION.cff`, the Colab notebook and all bundled
data checksums.

## 4. Run a smoke analysis

```bash
python run_dataset.py --proteins TrpCage --skip-advanced \
  --run-label smoke_v1_1_0
```

Inspect `results_smoke_v1_1_0/TrpCage/validation_report.csv` before launching a
dataset-scale calculation.

## 5. Run the benchmark

```bash
python run_dataset.py --database start2fold \
  --run-label publication_v1_1_0
```

Optional validation steps:

```bash
python validation/start2fold.py --all-levels \
  --results-dir results_publication_v1_1_0/

python validation/baselines.py \
  --results-dir results_publication_v1_1_0/ \
  --structures-dir structures_publication_v1_1_0/ \
  --pdb-dir data/pdb_cleaned/
```

## 6. Archive publication outputs

A publication archive should contain:

- the Git commit and release tag;
- `CITATION.cff`, `VERSION`, `config.py` and the input manifest;
- the exact dependency list and GPU/driver information;
- complete per-fragment and per-residue CSV outputs;
- validation and baseline tables;
- generated figures and logs;
- checksums for every archived file.

Deposit the archive in a repository that issues a persistent identifier, then
add the DOI to `CITATION.cff`, the GitHub release and the manuscript data
availability statement.

## Determinism and interpretation

`RANDOM_SEED` defaults to 42. Standard ESMFold inference is deterministic for an
unchanged environment and input, but GPU kernels and dependency changes may
still introduce numerical differences. Record the full environment rather than
relying on the seed alone.

Prefix structures are independent predictions. They are not chronological
frames, and Start2Fold protection labels are not transition-state ground truth.
See `SCIENTIFIC_SCOPE.md` before interpreting biological results.
