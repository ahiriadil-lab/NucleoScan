<p align="center">
  <img src="assets/nucleoscan-banner.png" alt="NucleoScan protein-fragment emergence banner" width="100%">
</p>

# NucleoScan

[![Quality checks](https://github.com/ahiriadil-lab/NucleoScan/actions/workflows/quality.yml/badge.svg)](https://github.com/ahiriadil-lab/NucleoScan/actions/workflows/quality.yml)
[![CodeQL](https://github.com/ahiriadil-lab/NucleoScan/actions/workflows/codeql.yml/badge.svg)](https://github.com/ahiriadil-lab/NucleoScan/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/ahiriadil-lab/NucleoScan?display_name=tag)](https://github.com/ahiriadil-lab/NucleoScan/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB.svg)](https://www.python.org/)
[![CITATION.cff](https://img.shields.io/badge/citation-CFF-green.svg)](CITATION.cff)

NucleoScan is a structure-guided research pipeline for ranking residue-level
structural emergence in small, predominantly single-domain proteins. It combines
independent ESMFold fragment predictions with native-contact recovery, backbone
agreement, hydrophobic compactness, model confidence and native topology.

> [!IMPORTANT]
> The continuous output is a **fragment-emergence score (FES)**. It is not a
> folding probability, a kinetic trajectory or direct evidence of
> transition-state membership. See [Scientific scope](SCIENTIFIC_SCOPE.md).

## Highlights

- Single-sequence structure prediction with ESMFold; no MSA is required.
- N-terminal prefix scanning, with optional bidirectional analysis.
- Per-fragment RMSD, local native-contact recovery, hydrophobic radius of
  gyration and pLDDT-aware scoring.
- Per-residue marginal, combined, ab initio and unified scoring modes.
- Start2Fold HDX comparison, transparent baselines and automated sanity checks.
- Versioned metadata, input checksums and CPU-compatible continuous integration.

## Method overview

```text
Dataset and reference PDB
          │
          ▼
Sequence extraction and fragment generation (L = 3 … N)
          │
          ▼
Independent ESMFold prediction for every fragment
          │
          ▼
RMSD + local Q + hydrophobic Rg + pLDDT
          │
          ▼
Composite fragment score and per-residue Δ-score
          │
          ▼
Unified FES + adaptive candidate threshold
          │
          ▼
CSV outputs, figures, sanity checks and HDX comparisons
```

For a prefix ending at residue *i*, the marginal signal is

```text
δ(i) = score(fragment 1…i) − score(fragment 1…i−1)
```

The structures at lengths *i* and *i−1* are predicted independently. Therefore,
δ(i) measures an association with structural emergence, not elapsed folding time.

## Requirements

- Python 3.10 or 3.12
- Linux recommended
- CUDA-capable NVIDIA GPU for ESMFold inference
- Approximately 8 GB GPU memory for the tested default configuration

The lightweight tests and metadata checks do not require a GPU.

## Installation

```bash
git clone https://github.com/ahiriadil-lab/NucleoScan.git
cd NucleoScan

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with the local CUDA runtime, then install
ESMFold and the remaining dependencies:

```bash
pip install "fair-esm[esmfold]"
pip install -r requirements.txt
```

The [PyTorch installation selector](https://pytorch.org/get-started/locally/)
provides the correct command for each CUDA version.

## Quick start

Run one protein first to verify the environment:

```bash
python run_dataset.py --proteins TrpCage --skip-advanced
```

Run the complete Start2Fold dataset:

```bash
python run_dataset.py --database start2fold
```

Useful variants:

```bash
# Restrict the run and label its outputs
python run_dataset.py --database v3 --proteins Ubiquitin CI2 \
  --run-label publication_v1_1_0

# Add C-terminal fragments
python run_dataset.py --proteins Ubiquitin --bidirectional

# Inspect commands without running ESMFold
python run_dataset.py --dry-run
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahiriadil-lab/NucleoScan/blob/main/RUN_ON_COLAB.ipynb)

## Scoring

The default `unified` score combines three normalized axes:

| Axis | Signal | Default weight |
|---|---|---:|
| Fragment addition | Marginal change when residue *i* is added | 0.35 |
| Intermediate contacts | Contact organization in an operational Q window | 0.40 |
| Native topology | Contact degree, secondary structure and betweenness | 0.25 |

Weights are redistributed when an axis lacks adequate sampling. Candidate flags
use a protein-length-adaptive threshold:

```text
threshold = mean(FES) + adaptive_sigma_factor(N) × standard_deviation(FES)
```

The historical output columns `nucleus_score` and `is_nucleus` remain unchanged
for file-format compatibility. They should be read as FES and exploratory
candidate flag, respectively.

## Data

The default benchmark contains 209 Start2Fold records covering 57 PDB entries.
The repository also includes the project-specific two-state folding table used
by `--database v3`. Data provenance, third-party terms and SHA-256 checksums are
documented in [DATA.md](DATA.md) and [DATA_MANIFEST.tsv](DATA_MANIFEST.tsv).

Reference structures missing from the local cache are retrieved from the RCSB
Protein Data Bank during a run.

## Outputs

Each protein is written to `results/<protein_name>/`:

| File | Contents |
|---|---|
| `fragment_scores.csv` | Per-fragment metrics and composite scores |
| `residue_scores.csv` | FES, component axes and candidate flags |
| `validation_report.csv` | Automated data and score sanity checks |
| `structural_importance.csv` | Native-topology importance metrics |
| `folding_nucleus.png` | Legacy filename for the FES profile |
| `composite_summary.png` | Summary of residue and fragment-level signals |

Large generated structures and results are intentionally excluded from Git.
Publication-specific output archives should be deposited separately with a
permanent identifier. See [Reproducibility](REPRODUCIBILITY.md).

## Validation and limitations

NucleoScan can compare FES with six Start2Fold HDX protection classes and with
four transparent baselines:

```bash
python validation/start2fold.py --all-levels --results-dir results/

python validation/baselines.py --results-dir results/ \
  --structures-dir structures/ --pdb-dir data/pdb_cleaned/
```

Important limitations:

- Start2Fold protection is not folding-nucleus ground truth.
- Prefix length is sequence context, not physical time.
- An intermediate-Q window is not a committor-defined transition-state ensemble.
- N-terminal scanning is less informative for C-terminal or discontinuous
  structural determinants.
- The method is best supported for small, predominantly single-domain proteins.

## Repository structure

```text
core/          structural analysis and residue scoring
modules/       contact, cooperativity, importance and sensitivity analyses
validation/    HDX comparisons, baselines and sanity checks
viz/           publication-oriented plots
data/          bundled benchmark inputs and adapters
tests/         CPU-only smoke and metadata tests
```

## Reproducibility and contribution

- [Reproducibility guide](REPRODUCIBILITY.md)
- [Scientific scope](SCIENTIFIC_SCOPE.md)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Citation

GitHub exposes the repository citation through [CITATION.cff](CITATION.cff).
Until a journal article or archived release DOI is available, cite the software
release directly:

```bibtex
@software{ahiri_nucleoscan_2026,
  author  = {Ahiri, Adil},
  title   = {NucleoScan},
  version = {1.1.0},
  year    = {2026},
  url     = {https://github.com/ahiriadil-lab/NucleoScan},
  note    = {ORCID: 0000-0001-5170-6538}
}
```

Please also cite ESMFold and Start2Fold when their predictions or annotations
are used. Full references are included in [CITATION.cff](CITATION.cff).

## License

The software is released under the [MIT License](LICENSE). Bundled third-party
data retain their original terms; see [DATA.md](DATA.md).
