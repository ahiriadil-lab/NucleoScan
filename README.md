# FoldNucleus

A structure-guided pipeline for ranking residue-level structural emergence in
single-domain proteins using ESMFold fragment predictions, native-contact
recovery, backbone agreement, compactness, model confidence and native topology.

> **Scientific scope.** FoldNucleus is the historical software name. The
> continuous output is interpreted as a **fragment-emergence score (FES)**: a
> ranking for structure-linked HDX protection propensity. Start2Fold HDX labels
> are not a ground truth for folding-nucleus membership, and an N-terminal
> prefix scan is not a folding-time trajectory. A literal transition-state or
> folding-nucleus interpretation requires independent Φ/Ψ measurements or
> committor-based simulations. See [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md).

## Overview

The pipeline implements a *delta scoring* strategy: for each N-terminal prefix
length L, ESMFold predicts an independent fragment structure and scores it
against the corresponding region of a native reference structure. The
per-residue contribution δ(i) = score(L) − score(L−1) measures the association
between adding residue i and a change in predicted structural organization.
It does not represent elapsed time or a causal folding step. Residues above an
adaptive threshold are reported as high-scoring candidates for follow-up.

Three structural quality metrics are fused into a composite score per fragment:

| Metric | Description | Direction |
|--------|-------------|-----------|
| Backbone RMSD | Cα RMSD vs. native (nm) | lower is better |
| Q-value | Fraction of native contacts formed | higher is better |
| Hydrophobic Rg | Radius of gyration of hydrophobic core (nm) | lower is better |

Four principal scoring methods are available (`SCORING_METHOD` in `config.py`):
`marginal`, `combined`, `ab_initio`, and `unified` (default, recommended).

## Requirements

**Python:** 3.10 or 3.12 (tested)

**GPU:** CUDA-capable GPU required for ESMFold inference.
Tested on NVIDIA RTX 3080 (8 GB VRAM) with CUDA 11.8 / 12.1.

**PyTorch:** Install before ESMFold — must match your CUDA version:
```bash
# CUDA 11.8
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**ESMFold:**
```bash
pip install "fair-esm[esmfold]"
```
Model weights (~2.5 GB) are downloaded automatically on first inference.

**All other dependencies:**
```bash
pip install -r requirements.txt
```

## Data

The primary benchmark is the **Start2Fold** HDX database (Pancsa *et al.* 2016),
included as `data/start2fold_data.json`. It curates residue-level kinetic and
equilibrium hydrogen/deuterium-exchange protection classes. These annotations
measure exchange protection and may overlap kinetically important regions, but
they do not by themselves establish transition-state or folding-nucleus
membership.

The pipeline also supports an alternative dataset (`--database v3`) via
`data/two_state_folding_v3.xlsx` (included), and a merged mode (`--database merged`).

PDB structures are downloaded automatically from the RCSB on first run.

## Installation

```bash
git clone https://github.com/ahiriadil-lab/FoldNucleus.git
cd FoldNucleus

# 1. Install PyTorch (see Requirements above)
# 2. Install ESMFold
pip install "fair-esm[esmfold]"
# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Usage

**Run the full pipeline on all proteins:**
```bash
python run_dataset.py
```

**Run on specific proteins:**
```bash
python run_dataset.py --database v3 --proteins Ubiquitin CI2
```

**Run the exploratory Φ-value comparison and sensitivity analysis:**
```bash
python run_dataset.py --database v3 --proteins Ubiquitin --sensitivity --phi-compare
```

The Φ-value utility is analysis infrastructure, not completed independent
validation. Its bundled fallback tables are suitable for software smoke tests
only. Publication-grade use requires verification of the experimental construct,
residue mapping, mutation-level uncertainty and inclusion criteria.

**Skip advanced analysis (faster):**
```bash
python run_dataset.py --skip-advanced
```

**Dry run (print commands without executing):**
```bash
python run_dataset.py --dry-run
```

**Evaluate scores against Start2Fold HDX protection labels:**
```bash
python validation/start2fold.py --all-levels --results-dir results/
```

**Reproduce the four baseline comparisons:**
```bash
python validation/baselines.py --results-dir results/ \
  --structures-dir structures/ --pdb-dir data/pdb_cleaned/
```

This evaluates FES alongside coverage-mean per-residue pLDDT, native contact
degree, deterministic random scores and a constant all-residue score. It writes
per-protein metrics and matched comparisons to `results/baseline_validation/`.

## Configuration

All parameters are centralised in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCORING_METHOD` | `"unified"` | Scoring method: `marginal`, `combined`, `ab_initio`, `unified` |
| `MARGINAL_SIGMA_FACTOR` | `1.0` | σ multiplier for adaptive marginal threshold |
| `UNIFIED_W_KINETIC` | `0.35` | Legacy parameter name: weight of the fragment-addition marginal axis |
| `UNIFIED_W_COOPERATIVE` | `0.40` | Weight of the operational intermediate-contact axis |
| `UNIFIED_W_TOPOLOGICAL` | `0.25` | Unified method: weight of topological (structural importance) axis |
| `CONTACT_CUTOFF_NM` | `0.45` | Native contact distance threshold (nm) |
| `MIN_FRAGMENT_LENGTH` | `5` | Standalone default; `run_dataset.py` sets it to 3 |
| `MAX_FRAGMENT_LENGTH` | `25` | Standalone default; `run_dataset.py` sets it to the chain length |
| `RANDOM_SEED` | `42` | Legacy reproducibility seed; unchanged ESMFold inputs are deterministic |
| `PLDDT_REJECT_THRESHOLD` | `40.0` | Reject fragments with mean pLDDT below this value |

The marginal scoring threshold is computed per protein:
```
threshold = μ(δ) + adaptive_sigma_factor(N) × σ(δ)
```
where `adaptive_sigma_factor(N)` scales with protein length (see `config.py`).

## Outputs

Results are written to `results/<protein_name>/`:

| File | Description |
|------|-------------|
| `residue_scores.csv` | Per-residue FES values, legacy candidate flags, and raw metrics |
| `fragment_scores.csv` | Per-fragment composite quality scores (RMSD, Q, Hydro Rg) |
| `folding_nucleus.png` | Legacy filename: bar chart of FES values with adaptive threshold |
| `validation_report.csv` | Automated sanity checks |
| `structural_importance.csv` | Per-residue structural importance (contact degree, SS persistence, betweenness) |
| `sensitivity_heatmap.png` | Candidate-set composition vs. threshold parameter sweep |
| `dataset_summary.csv` | Cross-protein summary of candidate residues and metrics |

## Interpreting Results

The continuous `nucleus_score` field is retained for backward compatibility and
should be read as FES. The `is_nucleus` field is a heuristic thresholded
candidate flag, not a calibrated biological classification. If the selected
candidate fraction is unexpectedly large or small:

- **Too high (>50%):** Threshold too low — increase `MARGINAL_SIGMA_FACTOR`
  (e.g., from 1.0 to 1.5) or switch to `scoring_method = "unified"`.
- **Too low (<10%):** Threshold too high or insufficient fragment coverage —
  decrease `MARGINAL_SIGMA_FACTOR` or reduce `MIN_FRAGMENT_LENGTH`.

Use `--sensitivity` to visualize candidate-set composition as a function of the
threshold. Use `--phi-compare` only as an exploratory comparison after auditing
the experimental Φ-values and residue mapping.

The method is best supported for small, predominantly single-domain proteins.
Performance against early HDX protection declines with chain length, and the
N-terminal scan has a systematic blind spot for C-terminal or discontinuous
determinants.

## Reproducing reported benchmark results

Pipeline outputs are excluded from Git because they are large. A manuscript or
release that reports benchmark statistics must link a versioned archive
containing the complete per-fragment and per-residue outputs, configuration,
tables and figures. Code plus input annotations alone are insufficient to
reproduce the reported numerical results.

## Citation

If you use this pipeline, please cite:

```bibtex
@software{foldnucleus,
  author  = {Ahiri, Adil},
  title   = {FoldNucleus},
  year    = {2026},
  url     = {https://github.com/ahiriadil-lab/FoldNucleus}
}
```

For ESMFold:
> Lin, Z., Akin, H., Rao, R., et al. (2023).
> Evolutionary-scale prediction of atomic-level protein structure with a language model.
> *Science*, 379(6637), 1123–1130. https://doi.org/10.1126/science.ade2574

For the Start2Fold database:
> Pancsa, R., Varadi, M., Tompa, P., & Vranken, W. F. (2016).
> Start2Fold: a database of hydrogen/deuterium exchange data on protein folding and stability.
> *Nucleic Acids Research*, 44(D1), D429–D434.
> https://doi.org/10.1093/nar/gkv1185

For the nucleation–condensation mechanism:
> Fersht, A. R. (1995).
> Optimization of rates of protein folding: the nucleation–condensation mechanism and its implications.
> *PNAS*, 92(24), 10869–10873. https://doi.org/10.1073/pnas.92.24.10869

## License

MIT — see `LICENSE`.
