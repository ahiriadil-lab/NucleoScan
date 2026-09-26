# NucleoScan

**Language-model prefix scanning for residue-level prioritization of
hydrogen/deuterium-exchange (HDX) protection.**

NucleoScan converts independently predicted amino-terminal protein fragments
into a per-residue **fragment-emergence score (FES)**. From a single sequence,
it ranks the residues whose structure emerges as sequence context grows, giving
a transparent, auditable shortlist for protection-focused experiments
(pulse-labeling HDX, mutagenesis, construct design).

> FES is a ranking score for structure-linked protection. It is **not** a
> folding probability, a kinetic trajectory, a folding time or evidence of
> transition-state / folding-nucleus membership. That would require independent
> Phi-values, Psi-values or committor calculations.

## Method

For a chain of length *L*:

1. **Prefix scan.** Every N-terminal prefix of length 3…*L* is predicted
   independently with ESMFold (15 Gly flanks on each side, removed afterwards;
   one deterministic model per prefix; no MSA). Prefix length is a
   sequence-perturbation coordinate, not elapsed time.
2. **Fragment score.** Each model is compared with the matching part of the
   reference structure (a supplied PDB, or the full-length ESMFold model):

   `fragment score = confidence gate × [2Q + backbone agreement + compactness] + normalized confidence`

   where Q is heavy-atom native-contact recovery (4.5 Å), backbone agreement
   comes from Cα RMSD, compactness combines hydrophobic fraction and Cα radius
   of gyration, and the pLDDT gate rejects or down-weights low-confidence
   fragments (thresholds depend on secondary-structure class).
3. **Residue score (FES).** Three min-max-normalized axes:

   `FES(r) = 0.35 × marginal(r) + 0.40 × intermediate-contact(r) + 0.25 × topology(r)`

   - *marginal*: change in fragment score when residue *r* is added;
   - *intermediate-contact*: residues in partially native-like fragments
     (Q ≈ 0.20–0.50), long-range contact fraction, contact-formation persistence;
   - *topology*: native contact degree, secondary-structure persistence,
     betweenness centrality.

   Weights and thresholds are fixed heuristics exposed in `config.py`; no
   model is fitted to experimental labels.

## Benchmark

Evaluated against Start2Fold HDX annotations (57 proteins, 182
protein-by-protection-class evaluations, 55–372 residues), with four baselines
and four ablations:

| Target | Median AUC-ROC (95 % bootstrap CI) | n proteins |
|---|---|---|
| Early kinetic protection | 0.711 (0.657–0.771) | 39 |
| Strong equilibrium protection | 0.762 (0.694–0.811) | 53 |

- **Where FES adds value:** at the head of the ranking. Median precision@K rises
  from 0.250 (confidence-only baseline) to 0.333 for early protection; the
  top-ranked set improves in 24/39 proteins (9 worse, 6 tied).
- **Honest control:** a confidence-only baseline gives comparable whole-ranking
  AUC-ROC (0.720, P = 0.179), so FES should be used to prioritize the top of a
  list, not as a stronger global classifier.
- **Applicability domain:** performance declines with chain length
  (Spearman ρ = −0.51, P = 0.001). Best suited to small, single-domain proteins.

A manuscript describes the method and benchmark.

## Requirements

- Python 3.10 or 3.12
- Linux and a CUDA-capable NVIDIA GPU
- Approximately 8 GB GPU memory

## Installation

```bash
git clone https://github.com/ahiriadil-lab/NucleoScan.git
cd NucleoScan
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "fair-esm[esmfold]"
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime before installing the
remaining dependencies.

## Analyze any sequence

Pass an amino-acid sequence directly:

```bash
python nucleoscan.py \
  --sequence MKTAYIAKQRQISFVKSHFSRQ \
  --name my_protein
```

Or use a single-sequence FASTA file:

```bash
python nucleoscan.py --fasta protein.fasta --name my_protein
```

When no reference structure is supplied, NucleoScan predicts a full-length
reference with ESMFold. An existing structure can be provided explicitly:

```bash
python nucleoscan.py \
  --fasta protein.fasta \
  --reference-pdb protein.pdb \
  --name my_protein
```

Results are written to `results/` and generated structures to `structures/`.

## Options

```bash
python nucleoscan.py --fasta protein.fasta --bidirectional
python nucleoscan.py --fasta protein.fasta --run-label experiment_1
python nucleoscan.py --sequence MKTAYIAKQRQISFVKSHFSRQ --dry-run
```

Outputs in `results/<name>/`: `residue_scores.csv` (FES `nucleus_score`, `is_nucleus` flag, per-axis scores and weights) and `fragment_metrics.csv` (per-prefix Q, RMSD, compactness, pLDDT).

## License

[MIT](LICENSE)
