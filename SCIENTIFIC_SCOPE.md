# Scientific scope and terminology

## What NucleoScan reports

NucleoScan's continuous per-residue output,
stored in the legacy `nucleus_score` column, is interpreted as a
**fragment-emergence score (FES)**. FES integrates:

- the change in fragment score when an N-terminal prefix is extended;
- native-contact organization in an operational intermediate-Q window; and
- descriptors of the native contact network.

The score is intended to rank candidates associated with structure-linked HDX
protection. The thresholded `is_nucleus` column is retained for file-format
compatibility and is an exploratory candidate flag.

## What NucleoScan does not establish

- **Prefix length is not folding time.** Structures at lengths L and L + 1 are
  predicted independently. The scan is a sequence-context perturbation series,
  not a kinetic trajectory.
- **Start2Fold protection is not folding-nucleus ground truth.** Early HDX
  protection may overlap kinetically important residues but does not by itself
  demonstrate stabilization of the rate-limiting transition state.
- **The intermediate-Q window is not a transition-state ensemble.** Internal
  names such as `TSE_Q_LO`, `extract_tse_frames` and `q_tse` are preserved for
  backward compatibility. They denote an operational intermediate-contact
  subset unless validated by a kinetic committor.
- **Model confidence is a strong baseline.** The full score does not improve
  global Start2Fold EARLY AUC-ROC over the pLDDT-only baseline in the reported
  benchmark. Any top-ranking advantage should be treated as exploratory until
  independently replicated.

## Evidence required for a folding-nucleus interpretation

A literal folding-nucleus claim requires an independent comparison with
mutation-resolved Φ-values, pairwise Ψ-values or a transition-state ensemble
defined through committor calculations. Such an analysis must document the
experimental construct, residue-number mapping, mutation-specific stability
change, uncertainty and inclusion criteria. Unmeasured residues must not be
treated as experimental negatives.

The bundled `validation/phi_values.py` utility is exploratory infrastructure.
Its fallback tables are not a publication-grade curated benchmark and do not,
on their own, validate folding-nucleus membership.

## Conceptual background

The distinction among structural occupancy, Φ-values, folding probability and
transition-state membership is discussed directly in:

- Faísca (2009), *J. Phys.: Condens. Matter* 21, 373102,
  <https://doi.org/10.1088/0953-8984/21/37/373102>;
- Faísca et al. (2008), *J. Chem. Phys.* 129, 095108,
  <https://doi.org/10.1063/1.2973624>; and
- Travasso, Faísca and Rey (2010), *J. Chem. Phys.* 133, 125102,
  <https://doi.org/10.1063/1.3485286>.

For frustration and structure-based simulation routes relevant to independent
testing, see Contessoto et al. (2013), <https://doi.org/10.1002/prot.24309>, and
SMOG 2/OpenSMOG (de Oliveira et al., 2022),
<https://doi.org/10.1002/pro.4209>.

## Compatibility policy

Renaming public fields and functions would break existing result archives and
downstream scripts. The following legacy identifiers therefore remain in the
code while their scientific interpretation is narrowed:

| Legacy identifier | Operational interpretation |
|---|---|
| `nucleus_score` | fragment-emergence score |
| `is_nucleus` | thresholded candidate flag |
| `UNIFIED_W_KINETIC` | fragment-addition marginal weight |
| `TSE_Q_LO`, `TSE_Q_HI` | intermediate-Q window bounds |
| `extract_tse_frames` | select intermediate-contact frames |
| `q_tse` | contact score within that operational window |
