"""Central configuration for NucleoScan."""
import os

# Folding engine
FOLDING_ENGINE = "esmfold"

# ESMFold inference settings
ESMFOLD_FLANKING_GLYCINES = 15        # Flanking residues on each side (15G left + 15G right)
ESMFOLD_CHUNK_SIZE = None             # ESMFold chunk size for long sequences (None = auto)
RANDOM_SEED = 42                      # Reproducibility seed

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAGMENTS_DIR = os.path.join(BASE_DIR, "fragments")
STRUCTURES_DIR = os.path.join(BASE_DIR, "structures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Reference protein
PDB_ID = "1L2Y"            # Trp-cage (20 aa)
REFERENCE_PDB = os.path.join(DATA_DIR, f"{PDB_ID}.pdb")
REFERENCE_FASTA = os.path.join(DATA_DIR, f"{PDB_ID}.fasta")

# Fragment settings
MIN_FRAGMENT_LENGTH = 5
MAX_FRAGMENT_LENGTH = 25
FRAGMENT_DIRECTION = "both"             # "N" (N-terminal), "C" (C-terminal), or "both" (bidirectional)

# Overridden by ``run_dataset --n-decoys``.
NSTRUCT = 1

# Analysis settings
CONTACT_CUTOFF_NM = 0.45   # Native contact cutoff in nm
HYDROPHOBIC_AA = set("LIVFWMA")  # Hydrophobic amino acids

# Scoring settings

# Heuristic threshold for sliding-window and z-score methods.
SCORE_THRESHOLD = 2.1

# Available methods are dispatched in ``core.score.score_residues``.
SCORING_METHOD = "unified"

# Marginal-score threshold on the [0, 3] scale.
SCORE_THRESHOLD_MARGINAL = 0.1

# Ab initio contact-onset settings
CONTACT_ONSET_THRESHOLD = 0.30

EMERGENT_CONTACT_CONSISTENCY = 0.50
EMERGENT_CONTACT_DEGREE_ONSET = 2

# Adaptive threshold multiplier: mean + factor × standard deviation.
MARGINAL_SIGMA_FACTOR = 1.0

Q_WEIGHT = 2.0

# Minimum local-Q signal retained for marginal scoring.
MIN_Q_SIGNAL = 0.05

# Combined-score weights
SCORING_COMBINED_W_MARGINAL    = 0.25
SCORING_COMBINED_W_IMPORTANCE  = 0.75

# Unified-score weights. Public names remain stable for compatibility.
UNIFIED_W_KINETIC      = 0.35   # Axis A: fragment-addition marginal signal; not physical time
UNIFIED_W_COOPERATIVE  = 0.40   # Axis B: operational intermediate-contact window; not a validated TSE
UNIFIED_W_TOPOLOGICAL  = 0.25   # Axis C: structural importance (native topology)

# Intermediate-contact sub-weights
UNIFIED_COOP_W_TSE     = 0.40   # Contact signal in the intermediate-Q window
UNIFIED_COOP_W_LR      = 0.30   # long-range contact fraction
UNIFIED_COOP_W_COOP    = 0.30   # cooperativity correlation

# Operational intermediate-Q window; it is not a kinetic transition state.
TSE_Q_LO               = 0.20   # Standard intermediate-Q lower bound
TSE_Q_HI               = 0.50   # Standard intermediate-Q upper bound
TSE_Q_LO_WIDE          = 0.15   # Wide intermediate-Q lower bound (fallback)
TSE_Q_HI_WIDE          = 0.55   # Wide intermediate-Q upper bound (fallback)
TSE_MIN_FRAMES         = 20     # Minimum frames for the intermediate-contact axis
TSE_MIN_FRAMES_PARTIAL = 10     # Partial sampling (reduced axis weight)

SEQ_SEP_LONG_RANGE     = 8      # |i-j| > this = long-range

# Contact maps
CONTACT_FORMATION_THRESHOLD = 0.5

# Structural importance: contact degree, secondary structure and betweenness.
IMPORTANCE_WEIGHTS = (1/3, 1/3, 1/3)
IMPORTANCE_CONTACT_THRESHOLD = 0.5

# Optional packing and burial terms
LOCAL_PACKING_WEIGHT = 0.0       # Weight in composite fragment score (0 = disabled)
HYDRO_BURIAL_WEIGHT  = 0.0       # Weight in composite fragment score (0 = disabled)
PACKING_RADIUS_NM    = 0.5       # Neighbour-counting radius (nm)

# Reference maximum SASA per residue type (Gly-X-Gly tripeptide, nm²)
# Source: Tien et al. 2013, Table 1 (theoretical max solvent-accessible areas)
MAX_SASA_PER_AA = {
    "ALA": 1.29, "ARG": 2.74, "ASN": 1.95, "ASP": 1.93, "CYS": 1.68,
    "GLN": 2.23, "GLU": 2.25, "GLY": 1.04, "HIS": 2.24, "ILE": 1.97,
    "LEU": 2.01, "LYS": 2.36, "MET": 2.24, "PHE": 2.40, "PRO": 1.59,
    "SER": 1.55, "THR": 1.72, "TRP": 2.85, "TYR": 2.63, "VAL": 1.74,
}

# pLDDT terms
PLDDT_WEIGHT          = 1.0   # Weight of mean pLDDT in composite fragment score (enabled for ESMFold)
PLDDT_GRADIENT_WEIGHT = 0.0   # Weight of per-residue ∂pLDDT/∂L in marginal scoring

# pLDDT confidence filtering
PLDDT_REJECT_THRESHOLD = 40.0                 # Default reject threshold (α/coil fragments)
PLDDT_REDUCED_WEIGHT_THRESHOLD = 60.0         # Default reduced-weight threshold
PLDDT_REDUCED_WEIGHT_FACTOR = 0.5             # Weight factor for reduced-weight range (×0.5)

# Secondary-structure-aware thresholds account for lower isolated β-strand confidence.
PLDDT_BETA_REJECT_THRESHOLD = 30.0            # β-dominant fragments: reject < 30
PLDDT_BETA_REDUCED_THRESHOLD = 45.0           # β-dominant fragments: reduced < 45
PLDDT_ALPHA_REJECT_THRESHOLD = 40.0           # α-dominant: unchanged
PLDDT_ALPHA_REDUCED_THRESHOLD = 60.0          # α-dominant: unchanged
PLDDT_COIL_REJECT_THRESHOLD = 50.0            # Coil/mixed: stricter (unstructured = noise)
PLDDT_COIL_REDUCED_THRESHOLD = 65.0           # Coil/mixed: stricter

# Exploratory FDR setting retained for compatibility.
FDR_TARGET = 0.10

# Multi-seed clustering
ENSEMBLE_CLUSTER_METHOD = "average"           # scipy.cluster.hierarchy linkage method
ENSEMBLE_RMSD_CUTOFF = 2.0                    # nm — cutoff distance for hierarchical clustering

# Contact-formation rate
CONTACT_RATE_WEIGHT          = 0.0   # Blend weight in marginal residue scoring
LONG_RANGE_CONTACT_SEPARATION = 12   # |i-j| > this → long-range contact
LONG_RANGE_CONTACT_WEIGHT    = 2.0   # Extra weight for long-range contacts

# Sliding fragments
FRAGMENT_MODE         = "nterminal"   # "nterminal" | "sliding"
SLIDING_WINDOW_SIZES  = [10, 15, 20]  # Window widths W (aa)
SLIDING_WINDOW_STEP   = 1             # Step size for start position i

# Fragment averaging
FRAGMENT_AVERAGING       = True   # Average scores across overlapping sliding fragments
PLDDT_WEIGHTED_AVERAGING = True   # Weight by fragment mean pLDDT

# Committor validation thresholds
COMMITTOR_ACCEPTANCE_RATE_LO = 0.10
COMMITTOR_ACCEPTANCE_RATE_HI = 0.60
COMMITTOR_DEFF_RATIO_THRESHOLD = 10.0
COMMITTOR_CROSSVAL_DQ_THRESHOLD = 0.10
PFOLD_TSE_LO = 0.40
PFOLD_TSE_HI = 0.60

# UniProt sequence settings (patched per-protein by run_dataset.py)
UNIPROT_CACHE_DIR = os.path.join(DATA_DIR, "uniprot")
DOMAIN_FASTA_DIR = os.path.join(DATA_DIR, "domain_fasta")
PDB_RAW_DIR = os.path.join(DATA_DIR, "pdb_raw")

# Set per protein by ``run_dataset.py``.
ALIGNMENT_MAP = None
PDB_N_RES = None

# Raw-metric cache
RAW_METRICS_DIR = os.path.join(RESULTS_DIR, "raw")


def adaptive_sigma_factor(n_residues: int) -> float:
    """
    Return MARGINAL_SIGMA_FACTOR appropriate for a protein of n_residues.

    Heuristic sigma multiplier used to control candidate-set size by length.

    Calibration:
      ≤40 aa  → 0.5  (typically a larger candidate fraction)
      ≤60 aa  → 0.75
      ≤80 aa  → 1.0
      >80 aa  → 1.5
    These bins are empirical software defaults, not a biological calibration.
    """
    if n_residues <= 40:
        return 0.5
    elif n_residues <= 60:
        return 0.75
    elif n_residues <= 80:
        return 1.0
    else:
        return 1.5


def adaptive_score_threshold(n_residues: int) -> float:
    """
    Return the legacy length-dependent score cutoff used for candidate flags.

    Calibration:
      ≤40 aa  → 2.1  (small proteins, ~30% cutoff on normalised scale)
      ≤100 aa → 1.8
      ≤200 aa → 1.5
      >200 aa → 1.2
    """
    if n_residues <= 40:
        return 2.1
    elif n_residues <= 100:
        return 1.8
    elif n_residues <= 200:
        return 1.5
    else:
        return 1.2
