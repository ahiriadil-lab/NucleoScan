"""
NucleoScan: fragment-emergence scores for any protein sequence.

  1  Write FASTA and get a reference structure (given PDB or full-length ESMFold)
  2  Generate N-terminal fragment FASTAs
  3  Predict each fragment with ESMFold (skips existing decoys)
  4  Analyze fragments, score residues, write CSVs to results/{name}/

Usage:
  python nucleoscan.py --sequence MKTAYIAKQRQISFVKSHFSRQ --name query
  python nucleoscan.py --fasta protein.fasta [--reference-pdb ref.pdb]
  python nucleoscan.py --sequence ACDEFG --dry-run
"""
import argparse
import importlib
import logging
import os
import shutil
import sys
import warnings

import pandas as pd

import config as _cfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "nucleoscan.log"), mode="a"),
    ],
)
log = logging.getLogger("nucleoscan")

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZOU")


def _normalize_sequence(sequence: str) -> str:
    """Return a validated, whitespace-free amino-acid sequence."""
    normalized = "".join(sequence.split()).upper()
    if len(normalized) < 3:
        raise ValueError("A protein sequence must contain at least 3 residues.")
    invalid = sorted(set(normalized) - _AMINO_ACIDS)
    if invalid:
        raise ValueError(
            "Unsupported amino-acid code(s): " + ", ".join(invalid)
        )
    return normalized


def _read_sequence_fasta(path: str) -> str:
    """Read exactly one protein sequence from a FASTA file."""
    records = []
    current = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        records.append("".join(current))
    if len(records) != 1:
        raise ValueError("The FASTA file must contain exactly one sequence.")
    return _normalize_sequence(records[0])


def _safe_name(name: str) -> str:
    """Create a filesystem-safe identifier for a custom sequence."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    safe = safe.strip("._-")
    if not safe:
        raise ValueError("The sequence name must contain a letter or number.")
    return safe


def _reference_residue_count(pdb_path: str) -> int:
    """Count standard amino-acid residues in the first PDB chain."""
    from Bio import PDB
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("reference", pdb_path)
    chain = next(iter(structure[0].get_chains()))
    return sum(1 for residue in chain.get_residues()
               if PDB.is_aa(residue, standard=True))


def _patch_config(name, n_res, results_dir, structures_dir, fragments_dir,
                  raw_dir, n_decoys, fasta_path, reference_pdb, direction):
    """Set run-specific config values, then reload modules that read them."""
    _cfg.PDB_ID = name
    _cfg.REFERENCE_PDB = reference_pdb
    _cfg.REFERENCE_FASTA = fasta_path
    _cfg.RESULTS_DIR = results_dir
    _cfg.STRUCTURES_DIR = structures_dir
    _cfg.FRAGMENTS_DIR = fragments_dir
    _cfg.RAW_METRICS_DIR = raw_dir
    _cfg.MIN_FRAGMENT_LENGTH = 3
    _cfg.MAX_FRAGMENT_LENGTH = n_res
    _cfg.NSTRUCT = n_decoys
    _cfg.SCORE_THRESHOLD = _cfg.adaptive_score_threshold(n_res)
    _cfg.ALIGNMENT_MAP = None
    _cfg.PDB_N_RES = n_res
    _cfg.FRAGMENT_DIRECTION = direction
    for mod in ("core.analyze", "modules.contact_profile", "core.score",
                "core.fragments"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])


def _reference_structure(sequence, fasta_path, results_dir, reference_pdb,
                         dry_run, n_decoys):
    """Return the reference PDB path: the given file or a full-length ESMFold model."""
    if reference_pdb:
        reference_pdb = os.path.abspath(reference_pdb)
        if not os.path.isfile(reference_pdb):
            raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")
        n_pdb = _reference_residue_count(reference_pdb)
        if n_pdb != len(sequence):
            raise ValueError(
                f"The reference PDB contains {n_pdb} residues, "
                f"but the sequence contains {len(sequence)}."
            )
        local = os.path.join(results_dir, "reference.pdb")
        if reference_pdb != os.path.abspath(local):
            shutil.copy2(reference_pdb, local)
        return local

    reference_dir = os.path.join(results_dir, "reference_model")
    predicted = os.path.join(reference_dir, "decoy_00000.pdb")
    if dry_run or os.path.exists(predicted):
        return predicted

    from run_esmfold import fold_fragment
    predictions = fold_fragment(
        fasta_path=fasta_path,
        out_dir=reference_dir,
        n_seeds=n_decoys,
        base_seed=_cfg.RANDOM_SEED,
        flank_glycines=_cfg.ESMFOLD_FLANKING_GLYCINES,
    )
    if not predictions:
        raise RuntimeError("ESMFold did not produce a usable full-length reference.")
    return predictions[0]


def step3_fold_fragments(direction, n_decoys, dry_run=False):
    """Predict every fragment FASTA with ESMFold (one model load, existing decoys kept)."""
    from run_esmfold import fold_fragment, get_model
    if not dry_run:
        get_model()
    lengths = range(_cfg.MIN_FRAGMENT_LENGTH, _cfg.MAX_FRAGMENT_LENGTH + 1)
    for d in ("N", "C") if direction == "both" else (direction,):
        prefix = "C" if d == "C" else ""
        for length in lengths:
            tag = f"frag{prefix}{length:02d}"
            fasta = os.path.join(_cfg.FRAGMENTS_DIR, f"fragment_{prefix}{length:02d}.fasta")
            out_dir = os.path.join(_cfg.STRUCTURES_DIR, tag)
            if not os.path.exists(fasta):
                log.warning(f"  [{tag}] FASTA not found, skipping.")
            elif os.path.exists(os.path.join(out_dir, "decoy_00000.pdb")):
                log.info(f"  [{tag}] already predicted, skipping.")
            elif dry_run:
                log.info(f"  [{tag}] DRY RUN: ESMFold ({n_decoys} seeds)")
            else:
                fold_fragment(
                    fasta_path=fasta, out_dir=out_dir, n_seeds=n_decoys,
                    base_seed=_cfg.RANDOM_SEED,
                    flank_glycines=_cfg.ESMFOLD_FLANKING_GLYCINES,
                    score_output=os.path.join(out_dir, "scores.tsv"),
                )


def step4_analyze(raw_dir, bidirectional=False):
    """Analyze fragment ensembles; return (fragment_df, residue_df)."""
    from core.analyze import (_load_traj, compute_native_contacts, analyze_fragment,
                              compute_relative_contact_order, compute_reference_ss)
    from core.score import compute_fragment_scores, score_residues

    os.makedirs(raw_dir, exist_ok=True)
    lengths = range(_cfg.MIN_FRAGMENT_LENGTH, _cfg.MAX_FRAGMENT_LENGTH + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_traj = _load_traj(_cfg.REFERENCE_PDB)
        native_pairs, total_native = compute_native_contacts(ref_traj)
        rco = compute_relative_contact_order(ref_traj)
        ref_ss = compute_reference_ss(ref_traj)

        metrics_n = [m for m in (
            analyze_fragment(n, ref_traj, native_pairs, total_native, save_raw=True,
                             raw_dir=raw_dir, direction="N", ref_ss=ref_ss)
            for n in lengths) if m is not None]
        if not metrics_n:
            log.warning("  [4] No N-terminal structures found.")
            return pd.DataFrame(), pd.DataFrame()
        fragment_df = compute_fragment_scores(metrics_n)

        c_fragment_df = None
        if bidirectional:
            metrics_c = [m for m in (
                analyze_fragment(n, ref_traj, native_pairs, total_native,
                                 save_raw=False, raw_dir=raw_dir, direction="C",
                                 ref_ss=ref_ss)
                for n in lengths) if m is not None]
            if metrics_c:
                c_fragment_df = compute_fragment_scores(metrics_c)

    method = "bidirectional" if bidirectional else _cfg.SCORING_METHOD
    if method == "combined":  # 'combined' needs an importance table; marginal scoring otherwise
        method = "marginal"

    residue_df = score_residues(
        fragment_df,
        max_length=_cfg.MAX_FRAGMENT_LENGTH,
        method=method,
        c_fragment_df=c_fragment_df,
        ref_pdb_path=_cfg.REFERENCE_PDB,
        relative_contact_order=rco,
    )
    nucleus = residue_df[residue_df["is_nucleus"]]["residue"].tolist()
    log.info(f"  [4] {len(metrics_n)} fragments — method={method} "
             f"— high-score candidates: {nucleus}")
    return fragment_df, residue_df


def run_protein(name, sequence, n_decoys=1, reference_pdb=None, dry_run=False,
                bidirectional=False, run_label=None):
    suffix = f"_{run_label}" if run_label else ""
    results_dir = os.path.join(BASE_DIR, f"results{suffix}", name)
    structures_dir = os.path.join(BASE_DIR, f"structures{suffix}", name)
    fragments_dir = os.path.join(BASE_DIR, "fragments", name)
    raw_dir = os.path.join(results_dir, "raw")
    for d in (results_dir, structures_dir, fragments_dir):
        os.makedirs(d, exist_ok=True)

    sequence = _normalize_sequence(sequence)
    n_res = len(sequence)
    log.info(f"\n{'=' * 60}\n  PROTEIN: {name}  {n_res} aa\n{'=' * 60}")

    fasta_path = os.path.join(results_dir, f"{name}.fasta")
    if os.path.exists(fasta_path):
        if _read_sequence_fasta(fasta_path) != sequence:
            raise ValueError(f"{fasta_path} already contains a different sequence; "
                             "choose another --name.")
    else:
        with open(fasta_path, "w", encoding="utf-8") as handle:
            handle.write(f">{name}\n{sequence}\n")

    direction = "both" if bidirectional else "N"
    _patch_config(name, n_res, results_dir, structures_dir, fragments_dir, raw_dir,
                  n_decoys, fasta_path, None, direction)
    reference = _reference_structure(sequence, fasta_path, results_dir,
                                     reference_pdb, dry_run, n_decoys)
    _cfg.REFERENCE_PDB = reference
    log.info(f"  [1] Reference → {reference}")
    _patch_config(name, n_res, results_dir, structures_dir, fragments_dir, raw_dir,
                  n_decoys, fasta_path, reference, direction)

    from core.fragments import main as gen_main
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_main(direction=direction)
    log.info(f"  [2] Fragment FASTAs written to {fragments_dir}")

    step3_fold_fragments(direction, n_decoys, dry_run)
    log.info(f"  [3] ESMFold prediction complete in {structures_dir}")
    if dry_run:
        return None

    fragment_df, residue_df = step4_analyze(raw_dir, bidirectional=bidirectional)
    if fragment_df.empty:
        return None
    fragment_df.to_csv(os.path.join(results_dir, "fragment_metrics.csv"), index=False)
    residue_df.to_csv(os.path.join(results_dir, "residue_scores.csv"), index=False)
    log.info(f"  Results saved to {results_dir}")
    return residue_df


def main():
    parser = argparse.ArgumentParser(
        description="NucleoScan — compute fragment-emergence scores for a protein sequence."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", metavar="AA", help="Raw amino-acid sequence")
    group.add_argument("--fasta", metavar="FILE", help="Single-sequence FASTA file")
    parser.add_argument("--name", help="Run name (default: FASTA stem or 'sequence')")
    parser.add_argument("--reference-pdb", metavar="FILE",
                        help="Reference PDB; otherwise ESMFold predicts one")
    parser.add_argument("--n-decoys", type=int, default=1,
                        help="ESMFold predictions per fragment (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print folding commands without executing")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Also fold C-terminal fragments and use bidirectional scoring")
    parser.add_argument("--low-mem", action="store_true",
                        help="Enable the low-memory GPU preset")
    parser.add_argument("--scoring-method",
                        choices=["combined", "ab_initio", "marginal", "zscore",
                                 "sliding_window", "bidirectional", "native", "unified"],
                        help="Override SCORING_METHOD from config for this run")
    parser.add_argument("--run-label",
                        help="Label for this run (results_{label}/, structures_{label}/)")
    args = parser.parse_args()

    if args.low_mem:
        _cfg.OPENFOLD_LOW_MEM = True
    if args.scoring_method:
        _cfg.SCORING_METHOD = args.scoring_method

    try:
        sequence = (_normalize_sequence(args.sequence) if args.sequence
                    else _read_sequence_fasta(args.fasta))
        default_name = (os.path.splitext(os.path.basename(args.fasta))[0]
                        if args.fasta else "sequence")
        run_protein(_safe_name(args.name or default_name), sequence,
                    n_decoys=args.n_decoys, reference_pdb=args.reference_pdb,
                    dry_run=args.dry_run, bidirectional=args.bidirectional,
                    run_label=args.run_label)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
