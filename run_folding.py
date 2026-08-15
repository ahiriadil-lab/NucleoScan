"""
Folding orchestrator: runs ESMFold structure prediction for each fragment.

ESMFold model is loaded once (singleton) and reused across all fragments.
Supports N-terminal, C-terminal, and bidirectional fragment modes.
Multi-seed ensemble (10 seeds) with hierarchical clustering per fragment.

Uses:
  - run_esmfold.py : ESMFold inference wrapper
"""
import glob
import os
import sys

from config import (
    NSTRUCT, FRAGMENTS_DIR, STRUCTURES_DIR,
    MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH,
    RAW_METRICS_DIR, RANDOM_SEED,
    FRAGMENT_MODE, SLIDING_WINDOW_SIZES, SLIDING_WINDOW_STEP,
    ESMFOLD_FLANKING_GLYCINES,
    FRAGMENT_DIRECTION,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from run_esmfold import fold_fragment as esmfold_fold_fragment, get_model
    _HAS_ESMFOLD = True
except ImportError:
    _HAS_ESMFOLD = False


# ---------------------------------------------------------------------------
# Single-fragment folding
# ---------------------------------------------------------------------------

def run_abinitio(
    length: int,
    dry_run: bool = False,
    direction: str = "N",
    nstruct: int = None,
) -> list:
    """
    Generate structure(s) for the given fragment length using ESMFold.

    Parameters
    ----------
    length    : int   Fragment length (residues).
    dry_run   : bool  Print command without executing.
    direction : str   'N' (N-terminal) or 'C' (C-terminal).
    nstruct   : int   Number of seeds (default: config.NSTRUCT = 10).

    Returns
    -------
    List of output PDB paths.
    """
    if nstruct is None:
        nstruct = NSTRUCT

    if direction == "C":
        fasta   = os.path.join(FRAGMENTS_DIR, f"fragment_C{length:02d}.fasta")
        out_dir = os.path.join(STRUCTURES_DIR, f"fragC{length:02d}")
        tag     = f"fragC{length:02d}"
    else:
        fasta   = os.path.join(FRAGMENTS_DIR, f"fragment_{length:02d}.fasta")
        out_dir = os.path.join(STRUCTURES_DIR, f"frag{length:02d}")
        tag     = f"frag{length:02d}"

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(fasta):
        raise FileNotFoundError(f"Fragment FASTA not found: {fasta}")

    # Skip if already predicted
    primary = os.path.join(out_dir, "decoy_00000.pdb")
    if os.path.exists(primary):
        print(f"  [{tag}] decoy_00000.pdb already exists — skipping.", flush=True)
        return [primary]

    if dry_run:
        print(f"  [{tag}] DRY RUN: ESMFold ({nstruct} seeds, flanking={ESMFOLD_FLANKING_GLYCINES}G)")
        return []

    if not _HAS_ESMFOLD:
        raise RuntimeError("ESMFold not available. Run: pip install fair-esm[esmfold]")

    print(f"  [{tag}] Launching ESMFold ({nstruct} seeds, no MSA) …", flush=True)

    pdbs = esmfold_fold_fragment(
        fasta_path=fasta,
        out_dir=out_dir,
        n_seeds=nstruct,
        base_seed=RANDOM_SEED,
        flank_glycines=ESMFOLD_FLANKING_GLYCINES,
        score_output=os.path.join(out_dir, "scores.tsv"),
    )

    print(f"  [{tag}] {len(pdbs)} structure(s) generated.", flush=True)
    return pdbs


# ---------------------------------------------------------------------------
# Phase 3 — Enhancement 4: Sliding fragment folding
# ---------------------------------------------------------------------------

def run_sliding_fragment(
    start: int,
    window_size: int,
    dry_run: bool = False,
    nstruct: int = None,
) -> list:
    """
    Generate structure(s) for a sliding/internal fragment seq[start:start+window_size].

    FASTA: fragments/fragment_S{start:03d}_W{window_size:02d}.fasta
    Output: structures/fragS{start:03d}_W{window_size:02d}/decoy_00000.pdb
    """
    if nstruct is None:
        nstruct = NSTRUCT

    tag     = f"fragS{start:03d}_W{window_size:02d}"
    fasta   = os.path.join(FRAGMENTS_DIR, f"fragment_S{start:03d}_W{window_size:02d}.fasta")
    out_dir = os.path.join(STRUCTURES_DIR, tag)

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(fasta):
        raise FileNotFoundError(f"Sliding fragment FASTA not found: {fasta}")

    primary = os.path.join(out_dir, "decoy_00000.pdb")
    if os.path.exists(primary):
        print(f"  [{tag}] Already predicted — skipping.", flush=True)
        return [primary]

    if dry_run:
        print(f"  [{tag}] DRY RUN: ESMFold (start={start}, W={window_size}, {nstruct} seeds)")
        return []

    if not _HAS_ESMFOLD:
        raise RuntimeError("ESMFold not available. Run: pip install 'fair-esm[esmfold]'")

    print(f"  [{tag}] Launching ESMFold (start={start}, W={window_size}, {nstruct} seeds) …", flush=True)

    pdbs = esmfold_fold_fragment(
        fasta_path=fasta,
        out_dir=out_dir,
        n_seeds=nstruct,
        base_seed=RANDOM_SEED,
        flank_glycines=ESMFOLD_FLANKING_GLYCINES,
        score_output=os.path.join(out_dir, "scores.tsv"),
    )

    print(f"  [{tag}] {len(pdbs)} structure(s) generated.", flush=True)
    return pdbs


def _build_sliding_fragment_list(sequence_length: int = None,
                                  window_sizes: list = None,
                                  step: int = None) -> list:
    """Return list of (start, window_size) pairs for sliding fragments."""
    if window_sizes is None:
        window_sizes = SLIDING_WINDOW_SIZES
    if step is None:
        step = SLIDING_WINDOW_STEP
    if step < 1:
        step = 1

    # Derive sequence length from FASTA if not provided
    if sequence_length is None:
        from config import REFERENCE_FASTA
        from core.fragments import read_fasta
        if os.path.exists(REFERENCE_FASTA):
            seq = read_fasta(REFERENCE_FASTA)
            sequence_length = len(seq)
        else:
            sequence_length = MAX_FRAGMENT_LENGTH

    pairs = []
    for ws in sorted(set(window_sizes)):
        if ws > sequence_length:
            continue
        for start in range(0, sequence_length - ws + 1, step):
            pairs.append((start, ws))
    return pairs


# ---------------------------------------------------------------------------
# Batched query JSON — single OpenFold call for all fragments
# ---------------------------------------------------------------------------

def run_all_fragments_batched(
    lengths: list,
    dry_run: bool = False,
    nstruct: int = None,
) -> dict:
    """
    Run all fragments sequentially using ESMFold singleton model.

    For ESMFold, 'batched' simply means the model is loaded once before the
    loop (via the singleton pattern) and reused for each fragment — avoiding
    repeated weight loading. This is the default behaviour anyway.

    Returns
    -------
    dict mapping length -> list of generated PDB paths.
    """
    if nstruct is None:
        nstruct = NSTRUCT

    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    results = {}

    if not dry_run:
        if not _HAS_ESMFOLD:
            raise RuntimeError("ESMFold not available. Run: pip install 'fair-esm[esmfold]'")
        get_model()  # load once

    for length in lengths:
        fasta_path = os.path.join(FRAGMENTS_DIR, f"fragment_{length:02d}.fasta")
        if not os.path.exists(fasta_path):
            print(f"  [frag{length:02d}] FASTA not found, skipping.", flush=True)
            continue

        out_dir = os.path.join(STRUCTURES_DIR, f"frag{length:02d}")
        primary = os.path.join(out_dir, "decoy_00000.pdb")
        if os.path.exists(primary):
            print(f"  [frag{length:02d}] Already predicted — skipping.", flush=True)
            results[length] = [primary]
            continue

        if dry_run:
            print(f"  [frag{length:02d}] DRY RUN: ESMFold ({nstruct} seeds)", flush=True)
            continue

        try:
            pdbs = esmfold_fold_fragment(
                fasta_path=fasta_path,
                out_dir=out_dir,
                n_seeds=nstruct,
                base_seed=RANDOM_SEED,
                flank_glycines=ESMFOLD_FLANKING_GLYCINES,
                score_output=os.path.join(out_dir, "scores.tsv"),
            )
            results[length] = pdbs
        except Exception as e:
            print(f"  [frag{length:02d}] ERROR: {e}", file=sys.stderr, flush=True)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    dry_run: bool = False,
    batched: bool = False,
    nstruct: int = None,
):
    """
    Run ESMFold structure prediction for all fragments.

    Supports N-terminal, C-terminal, or bidirectional (both) fragments.

    Parameters
    ----------
    dry_run : Print commands without executing.
    batched : (Deprecated for ESMFold; model loaded once as singleton)
    nstruct : Override number of seeds per fragment (default: 10).
    """
    os.makedirs(STRUCTURES_DIR, exist_ok=True)

    if nstruct is None:
        nstruct = NSTRUCT

    # Load ESMFold model once (singleton)
    if not dry_run and not batched:
        try:
            get_model()
        except Exception as e:
            print(f"ERROR loading ESMFold: {e}", file=sys.stderr, flush=True)
            raise

    # Phase 3 — Sliding mode
    if FRAGMENT_MODE == "sliding":
        pairs = _build_sliding_fragment_list()
        print(
            f"ESMFold sliding-fragment folding: {len(pairs)} fragments "
            f"(W={SLIDING_WINDOW_SIZES}, step={SLIDING_WINDOW_STEP}), "
            f"{nstruct} seed(s) each",
            flush=True,
        )
        for start, ws in pairs:
            tag = f"fragS{start:03d}_W{ws:02d}"
            try:
                run_sliding_fragment(start, ws, dry_run=dry_run, nstruct=nstruct)
            except FileNotFoundError as e:
                print(f"  [{tag}] SKIPPED: {e}", flush=True)
            except RuntimeError as e:
                print(f"  [{tag}] ERROR: {e}", file=sys.stderr, flush=True)
        print("All sliding fragments done.", flush=True)
        return

    lengths = list(range(MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH + 1))

    # Determine directions
    directions = []
    if FRAGMENT_DIRECTION in ("N", "both"):
        directions.append("N")
    if FRAGMENT_DIRECTION in ("C", "both"):
        directions.append("C")

    if not directions:
        directions = ["N"]

    # Sequential — one call per (length, direction) pair
    total_frags = len(lengths) * len(directions)
    print(
        f"ESMFold folding: {total_frags} fragments "
        f"({len(lengths)} length(s) × {len(directions)} direction(s)), "
        f"{nstruct} seed(s) each",
        flush=True,
    )

    for direction in directions:
        for length in lengths:
            if direction == "C":
                tag = f"fragC{length:02d}"
            else:
                tag = f"frag{length:02d}"

            try:
                run_abinitio(length, dry_run=dry_run, direction=direction, nstruct=nstruct)
            except FileNotFoundError as e:
                print(f"  [{tag}] SKIPPED: {e}", flush=True)
            except RuntimeError as e:
                print(f"  [{tag}] ERROR: {e}", file=sys.stderr, flush=True)

    print("All fragments done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ESMFold fragment folding orchestrator (multi-seed ensemble)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--batched", action="store_true",
                        help="Load model once then run all fragments (default behaviour)")
    parser.add_argument("--nstruct", type=int, default=None,
                        help="Override number of seeds per fragment (default: 10)")
    args = parser.parse_args()

    main(
        dry_run=args.dry_run,
        batched=args.batched,
        nstruct=args.nstruct,
    )
