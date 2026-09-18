"""
ESMFold inference wrapper for fragment structure prediction.

Generates single-sequence protein structure predictions with ESMFold.

Protocol:
  - Loads ESMFold model once (singleton, reused for all fragments)
  - Reads fragment FASTA file
  - Adds flanking glycines (15G each side) for robustness
  - Runs n_seeds predictions with different random seeds
  - Clusters structures (hierarchical clustering on RMSD)
  - Selects centroid of majority cluster
  - Strips flanking glycines from output PDB
  - Extracts pLDDT confidence from B-factor column
  - Rejects if mean pLDDT < 40 (PLDDT_REJECT_THRESHOLD)
  - Writes confidence TSV

Usage:
    python run_esmfold.py <fasta> <out_dir> \\
        [--n-seeds 10] [--base-seed 42] [--flank-glycines 15]
"""

import os
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Must be set before "import esm"
os.environ["OPENFOLD_FORCE_FP16"] = "1"

# Configure CUDA allocation before loading PyTorch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    import esm
    _HAS_ESMFOLD = True
except ImportError:
    _HAS_ESMFOLD = False

_MODEL = None


def get_model():
    """Load ESMFold v1 model (singleton). Loads weights once, reuses for all fragments."""
    global _MODEL
    if not _HAS_ESMFOLD:
        raise ImportError("ESMFold not installed. Run: pip install fair-esm[esmfold]")
    if _MODEL is None:
        print("  [ESMFold] Loading model weights (first run only) …", flush=True)
        _MODEL = esm.pretrained.esmfold_v1()
        _MODEL = _MODEL.eval().half().cuda()  # float16 to fit in 8GB VRAM
        print("  [ESMFold] Model ready (float16 GPU).", flush=True)
    return _MODEL


def _read_fasta(fasta_path: str) -> Tuple[str, str]:
    """Return (header, sequence) from a single-record FASTA file."""
    header = ""
    seq_lines = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            else:
                seq_lines.append(line)
    return header, "".join(seq_lines)


def _add_flanking_glycines(sequence: str, n_flank: int = 15) -> str:
    """Add flanking glycines to sequence: (15G) + sequence + (15G)."""
    return "G" * n_flank + sequence + "G" * n_flank


def _strip_flanking_glycines(pdb_string: str, n_flank_left: int, n_flank_right: int,
                              frag_len: int) -> str:
    """Remove flanking glycine residues from PDB string and renumber.

    ESMFold numbers residues 1..total where total = n_flank_left + frag_len + n_flank_right.
    Fragment residues sit at positions (n_flank_left+1)..(n_flank_left+frag_len).
    After stripping, residues are renumbered 1..frag_len.
    """
    lo = n_flank_left + 1          # first residue to keep (1-indexed PDB)
    hi = n_flank_left + frag_len   # last  residue to keep

    lines = pdb_string.split("\n")
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                resnum = int(line[22:26].strip())
            except (ValueError, IndexError):
                out_lines.append(line)
                continue
            if lo <= resnum <= hi:
                new_resnum = resnum - n_flank_left   # renumber to 1..frag_len
                line = line[:22] + f"{new_resnum:4d}" + line[26:]
                out_lines.append(line)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def extract_plddt(pdb_path: str) -> Tuple[float, np.ndarray]:
    """
    Extract per-residue pLDDT from B-factor column of a PDB file.

    ESMFold (like AlphaFold) stores per-atom pLDDT in the B-factor column.
    We average over CA atoms to get one value per residue.

    Returns
    -------
    (mean_plddt, per_residue_plddt) — mean_plddt is NaN if PDB unreadable.
    """
    ca_bfactors = []
    prev_resnum = None
    atom_bfactors = []

    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                resnum = line[22:26].strip()
                try:
                    bfactor = float(line[60:66])
                except (ValueError, IndexError):
                    continue

                if atom_name == "CA":
                    if resnum != prev_resnum and atom_bfactors:
                        ca_bfactors.append(np.mean(atom_bfactors))
                        atom_bfactors = []
                    atom_bfactors.append(bfactor)
                    prev_resnum = resnum

        if atom_bfactors:
            ca_bfactors.append(np.mean(atom_bfactors))

        if not ca_bfactors:
            return float("nan"), np.array([])

        arr = np.array(ca_bfactors, dtype=float)
        return float(np.mean(arr)), arr

    except Exception:
        return float("nan"), np.array([])


def write_confidence_tsv(
    out_path: str,
    decoy_paths: list,
    mean_plddts: list,
) -> None:
    """
    Write a TSV with decoy scores (pLDDT confidence).

    Format: <decoy_name> TAB <mean_plddt> TAB <min_plddt>
    """
    with open(out_path, "a") as fh:
        for path, plddt in zip(decoy_paths, mean_plddts):
            name = os.path.basename(path)
            fh.write(f"{name}\t{plddt:.4f}\n")


def predict_structure(sequence: str, seed: int = 42) -> str:
    """
    Predict structure for a single sequence using ESMFold.

    Parameters
    ----------
    sequence : str  Amino-acid sequence
    seed     : int  Random seed for reproducibility

    Returns
    -------
    str  PDB format string
    """
    if not _HAS_ESMFOLD:
        raise ImportError("ESMFold not installed. Run: pip install 'fair-esm[esmfold]'")
    import torch as _torch
    model = get_model()
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)

    with _torch.no_grad():
        pdb_string = model.infer_pdb(sequence)

    return pdb_string


def save_pdb(pdb_string: str, path: str) -> None:
    """Save PDB string to file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(pdb_string)


def predict_multi_seed(sequence: str, n_seeds: int = 10, base_seed: int = 42) -> list:
    """
    Run n_seeds predictions with seeds [base_seed, base_seed+1, ...].

    Returns list of PDB strings.
    """
    pdbs = []
    for i in range(n_seeds):
        seed = base_seed + i
        print(f"    Seed {i+1}/{n_seeds} (seed={seed}) …", flush=True)
        pdb_string = predict_structure(sequence, seed=seed)
        pdbs.append(pdb_string)
    return pdbs


def cluster_and_select_centroid(
    pdb_strings: list, reject_threshold: float = 40.0
) -> Tuple[Optional[str], float, list, list]:
    """
    Cluster multi-seed predictions using hierarchical clustering on RMSD.

    Pre-filters by pLDDT: rejects structures with mean pLDDT < reject_threshold.

    Returns
    -------
    (centroid_pdb_string, inter_seed_rmsd_mean, pdb_filtered, plddt_filtered)
    Returns (None, NaN, [], []) if all seeds rejected.
    """
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        import mdtraj as md
    except ImportError:
        raise ImportError("scipy and mdtraj required for clustering")

    pdb_filtered = []
    plddt_filtered = []

    with tempfile.TemporaryDirectory(prefix="esmfold_cluster_") as tmp_dir:
        # Use temporary files because mdtraj loads PDB paths.
        for i, pdb_str in enumerate(pdb_strings):
            tmp_file = os.path.join(tmp_dir, f"seed_{i:02d}.pdb")
            save_pdb(pdb_str, tmp_file)
            mean_plddt, _ = extract_plddt(tmp_file)
            if not np.isnan(mean_plddt) and mean_plddt >= reject_threshold:
                pdb_filtered.append(pdb_str)
                plddt_filtered.append(mean_plddt)

        if not pdb_filtered:
            print(f"    All seeds rejected (pLDDT < {reject_threshold})", flush=True)
            return None, float("nan"), [], []

        if len(pdb_filtered) == 1:
            print(f"    Only 1 seed passed pLDDT filter; using it as centroid.", flush=True)
            return pdb_filtered[0], 0.0, pdb_filtered, plddt_filtered

        print(f"    Clustering {len(pdb_filtered)} seeds …", flush=True)
        trajectories = []
        for pdb_str in pdb_filtered:
            tmp_pdb = tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", dir=tmp_dir, delete=False
            )
            tmp_pdb.write(pdb_str)
            tmp_pdb.close()
            traj = md.load(tmp_pdb.name)
            trajectories.append(traj)

        n_structs = len(trajectories)
        rmsd_matrix = np.zeros((n_structs, n_structs))

        for i in range(n_structs):
            ca_i = trajectories[i].topology.select("name CA")
            for j in range(i + 1, n_structs):
                ca_j = trajectories[j].topology.select("name CA")
                if len(ca_i) == len(ca_j) and len(ca_i) > 0:
                    rmsd_vals = md.rmsd(
                        trajectories[j], trajectories[i],
                        atom_indices=ca_j, ref_atom_indices=ca_i
                    )
                    rmsd = float(rmsd_vals[0])
                else:
                    rmsd = float("nan")
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        from config import ENSEMBLE_CLUSTER_METHOD, ENSEMBLE_RMSD_CUTOFF
        condensed = np.array([
            rmsd_matrix[i, j]
            for i in range(n_structs)
            for j in range(i + 1, n_structs)
        ])

        Z = linkage(condensed, method=ENSEMBLE_CLUSTER_METHOD)
        clusters = fcluster(Z, ENSEMBLE_RMSD_CUTOFF, criterion="distance")

        unique, counts = np.unique(clusters, return_counts=True)
        majority_cluster = unique[np.argmax(counts)]
        majority_indices = np.where(clusters == majority_cluster)[0]

        print(f"    Majority cluster: {len(majority_indices)}/{len(pdb_filtered)} seeds", flush=True)

        centroid_idx = min(
            majority_indices,
            key=lambda idx: np.mean(rmsd_matrix[idx, majority_indices])
        )
        inter_seed_rmsd = float(np.mean(rmsd_matrix[centroid_idx, majority_indices]))

        print(
            f"    Centroid: structure {centroid_idx} "
            f"(intra-cluster RMSD={inter_seed_rmsd:.3f} nm)",
            flush=True,
        )

        return pdb_filtered[centroid_idx], inter_seed_rmsd, pdb_filtered, plddt_filtered


def fold_fragment(
    fasta_path: str,
    out_dir: str,
    n_seeds: int = 1,
    base_seed: int = 42,
    flank_glycines: int = 15,
    score_output: Optional[str] = None,
) -> list:
    """
    Main entry point: fold one fragment FASTA, write decoy_*.pdb to out_dir.

    ESMFold is deterministic — per-residue pLDDT is the uncertainty proxy, not
    inter-seed variance. `n_seeds > 1` is kept for future input-perturbation
    ensembling but defaults to 1.

    Writes:
      decoy_00000.pdb   — centroid (or single prediction if n_seeds=1)
      seed_{i:02d}.pdb  — individual seeds when n_seeds > 1
      scores.tsv        — pLDDT mean/std/min/max + inter-seed RMSD stats

    Returns [centroid_path] or [] if rejected by pLDDT filter.
    """
    os.makedirs(out_dir, exist_ok=True)

    header, sequence = _read_fasta(fasta_path)
    frag_len = len(sequence)
    query_name = os.path.splitext(os.path.basename(fasta_path))[0]

    print(f"  [{query_name}] Fragment: {frag_len} aa", flush=True)

    flanked_seq = _add_flanking_glycines(sequence, n_flank=flank_glycines)
    print(f"  [{query_name}] Flanked: {len(flanked_seq)} aa (15G + {frag_len} + 15G)", flush=True)

    print(f"  [{query_name}] Running ESMFold (n={n_seeds}) …", flush=True)
    pdb_strings = predict_multi_seed(flanked_seq, n_seeds=n_seeds, base_seed=base_seed)

    from config import PLDDT_REJECT_THRESHOLD
    centroid_pdb, inter_seed_rmsd, pdb_filtered, plddt_filtered = cluster_and_select_centroid(
        pdb_strings, reject_threshold=PLDDT_REJECT_THRESHOLD
    )

    if centroid_pdb is None:
        print(f"  [{query_name}] Rejected by pLDDT filter (<{PLDDT_REJECT_THRESHOLD}). No prediction saved.",
              flush=True)
        return []

    centroid_pdb = _strip_flanking_glycines(centroid_pdb, flank_glycines, flank_glycines, frag_len)
    out_path = os.path.join(out_dir, "decoy_00000.pdb")
    save_pdb(centroid_pdb, out_path)

    if n_seeds > 1:
        for i, pdb_str in enumerate(pdb_filtered):
            stripped = _strip_flanking_glycines(pdb_str, flank_glycines, flank_glycines, frag_len)
            save_pdb(stripped, os.path.join(out_dir, f"seed_{i:02d}.pdb"))

    if score_output is None:
        score_output = os.path.join(out_dir, "scores.tsv")

    # pLDDT is the confidence proxy for the selected structure.
    mean_plddt, per_res_plddt = extract_plddt(out_path)
    if per_res_plddt.size > 0:
        plddt_std = float(np.std(per_res_plddt))
        plddt_min = float(np.min(per_res_plddt))
        plddt_max = float(np.max(per_res_plddt))
        plddt_median = float(np.median(per_res_plddt))
    else:
        plddt_std = plddt_min = plddt_max = plddt_median = float("nan")

    inter_seed_plddt_std = float(np.std(plddt_filtered)) if len(plddt_filtered) > 1 else 0.0
    n_seeds_passed = len(pdb_filtered)

    write_confidence_tsv(score_output, [out_path], [mean_plddt])
    with open(score_output, "a") as fh:
        fh.write(
            f"# plddt_mean={mean_plddt:.4f}\t"
            f"plddt_std={plddt_std:.4f}\t"
            f"plddt_min={plddt_min:.4f}\t"
            f"plddt_max={plddt_max:.4f}\t"
            f"plddt_median={plddt_median:.4f}\t"
            f"inter_seed_rmsd={inter_seed_rmsd:.4f}\t"
            f"inter_seed_plddt_std={inter_seed_plddt_std:.4f}\t"
            f"n_seeds_passed={n_seeds_passed}/{n_seeds}\n"
        )

    print(
        f"  [{query_name}] pLDDT mean={mean_plddt:.1f} std={plddt_std:.1f} "
        f"[{plddt_min:.1f}–{plddt_max:.1f}] | "
        f"passed={n_seeds_passed}/{n_seeds}",
        flush=True,
    )

    return [out_path]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESMFold fragment structure prediction")
    parser.add_argument("fasta", help="Fragment FASTA file")
    parser.add_argument("out_dir", help="Output directory for decoy PDBs")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default: 42)")
    parser.add_argument("--flank-glycines", type=int, default=15, help="Flanking glycines (default: 15)")
    parser.add_argument("--score-output", default=None, help="Path for confidence TSV")
    args = parser.parse_args()

    decoys = fold_fragment(
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        flank_glycines=args.flank_glycines,
        score_output=args.score_output,
    )

    print(f"Done: {len(decoys)} decoy(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
