"""Reproduce HDX comparisons for FES and four transparent baselines.

Methods
-------
``fes``
    Fragment-emergence score from the legacy ``nucleus_score`` column.
``plddt``
    Mean per-residue ESMFold confidence across all N-terminal fragments that
    cover the residue. Fragment predictions are read directly from PDB files.
``contact_degree``
    Native C-alpha contact degree using an 8 A cutoff and sequence separation
    of at least three residues.
``random``
    Uniform random scores with a deterministic protein-specific seed.
``all_residue``
    Constant score for every residue. AUC is 0.5; tie-aware expected P@K is
    the protected-residue prevalence.

Start2Fold classes are HDX protection labels, not folding-nucleus truth.
Outputs are written to ``<results-dir>/baseline_validation/``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from run_esmfold import extract_plddt
from validation.start2fold import ALL_LEVELS, load_start2fold


METHODS = ("fes", "plddt", "contact_degree", "random", "all_residue")


def confidence_baseline(structures_dir: Path, n_residues: int) -> np.ndarray:
    """Mean ESMFold pLDDT across N-prefix predictions covering each residue."""
    score_sum = np.zeros(n_residues, dtype=float)
    coverage = np.zeros(n_residues, dtype=int)

    for frag_dir in sorted(structures_dir.glob("frag[0-9]*")):
        pdb_path = frag_dir / "decoy_00000.pdb"
        if not pdb_path.exists():
            candidates = sorted(frag_dir.glob("*.pdb"))
            if not candidates:
                continue
            pdb_path = candidates[0]

        _, per_residue = extract_plddt(str(pdb_path))
        if per_residue is None or len(per_residue) == 0:
            continue
        n = min(n_residues, len(per_residue))
        valid = np.isfinite(per_residue[:n])
        indices = np.flatnonzero(valid)
        score_sum[indices] += per_residue[:n][indices]
        coverage[indices] += 1

    scores = np.full(n_residues, np.nan, dtype=float)
    covered = coverage > 0
    scores[covered] = score_sum[covered] / coverage[covered]
    return scores


def native_contact_degree(pdb_path: Path, n_residues: int) -> np.ndarray:
    """C-alpha contact degree at 8 A with |i-j| >= 3, consecutive indexing."""
    from Bio.PDB import PDBParser, is_aa

    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = [r for r in chain.get_residues() if is_aa(r, standard=True) and "CA" in r]
    coords = np.array([r["CA"].coord for r in residues], dtype=float)
    n = min(n_residues, len(coords))
    degree = np.full(n_residues, np.nan, dtype=float)
    if n == 0:
        return degree

    degree[:n] = 0.0
    for i in range(n):
        for j in range(i + 3, n):
            if np.linalg.norm(coords[i] - coords[j]) <= 8.0:
                degree[i] += 1.0
                degree[j] += 1.0
    return degree


def precision_at_k_tie_aware(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Expected P@K when ties at the selection boundary are broken at random."""
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    if k <= 0 or len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))

    expected_tp = 0.0
    remaining = k
    for value in np.unique(scores)[::-1]:
        group = scores == value
        group_n = int(group.sum())
        group_pos = int(labels[group].sum())
        if group_n <= remaining:
            expected_tp += group_pos
            remaining -= group_n
        else:
            expected_tp += remaining * group_pos / group_n
            remaining = 0
        if remaining == 0:
            break
    return expected_tp / k


def evaluate_scores(scores: np.ndarray, protected: set[int]) -> dict | None:
    """Evaluate continuous scores against one HDX protection class."""
    residues = np.arange(1, len(scores) + 1)
    labels = np.array([int(r in protected) for r in residues], dtype=int)
    valid = np.isfinite(scores)
    if valid.sum() < 2 or labels[valid].sum() == 0 or labels[valid].sum() == valid.sum():
        return None

    valid_scores = scores[valid]
    valid_labels = labels[valid]
    k = int(valid_labels.sum())
    return {
        "n_scored": int(valid.sum()),
        "n_protected": k,
        "auc_roc": float(roc_auc_score(valid_labels, valid_scores)),
        "precision_at_k": precision_at_k_tie_aware(valid_scores, valid_labels, k),
    }


def _stable_random_scores(pdb_id: str, n_residues: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(pdb_id.encode("utf-8")).digest()
    protein_seed = int.from_bytes(digest[:8], "little") ^ int(seed)
    return np.random.default_rng(protein_seed).random(n_residues)


def _find_native_pdb(pdb_dir: Path, pdb_id: str) -> Path | None:
    matches = sorted(pdb_dir.glob(f"{pdb_id.lower()}_*.pdb"))
    if not matches:
        matches = sorted(pdb_dir.glob(f"{pdb_id.upper()}_*.pdb"))
    return matches[0] if matches else None


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Standard Cliff's delta between two finite samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(x[:, None] > y[None, :]) - np.mean(x[:, None] < y[None, :]))


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare FES with each baseline using matched proteins."""
    rows = []
    for level in ALL_LEVELS:
        level_df = metrics[metrics["level"] == level]
        for metric in ("auc_roc", "precision_at_k"):
            pivot = level_df.pivot_table(
                index=["name", "pdb_id"], columns="method", values=metric, aggfunc="first"
            )
            if "fes" not in pivot:
                continue
            for comparator in METHODS[1:]:
                if comparator not in pivot:
                    continue
                paired = pivot[["fes", comparator]].dropna()
                if paired.empty:
                    continue
                x = paired["fes"].to_numpy(dtype=float)
                y = paired[comparator].to_numpy(dtype=float)
                differences = x - y
                if np.allclose(differences, 0.0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(wilcoxon(x, y, alternative="greater").pvalue)
                    except ValueError:
                        p_value = float("nan")
                rows.append({
                    "level": level,
                    "metric": metric,
                    "comparator": comparator,
                    "n_pairs": len(paired),
                    "median_fes": float(np.median(x)),
                    "median_comparator": float(np.median(y)),
                    "wilcoxon_p_greater": p_value,
                    "cliffs_delta": cliffs_delta(x, y),
                })
    return pd.DataFrame(rows)


def run(
    results_dir: Path,
    structures_dir: Path,
    pdb_dir: Path,
    s2f_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all available proteins and write long-format results."""
    summary_path = results_dir / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing dataset summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    s2f = load_start2fold(str(s2f_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, row in summary.iterrows():
        name = str(row["name"])
        pdb_id = str(row["pdb_id"]).lower()
        protein_dir = results_dir / name
        score_path = protein_dir / "residue_scores.csv"
        if pdb_id not in s2f or not score_path.exists():
            continue

        residue_df = pd.read_csv(score_path).sort_values("residue")
        if "nucleus_score" not in residue_df:
            continue
        fes = residue_df["nucleus_score"].to_numpy(dtype=float)
        n_residues = len(fes)

        score_sets = {
            "fes": fes,
            "plddt": confidence_baseline(structures_dir / name, n_residues),
            "random": _stable_random_scores(pdb_id, n_residues, seed),
            "all_residue": np.ones(n_residues, dtype=float),
        }
        native_path = _find_native_pdb(pdb_dir, pdb_id)
        score_sets["contact_degree"] = (
            native_contact_degree(native_path, n_residues)
            if native_path is not None
            else np.full(n_residues, np.nan)
        )

        baseline_df = pd.DataFrame({
            "residue": np.arange(1, n_residues + 1),
            **score_sets,
        })
        baseline_df.to_csv(protein_dir / "baseline_scores.csv", index=False)

        for level in ALL_LEVELS:
            protected = s2f[pdb_id].get(level, set())
            if not protected:
                continue
            for method, scores in score_sets.items():
                values = evaluate_scores(scores, protected)
                if values is not None:
                    rows.append({
                        "name": name,
                        "pdb_id": pdb_id,
                        "level": level,
                        "method": method,
                        **values,
                    })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "baseline_metrics_long.csv", index=False)
    comparisons = paired_comparisons(metrics) if not metrics.empty else pd.DataFrame()
    comparisons.to_csv(output_dir / "baseline_comparisons.csv", index=False)
    return metrics, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce FES and baseline comparisons against Start2Fold HDX labels"
    )
    parser.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    parser.add_argument("--structures-dir", type=Path, default=_ROOT / "structures")
    parser.add_argument("--pdb-dir", type=Path, default=_ROOT / "data" / "pdb_cleaned")
    parser.add_argument("--s2f-path", type=Path, default=_ROOT / "data" / "start2fold_data.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir / "baseline_validation"
    metrics, comparisons = run(
        args.results_dir,
        args.structures_dir,
        args.pdb_dir,
        args.s2f_path,
        output_dir,
        args.seed,
    )
    print(f"Per-protein metrics: {len(metrics)} rows → {output_dir / 'baseline_metrics_long.csv'}")
    print(f"Matched comparisons: {len(comparisons)} rows → {output_dir / 'baseline_comparisons.csv'}")


if __name__ == "__main__":
    main()
