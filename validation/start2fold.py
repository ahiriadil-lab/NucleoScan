"""
start2fold.py — Evaluate residue scores against Start2Fold HDX protection.

The Start2Fold database (Pancsa & Tompa 2016) provides per-residue HDX
protection levels along two independent axes:

  Folding at residue resolution (Type = "folding"):
    EARLY        — protected at the earliest reported refolding stage
    INTERMEDIATE — protected at intermediate times
    LATE         — protected at later reported stages

  Stability at residue resolution (Type = "stability"):
    STRONG       — strong equilibrium protection
    MEDIUM       — moderately stable
    WEAK         — marginally stable

All six levels are evaluated as distinct experimental protection classes.
EARLY is the primary kinetic-protection benchmark and STRONG is the strongest
equilibrium-protection class. Neither is a ground truth for folding-nucleus or
transition-state membership; HDX protection and kinetic barrier stabilization
are different observables.

Residue numbering: fragment-local 1-indexed (Sequence[p-1] = residue p),
which coincides with PDB resSeq for all Start2Fold entries where
"Sequence is PDB" is True (entire database).

Usage:
  python validation/start2fold.py --all --results-dir results/
  python validation/start2fold.py --all-levels --results-dir results/
  python validation/start2fold.py --protein 2abd --level early --results-dir results/
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_ROOT = _HERE.parent

sys.path.insert(0, str(_ROOT))

from validation.advanced_metrics import (
    aggregate_dataset_metrics,
    bootstrap_metric_ci,
    compute_mcc,
    compute_precision_at_k,
    compute_roc_auc,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def load_start2fold(path: str) -> dict:
    """
    Parse Start2Fold JSON and return per-PDB residue sets.

    Returns:
        {
          pdb_id (str, lowercase): {
            "early": set[int],      # union of all EARLY folding experiments
            "intermediate": set[int],
            "late": set[int],
            "strong": set[int],     # stability STRONG
            "medium": set[int],
            "weak": set[int],
            "early_consensus": set[int],  # residues in >=50% of EARLY experiments
            "n_early_experiments": int,
            "sequence": str,        # from first entry of this PDB
            "fragment": str,        # "start-end" in UniProt coords
          }
        }

    Raises warnings (not errors) for residue checksum mismatches — those are
    data-quality issues that should not silently pass.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    # Group by (pdb_id, protein identity)
    from collections import defaultdict
    raw: dict[str, dict] = {}  # pdb_id -> accumulated data

    for rec in records:
        pdb = str(rec.get("PDB code", "")).strip().lower()
        if not pdb:
            continue

        ptype = str(rec.get("Type", "")).strip().lower()
        level = str(rec.get("Protection Level", "")).strip().upper()
        sequence = str(rec.get("Sequence", "")).strip()
        fragment = str(rec.get("Fragment (UniProt residues)", "")).strip()
        residues_raw = str(rec.get("Residues", "")).strip()
        exp_id = rec.get("Experiment ID", "")

        if pdb not in raw:
            raw[pdb] = {
                "sequence": sequence,
                "fragment": fragment,
                "folding_exps": {},   # level -> {exp_id -> set[int]}
                "stability_exps": {}, # level -> {exp_id -> set[int]}
            }

        # Parse residue positions
        pos_set = _parse_residues(residues_raw, sequence, pdb)

        bucket = raw[pdb]["folding_exps"] if ptype == "folding" else raw[pdb]["stability_exps"]
        bucket.setdefault(level, {}).setdefault(str(exp_id), set()).update(pos_set)

    # Build output
    result = {}
    for pdb, data in raw.items():
        fe = data["folding_exps"]
        se = data["stability_exps"]

        def _union(level_dict, level_key):
            sets = list(level_dict.get(level_key, {}).values())
            return set().union(*sets) if sets else set()

        def _consensus(level_dict, level_key, threshold=0.5):
            exps = level_dict.get(level_key, {})
            if not exps:
                return set()
            n = len(exps)
            from collections import Counter
            cnt = Counter()
            for s in exps.values():
                cnt.update(s)
            return {r for r, c in cnt.items() if c / n >= threshold}

        n_early = len(fe.get("EARLY", {}))

        result[pdb] = {
            "early": _union(fe, "EARLY"),
            "intermediate": _union(fe, "INTERMEDIATE"),
            "late": _union(fe, "LATE"),
            "strong": _union(se, "STRONG"),
            "medium": _union(se, "MEDIUM"),
            "weak": _union(se, "WEAK"),
            "early_consensus": _consensus(fe, "EARLY"),
            "n_early_experiments": n_early,
            "sequence": data["sequence"],
            "fragment": data["fragment"],
        }

    return result


def _parse_residues(residues_raw: str, sequence: str, pdb: str) -> set:
    """
    Parse "A8;A9;E10" → {8, 9, 10}.
    Validates AA checksum: Sequence[pos-1] must equal the letter prefix.
    Warns (does not raise) on mismatch.
    """
    positions = set()
    if not residues_raw or residues_raw.lower() in ("nan", "none", ""):
        return positions

    for token in residues_raw.split(";"):
        token = token.strip()
        if not token:
            continue
        # Strip leading amino-acid letter (optional single uppercase letter)
        if token and token[0].isalpha() and token[0].isupper():
            aa_letter = token[0]
            pos_str = token[1:]
        else:
            aa_letter = None
            pos_str = token

        try:
            pos = int(pos_str)
        except ValueError:
            continue

        if pos <= 0:
            continue

        # Checksum: verify AA letter matches sequence (0-indexed)
        if aa_letter and sequence and 1 <= pos <= len(sequence):
            expected = sequence[pos - 1].upper()
            if expected != aa_letter.upper():
                warnings.warn(
                    f"[start2fold] {pdb}: residue {token} — expected {expected} at pos {pos}, got {aa_letter}",
                    stacklevel=3,
                )

        positions.add(pos)

    return positions


# ---------------------------------------------------------------------------
# Evaluation per protein
# ---------------------------------------------------------------------------

# Folding axis (kinetic) — ordered early→late
FOLDING_LEVELS   = ["early", "intermediate", "late"]
# Stability axis (thermodynamic) — ordered strong→weak
STABILITY_LEVELS = ["strong", "medium", "weak"]
ALL_LEVELS       = FOLDING_LEVELS + STABILITY_LEVELS


def evaluate_protein(
    pdb_id: str,
    protein_dir: Path,
    s2f_data: dict,
    level: str = "early",
) -> dict | None:
    """
    Evaluate one protein's continuous legacy ``nucleus_score`` (interpreted as
    fragment-emergence score) against one Start2Fold HDX protection label.

    Returns dict with all metrics, or None if data is unavailable.
    """
    if pdb_id not in s2f_data:
        return None

    protected_residues = s2f_data[pdb_id].get(level, set())
    if not protected_residues:
        return None

    scores_path = protein_dir / "residue_scores.csv"
    if not scores_path.exists():
        return None

    df = pd.read_csv(scores_path)
    if "residue" not in df.columns or "nucleus_score" not in df.columns:
        return None

    df = df.sort_values("residue").reset_index(drop=True)
    n = len(df)
    scores = df["nucleus_score"].fillna(0.0).to_numpy(dtype=float)
    score_range = scores.max() - scores.min()
    scores_norm = (scores - scores.min()) / score_range if score_range > 0 else scores.copy()

    residues = df["residue"].to_numpy(dtype=int)
    is_nucleus = df["is_nucleus"].astype(bool).to_numpy() if "is_nucleus" in df.columns else None

    labels = np.array([1 if r in protected_residues else 0 for r in residues])
    if labels.sum() == 0:
        return None

    pred_set = set(residues[is_nucleus]) if is_nucleus is not None else set()

    tp = len(pred_set & protected_residues)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(protected_residues)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "pdb_id": pdb_id,
        "level": level,
        "n_residues": n,
        "n_gt": len(protected_residues),  # legacy output key: number of protected residues
        "n_protected": len(protected_residues),
        "n_predicted_nucleus": len(pred_set),
        "n_candidates": len(pred_set),
        "n_early_experiments": s2f_data[pdb_id]["n_early_experiments"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": compute_mcc(pred_set, protected_residues, n),
        "auc_roc": compute_roc_auc(scores_norm, labels),
        "precision_at_k": compute_precision_at_k(
            scores_norm, protected_residues, len(protected_residues)
        ),
    }


def evaluate_protein_all_levels(
    pdb_id: str,
    protein_dir: Path,
    s2f_data: dict,
) -> dict:
    """
    Evaluate one protein against all 6 Start2Fold HDX levels.

    Returns a flat dict with metrics for every level, suitable for
    inclusion in dataset_summary.csv:

      s2f_early_f1, s2f_early_precision, s2f_early_recall, s2f_early_auc,
      s2f_early_mcc, s2f_early_n_gt,
      s2f_intermediate_f1, … (same for intermediate/late/strong/medium/weak)

    Levels with no ground-truth data are filled with NaN.
    """
    result = {}
    for level in ALL_LEVELS:
        m = evaluate_protein(pdb_id, protein_dir, s2f_data, level=level)
        prefix = f"s2f_{level}"
        if m is None:
            result[f"{prefix}_f1"]        = float("nan")
            result[f"{prefix}_precision"] = float("nan")
            result[f"{prefix}_recall"]    = float("nan")
            result[f"{prefix}_auc"]       = float("nan")
            result[f"{prefix}_mcc"]       = float("nan")
            result[f"{prefix}_n_gt"]      = 0
        else:
            result[f"{prefix}_f1"]        = m["f1"]
            result[f"{prefix}_precision"] = m["precision"]
            result[f"{prefix}_recall"]    = m["recall"]
            result[f"{prefix}_auc"]       = m["auc_roc"]
            result[f"{prefix}_mcc"]       = m["mcc"]
            result[f"{prefix}_n_gt"]      = m["n_gt"]

    # Save per-protein CSV with all-levels detail
    rows = []
    for level in ALL_LEVELS:
        prefix = f"s2f_{level}"
        rows.append({
            "level":     level,
            "axis":      "folding"   if level in FOLDING_LEVELS else "stability",
            "f1":        result[f"{prefix}_f1"],
            "precision": result[f"{prefix}_precision"],
            "recall":    result[f"{prefix}_recall"],
            "auc_roc":   result[f"{prefix}_auc"],
            "mcc":       result[f"{prefix}_mcc"],
            "n_gt":      result[f"{prefix}_n_gt"],
        })
    detail_path = protein_dir / "start2fold_validation.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    return result


# ---------------------------------------------------------------------------
# Dataset-level runners
# ---------------------------------------------------------------------------

def _load_summary(results_path: Path) -> pd.DataFrame:
    summary_csv = results_path / "dataset_summary.csv"
    if summary_csv.exists():
        return pd.read_csv(summary_csv)
    return _infer_summary(results_path)


def _print_and_save_aggregate(all_metrics: list, level: str, out_path: Path):
    scalar_keys = ["precision", "recall", "f1", "mcc", "auc_roc", "precision_at_k"]
    scalar_metrics = [{k: m[k] for k in scalar_keys if k in m} for m in all_metrics]
    agg = aggregate_dataset_metrics(scalar_metrics)

    print(f"\n{'='*55}")
    print(f"Dataset summary — Start2Fold {level.upper()} (n={len(all_metrics)} proteins)")
    print(f"{'='*55}")
    for key in scalar_keys:
        a = agg.get(key, {})
        print(f"  {key:18s}  median={a.get('median', float('nan')):.3f}  "
              f"IQR=[{a.get('iqr_lo', float('nan')):.3f}, {a.get('iqr_hi', float('nan')):.3f}]  "
              f"CI95=[{a.get('ci_lo', float('nan')):.3f}, {a.get('ci_hi', float('nan')):.3f}]")

    agg_rows = [{"metric": k, **v} for k, v in agg.items()]
    agg_path = out_path / f"start2fold_{level}_aggregate.csv"
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print(f"Aggregate results → {agg_path}")
    return agg


def run_all(results_dir: str, output_dir: str, s2f_path: str, level: str = "early"):
    """Evaluate all proteins against one Start2Fold level."""
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Start2Fold from {s2f_path} …")
    s2f = load_start2fold(s2f_path)
    print(f"  {len(s2f)} proteins with HDX data")

    summary = _load_summary(results_path)
    all_metrics, skipped = [], []

    for _, row in summary.iterrows():
        name = str(row.get("name", row.get("Name", ""))).strip()
        pdb_id = str(row.get("pdb_id", row.get("PDB", ""))).strip().lower()
        protein_dir = results_path / name

        if not protein_dir.exists():
            skipped.append((name, "no directory"))
            continue

        metrics = evaluate_protein(pdb_id, protein_dir, s2f, level=level)
        if metrics is None:
            skipped.append((name, f"no {level.upper()} data or missing residue_scores.csv"))
            continue

        metrics["name"] = name
        all_metrics.append(metrics)
        print(f"  {name:35s} ({pdb_id})  F1={metrics['f1']:.3f}  "
              f"P@K={metrics['precision_at_k']:.3f}  AUC={metrics['auc_roc']:.3f}  "
              f"MCC={metrics['mcc']:.3f}  n_gt={metrics['n_gt']}")

    if skipped:
        print(f"\nSkipped {len(skipped)}: {', '.join(n for n, _ in skipped[:5])}"
              + (f" … (+{len(skipped)-5} more)" if len(skipped) > 5 else ""))

    if not all_metrics:
        print("No proteins evaluated.")
        return

    per_protein = pd.DataFrame(all_metrics)
    pp_path = out_path / f"start2fold_{level}_per_protein.csv"
    per_protein.to_csv(pp_path, index=False)
    print(f"\nPer-protein results → {pp_path}")

    agg = _print_and_save_aggregate(all_metrics, level, out_path)
    return per_protein, agg


def run_all_levels(results_dir: str, output_dir: str, s2f_path: str):
    """
    Evaluate all proteins against ALL 6 HDX levels.
    Outputs:
      - start2fold_all_levels_long.csv   (one row per protein × level)
      - start2fold_all_levels_wide.csv   (one row per protein, columns per level)
      - start2fold_{level}_aggregate.csv for each level that has ≥1 protein
    """
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Start2Fold from {s2f_path} …")
    s2f = load_start2fold(s2f_path)
    print(f"  {len(s2f)} proteins with HDX data")
    print(f"  Evaluating levels: {ALL_LEVELS}\n")

    summary = _load_summary(results_path)
    long_rows = []  # (name, pdb_id, level, metrics…)

    for _, row in summary.iterrows():
        name = str(row.get("name", row.get("Name", ""))).strip()
        pdb_id = str(row.get("pdb_id", row.get("PDB", ""))).strip().lower()
        protein_dir = results_path / name

        if not protein_dir.exists():
            continue

        levels_found = []
        for lv in ALL_LEVELS:
            m = evaluate_protein(pdb_id, protein_dir, s2f, level=lv)
            if m is not None:
                m["name"] = name
                long_rows.append(m)
                levels_found.append(f"{lv.upper()}({m['n_gt']})")

        if levels_found:
            print(f"  {name:35s} ({pdb_id})  levels: {', '.join(levels_found)}")
        else:
            print(f"  {name:35s} ({pdb_id})  [no HDX data]")

    if not long_rows:
        print("No proteins evaluated.")
        return

    long_df = pd.DataFrame(long_rows)
    long_path = out_path / "start2fold_all_levels_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nLong-format results → {long_path}")

    # Wide format: one row per (name, pdb_id), columns = {level}_{metric}
    scalar_keys = ["precision", "recall", "f1", "mcc", "auc_roc", "precision_at_k", "n_gt"]
    wide_rows = {}
    for r in long_rows:
        key = (r["name"], r["pdb_id"])
        if key not in wide_rows:
            wide_rows[key] = {"name": r["name"], "pdb_id": r["pdb_id"],
                              "n_residues": r["n_residues"],
                              "n_early_experiments": r["n_early_experiments"]}
        lv = r["level"]
        for sk in scalar_keys:
            wide_rows[key][f"{lv}_{sk}"] = r.get(sk)

    wide_df = pd.DataFrame(list(wide_rows.values()))
    wide_path = out_path / "start2fold_all_levels_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"Wide-format results  → {wide_path}")

    # Per-level aggregates
    print()
    for lv in ALL_LEVELS:
        lv_metrics = [r for r in long_rows if r["level"] == lv]
        if lv_metrics:
            _print_and_save_aggregate(lv_metrics, lv, out_path)

    return long_df, wide_df


def _infer_summary(results_path: Path) -> pd.DataFrame:
    """Fallback: infer protein name/pdb_id from FASTA files in subdirectories."""
    rows = []
    for d in sorted(results_path.iterdir()):
        if not d.is_dir():
            continue
        fastas = list(d.glob("*.fasta"))
        pdb_id = fastas[0].stem.lower() if fastas else d.name.lower()
        rows.append({"name": d.name, "pdb_id": pdb_id})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare NucleoScan scores with Start2Fold HDX protection classes"
    )
    parser.add_argument("--results-dir", default="results_esmfold_v2",
                        help="Root results directory")
    parser.add_argument("--s2f-path", default="data/start2fold_data.json",
                        help="Path to start2fold_data.json")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <results-dir>/start2fold_validation)")
    parser.add_argument("--level", default="early",
                        choices=ALL_LEVELS,
                        help="Start2Fold HDX protection class to evaluate (single level)")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Evaluate all proteins in results-dir (single level)")
    parser.add_argument("--all-levels", action="store_true", dest="run_all_levels",
                        help="Evaluate all proteins against ALL 6 HDX levels")
    parser.add_argument("--protein", default=None,
                        help="Evaluate a single protein by PDB ID")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    s2f_path = Path(args.s2f_path)
    if not s2f_path.is_absolute():
        s2f_path = _ROOT / s2f_path
    output_dir = args.output or str(results_dir / "start2fold_validation")

    if not s2f_path.exists():
        sys.exit(f"Start2Fold data not found: {s2f_path}")

    if args.run_all_levels:
        run_all_levels(str(results_dir), output_dir, str(s2f_path))

    elif args.run_all:
        run_all(str(results_dir), output_dir, str(s2f_path), level=args.level)

    elif args.protein:
        s2f = load_start2fold(str(s2f_path))
        pdb_id = args.protein.lower()
        # Find matching directory via dataset_summary
        summary_csv = results_dir / "dataset_summary.csv"
        if summary_csv.exists():
            summary = pd.read_csv(summary_csv)
            row = summary[summary["pdb_id"].str.lower() == pdb_id]
            if row.empty:
                sys.exit(f"PDB {pdb_id} not found in dataset_summary.csv")
            name = row.iloc[0]["name"]
        else:
            name = pdb_id
        protein_dir = results_dir / name
        m = evaluate_protein(pdb_id, protein_dir, s2f, level=args.level)
        if m is None:
            print(f"No {args.level.upper()} data for {pdb_id}")
        else:
            for k, v in m.items():
                print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
