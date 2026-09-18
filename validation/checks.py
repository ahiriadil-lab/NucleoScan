"""
Validation module for NucleoScan.

Checks fragment metrics and residue scores for pathological patterns that
indicate degenerate input data (e.g. insufficient decoys) or
systematic scoring artefacts.

Usage
-----
    from validate import run_validation
    run_validation(fragment_df, residue_df, results_dir)
"""
import os
import numpy as np
import pandas as pd


# Helpers

_WARN = "WARNING"
_OK   = "OK"


def _check(condition: bool, code: str, message: str) -> dict:
    return {
        "code":    code,
        "status":  _WARN if condition else _OK,
        "message": message,
    }


# 1. Fragment-level metrics

def validate_fragment_metrics(fragment_df: pd.DataFrame) -> list:
    """
    Validate per-fragment aggregate metrics.

    Checks
    ------
    FM-01  Q-values all zero (no native contacts detected)
    FM-02  hydro_rg all NaN (no sidechain atoms — backbone-only output)
    FM-03  norm_q constant (std < 0.01) — metric carries no information
    FM-04  norm_hydro_rg constant (std < 0.01)
    """
    issues = []
    if fragment_df.empty:
        issues.append(_check(True, "FM-00", "fragment_df is empty — no data to validate"))
        return issues

    # FM-01
    q_vals = fragment_df["q_mean"].values
    issues.append(_check(
        np.all(q_vals == 0),
        "FM-01",
        "All Q-values are 0 — no native contacts detected. "
        "Likely cause: invalid ESMFold output or wrong reference structure.",
    ))

    # FM-02
    hydro = fragment_df["hydro_rg_mean"].values
    issues.append(_check(
        np.all(np.isnan(hydro)),
        "FM-02",
        "hydro_rg is NaN for all fragments — no hydrophobic sidechain atoms found. "
        "Check the ESMFold output.",
    ))

    # FM-03
    if "norm_q" in fragment_df.columns:
        nq = fragment_df["norm_q"].values
        issues.append(_check(
            float(np.nanstd(nq)) < 0.01,
            "FM-03",
            f"norm_q is nearly constant (std={np.nanstd(nq):.4f}) — "
            "Q metric contributes nothing to scoring.",
        ))

    # FM-04
    if "norm_hydro_rg" in fragment_df.columns:
        nh = fragment_df["norm_hydro_rg"].values
        issues.append(_check(
            float(np.nanstd(nh)) < 0.01,
            "FM-04",
            f"norm_hydro_rg is nearly constant (std={np.nanstd(nh):.4f}) — "
            "hydro_rg metric contributes nothing to scoring.",
        ))

    # FM-05 — at least one fragment reaches a native-like conformation
    # Standard native-like threshold: RMSD ≤ 0.5 nm (Lindorff-Larsen et al., 2011)
    if "rmsd_mean" in fragment_df.columns:
        has_native_like = np.any(fragment_df["rmsd_mean"].values < 0.5)
        issues.append(_check(
            not has_native_like,
            "FM-05",
            "No fragment has rmsd_mean < 0.5 nm — no native-like conformations sampled. "
            "Decoys may be severely misfolded or the reference structure is incorrect.",
        ))

    return issues


# 2. Residue-level scores

def validate_residue_scores(residue_df: pd.DataFrame) -> list:
    """
    Run software sanity checks on per-residue FES and candidate flags.

    Checks
    ------
    RS-01  Scores monotonically decrease N→C (Spearman |rho| > 0.95) — positional bias
    RS-02  Candidate set is a contiguous N-terminal block — positional artefact
    RS-03  Candidate fraction > 60% — threshold too low / scores degenerate
    RS-04  Candidate fraction = 0% — threshold too high / scores degenerate
    """
    issues = []
    if residue_df.empty:
        issues.append(_check(True, "RS-00", "residue_df is empty — no data to validate"))
        return issues

    scores   = residue_df["nucleus_score"].values
    residues = residue_df["residue"].values
    nucleus  = residue_df[residue_df["is_nucleus"]]["residue"].values

    # RS-01 — Spearman correlation of score vs position
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(residues, scores)
        issues.append(_check(
            abs(rho) > 0.95,
            "RS-01",
            f"FES values are monotone along sequence (Spearman rho={rho:.3f}). "
            "This is a positional bias artefact, not a biological signal. "
            "Consider switching to SCORING_METHOD='marginal'.",
        ))
    except ImportError:
        pass  # scipy not available; skip

    # RS-02 — candidate set is [1..k]
    if len(nucleus) > 0:
        contiguous_from_1 = (nucleus.min() == 1) and (
            np.all(np.diff(sorted(nucleus)) == 1)
        )
        issues.append(_check(
            contiguous_from_1,
            "RS-02",
            f"Candidate set {sorted(nucleus.tolist())[:10]}… is a contiguous N-terminal block. "
            "This is characteristic of sliding-window N-terminal bias.",
        ))
    else:
        issues.append(_check(False, "RS-02", "No selected candidates to check continuity."))

    # RS-03 — fraction too high
    frac = len(nucleus) / max(len(residues), 1)
    issues.append(_check(
        frac > 0.60,
        "RS-03",
        f"Candidate fraction = {frac:.1%} (> 60%) — SCORE_THRESHOLD may be too low "
        "or scores are degenerate.",
    ))

    # RS-04 — fraction = 0
    issues.append(_check(
        len(nucleus) == 0,
        "RS-04",
        "No candidates selected (0%) — SCORE_THRESHOLD may be too high "
        "or fragment scores are all equal.",
    ))

    return issues


# 3. Advanced results

def validate_advanced_results(results_dir: str) -> list:
    """
    Validate advanced analysis outputs (FEL, clusters, contacts).

    Checks
    ------
    ADV-01  FEL minima are all identical (degenerate landscape)
    ADV-02  No native cluster found in cluster_evolution.csv
    ADV-03  contact_formation CSV has no non-empty formation_length entries
    """
    issues = []

    # ADV-01 — FEL: check free_energy_landscape CSV if present
    fel_csv = os.path.join(results_dir, "fel_summary.csv")
    if os.path.exists(fel_csv):
        try:
            fel_df = pd.read_csv(fel_csv)
            if "min_fe" in fel_df.columns:
                mins = fel_df["min_fe"].dropna().values
                issues.append(_check(
                    len(np.unique(mins)) == 1 and len(mins) > 1,
                    "ADV-01",
                    f"FEL minima are all identical ({mins[0]:.3f} kT) — "
                    "landscape is degenerate.",
                ))
        except Exception:
            pass

    # ADV-02 — cluster_evolution.csv
    clust_csv = os.path.join(results_dir, "cluster_evolution.csv")
    if os.path.exists(clust_csv):
        try:
            cev = pd.read_csv(clust_csv)
            has_native = "native" in cev.get("state", pd.Series()).values
            issues.append(_check(
                not has_native,
                "ADV-02",
                "No 'native' cluster detected in cluster_evolution.csv — "
                "all decoys may be unfolded/intermediate.",
            ))
        except Exception:
            pass

    # ADV-03 — contact_formation.csv
    cf_csv = os.path.join(results_dir, "contact_formation.csv")
    if os.path.exists(cf_csv):
        try:
            cf_df = pd.read_csv(cf_csv)
            if "formation_length" in cf_df.columns:
                non_null = cf_df["formation_length"].dropna()
                issues.append(_check(
                    len(non_null) == 0,
                    "ADV-03",
                    "contact_formation.csv has no formation_length entries — "
                    "no contacts formed across fragment lengths.",
                ))
        except Exception:
            pass

    return issues


def run_validation(fragment_df: pd.DataFrame,
                   residue_df: pd.DataFrame,
                   results_dir: str) -> pd.DataFrame:
    """
    Run all validation checks, save report CSV, and print a coloured summary.

    Returns
    -------
    pd.DataFrame  — validation report (code, status, message)
    """
    all_issues = []
    all_issues += validate_fragment_metrics(fragment_df)
    all_issues += validate_residue_scores(residue_df)
    all_issues += validate_advanced_results(results_dir)

    report_df = pd.DataFrame(all_issues)

    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "validation_report.csv")
    report_df.to_csv(report_path, index=False)

    # Print summary
    warnings_found = report_df[report_df["status"] == _WARN]
    ok_count = (report_df["status"] == _OK).sum()

    print("\n=== Validation Report ===")
    if warnings_found.empty:
        print(f"  All {ok_count} checks passed — no issues detected.")
    else:
        print(f"  {len(warnings_found)} WARNING(s) | {ok_count} OK")
        for _, row in warnings_found.iterrows():
            print(f"  [{row['code']}] {row['message']}")
    print(f"  Full report → {report_path}")

    return report_df
