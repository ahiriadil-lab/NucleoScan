"""
Legacy exploratory comparison of thresholded NucleoScan candidates with
bundled literature-derived high-Φ residue lists.

Start2Fold does not contain Φ-values and is not the source of these lists. The
lists below are approximate, have not been curated as a mutation-level
benchmark, and must not be presented as folding-nucleus ground truth. This
script is retained for compatibility and software exploration; publication use
requires audited citations, constructs, residue mappings and Φ-value criteria.

Usage:
    python evaluate_results.py
    python evaluate_results.py --summary results/dataset_summary.csv
"""

import argparse
import ast
import logging
import math
import os
import re
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Approximate literature-derived high-Φ examples (nominally Φ > 0.5)
# Key: PDB ID (lowercase), Value: list of high-φ residue numbers (PDB numbering)
# Smoke-test reference only: these lists are not sourced from Start2Fold and
# are not a publication-grade curated benchmark.
EXP_PHI = {
    "1csp": [5, 6, 7, 10],           # CspB: Lys5, Val6, Lys7, Asn10 (φ ≈ 1)
    "1imq": [],                        # Im9: moderate φ 0.3–0.5 only, none > 0.5 clearly
    "2abd": [5, 9, 12, 15, 73, 74, 77, 80],  # ACBP: Phe5,Ala9,Val12,Leu15,Tyr73,Ile74,Val77,Leu80
    "1ris": [6, 8, 26, 30],           # S6: V6, I8, I26, L30
    "1aps": [],                        # AcP: distributed 0.2–0.5, none φ ≈ 1
    "1poh": [],                        # HPr: no φ-values published
    "2ci2": [16, 49, 57],             # CI2: Ala16 (φ≈1), Leu49, Ile57 (PDB numbers)
    "1pgb": [47],                      # Protein G: Asp47 (φ ≈ 1)
    "1hz6": [1, 2, 3, 4, 5, 6, 7, 8], # Protein L: β1-turn1-β2 (φ 0.5–1.0) ~residues 1–8
    "1ubq": [1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18],  # Ubiquitin: β1-β2 + helix
    "1ten": [9, 14, 17, 22, 31],      # TNfn3: B-C-E-F ring hydrophobe (approx)
    "2hbb": [],                        # NTL9: max φ = 0.40 < 0.5, no high-φ residues
    "1bdd": [27, 32, 34],             # BdpA/Z-domain: Ala27, Ile32, Leu34 (H2)
    "1prb": [],                        # Protein B: "H2 (φ → 1)" but no residue numbers
    "1lmb": [14, 33],                  # λ-Repressor: Asp14, Gln33
    "1o6x": [],                        # ADA2h: 1 contact β1 but no specific residue numbers
    "1urn": [],                        # U1A: fractional 0.2–0.5, none > 0.5
    "2f21": [],                        # FBP28 WW: limited φ data
    "2a3d": [],                        # α3D: not yet characterised by φ-values
    "1l2y": [6],                       # Trp-cage: Trp6 (critical contact)
    "1w4h": [],                        # Peripheral SBD: no φ-value study available
}

# Proteins where we have enough data for meaningful evaluation
EVALUABLE = {pdb for pdb, refs in EXP_PHI.items() if refs}

TOLERANCE = 2  # ±N residues for matching (neighbouring residue tolerance)


# Helper functions

def parse_nucleus(nucleus_str: str) -> list:
    """Parse the legacy candidate-list field from dataset_summary.csv."""
    if not isinstance(nucleus_str, str) or nucleus_str.strip() in ("", "[]"):
        return []
    try:
        return ast.literal_eval(nucleus_str)
    except Exception:
        nums = re.findall(r"\d+", nucleus_str)
        return [int(x) for x in nums]


def match_with_tolerance(pred_list: list, ref_list: list, tol: int = TOLERANCE):
    """
    Return (tp, fp, fn) with ±tol residue matching.

    Each reference residue is matched by at most one predicted residue
    (greedy, closest-first assignment).
    """
    if not ref_list:
        return 0, len(pred_list), 0
    if not pred_list:
        return 0, 0, len(ref_list)

    pred_arr = np.array(sorted(pred_list))
    ref_arr = np.array(sorted(ref_list))

    matched_pred = set()
    matched_ref = set()

    for ri, r in enumerate(ref_arr):
        candidates = np.where(np.abs(pred_arr - r) <= tol)[0]
        if len(candidates) == 0:
            continue
        for ci in candidates[np.argsort(np.abs(pred_arr[candidates] - r))]:
            if ci not in matched_pred:
                matched_pred.add(ci)
                matched_ref.add(ri)
                break

    tp = len(matched_ref)
    fp = len(pred_list) - len(matched_pred)
    fn = len(ref_list) - tp
    return tp, fp, fn


def compute_f1(tp: int, fp: int, fn: int) -> tuple:
    """Return (precision, recall, F1) from TP/FP/FN counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def parse_beta_T(val) -> float:
    """Parse βT field which may be a range string like '0.6–0.7'."""
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace("≥", "").replace("~", "").replace("≈", "")
    m = re.findall(r"[\d.]+", s)
    if len(m) >= 2:
        return (float(m[0]) + float(m[1])) / 2
    if len(m) == 1:
        return float(m[0])
    return float("nan")


def mech_group(m: str) -> str:
    """Map raw mechanism string to a canonical folding mechanism category."""
    if not isinstance(m, str):
        return "Unknown"
    m_low = m.lower()
    if "nucleation" in m_low or "nc" in m_low:
        if "polariz" in m_low:
            return "NC (polarized)"
        if "diffuse" in m_low:
            return "NC (diffuse)"
        if "moving" in m_low:
            return "NC (moving TS)"
        if "textbook" in m_low or "framework" in m_low:
            return "NC (framework)"
        return "NC"
    if "ultrafast" in m_low or "downhill" in m_low:
        return "Ultrafast"
    if "diffusion" in m_low:
        return "Diffusion-collision"
    return "Other"


# Main evaluation

def evaluate(summary_path: str, kinetics_path: str) -> None:
    """Run the legacy exploratory comparison and print a report."""
    log.info("Loading %s", summary_path)
    summary = pd.read_csv(summary_path)
    log.info("Loading %s", kinetics_path)
    xlsx = pd.read_excel(kinetics_path)

    summary["pdb_lower"] = summary["pdb_id"].str.lower().str.strip()
    xlsx["pdb_lower"] = xlsx["PDB"].str.lower().str.strip()

    xlsx["kf_val"] = pd.to_numeric(xlsx["kf (s⁻¹)"], errors="coerce")
    xlsx["ln_kf"] = np.log(xlsx["kf_val"].clip(lower=1e-12))
    xlsx["beta_T"] = xlsx["βT"].apply(parse_beta_T)
    xlsx["n_res"] = pd.to_numeric(xlsx["Domain\nlength (aa)"], errors="coerce")

    merged = summary.merge(xlsx[["pdb_lower", "Protein", "Mechanism",
                                  "kf_val", "ln_kf", "beta_T", "n_res"]],
                           on="pdb_lower", how="left")

    print("=" * 70)
    print("NucleoScan — EXPLORATORY HIGH-Φ COMPARISON")
    print("=" * 70)
    print(f"Proteins in summary: {len(summary)}")
    print("WARNING: bundled high-Φ lists are approximate and unaudited.")
    print(f"Proteins with a bundled high-Φ list: {len(EVALUABLE)}")
    print(f"Tolerance: ±{TOLERANCE} residues")
    print()

    # Per-protein evaluation
    records = []
    for _, row in merged.iterrows():
        pdb = row["pdb_lower"]
        pred = parse_nucleus(row["nucleus_residues"])
        ref  = EXP_PHI.get(pdb, [])
        has_ref = len(ref) > 0

        tp, fp, fn = match_with_tolerance(pred, ref)
        prec, rec, f1 = compute_f1(tp, fp, fn)

        records.append({
            "name":      row["name"],
            "pdb":       pdb,
            "n_res":     row.get("n_res", row["n_residues"]),
            "pred":      pred,
            "ref":       ref,
            "has_ref":   has_ref,
            "n_pred":    len(pred),
            "n_ref":     len(ref),
            "tp":        tp, "fp": fp, "fn": fn,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "mechanism": row.get("Mechanism", ""),
            "beta_T":    row.get("beta_T", float("nan")),
            "ln_kf":     row.get("ln_kf", float("nan")),
            "kf_val":    row.get("kf_val", float("nan")),
        })

    results = pd.DataFrame(records)
    has_ref_df = results[results["has_ref"]]
    no_ref_df  = results[~results["has_ref"]]

    # Section 1 — Overall metrics
    total_tp = has_ref_df["tp"].sum()
    total_fp = has_ref_df["fp"].sum()
    total_fn = has_ref_df["fn"].sum()
    micro_prec, micro_rec, micro_f1 = compute_f1(total_tp, total_fp, total_fn)
    macro_f1 = has_ref_df["f1"].mean()
    macro_f1_std = has_ref_df["f1"].std()

    print("─" * 70)
    print(f"{'OVERALL METRICS':^70}")
    print("─" * 70)
    print(f"  Proteins evaluated (has φ-ref):  {len(has_ref_df)}")
    print(f"  Micro-F1:   {micro_f1:.3f}  (P={micro_prec:.3f}  R={micro_rec:.3f})")
    print(f"  Micro TP/FP/FN: {int(total_tp)} / {int(total_fp)} / {int(total_fn)}")
    print(f"  Macro-F1:   {macro_f1:.3f} ± {macro_f1_std:.3f}")
    print()

    # Section 2 — Per-protein table
    print("─" * 70)
    print(f"{'PER-PROTEIN RESULTS':^70}")
    print("─" * 70)
    print(f"{'Protein':<25} {'PDB':>5} {'Pred':>6} {'Ref':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'F1':>6}")
    print("─" * 70)

    for _, r in has_ref_df.sort_values("f1", ascending=False).iterrows():
        print(f"{r['name'][:25]:<25} {r['pdb']:>5} {r['n_pred']:>6} {r['n_ref']:>5} "
              f"{int(r['tp']):>3} {int(r['fp']):>3} {int(r['fn']):>3} {r['f1']:>6.3f}")

    print("─" * 70)
    print("  Descriptive overlap only; these values are not validation metrics.")
    print()

    # Section 3 — Mechanism breakdown
    print("─" * 70)
    print(f"{'BY FOLDING MECHANISM':^70}")
    print("─" * 70)

    results["mech_group"] = results["mechanism"].apply(mech_group)

    for group, grp in results.groupby("mech_group"):
        has_r = grp[grp["has_ref"]]
        if has_r.empty:
            print(f"  {group:<30} (n={len(grp)}, no φ-ref)")
            continue
        print(f"  {group:<30} n={len(has_r)}  Macro-F1={has_r['f1'].mean():.3f} ± {has_r['f1'].std():.3f}")
    print()

    # Section 4 — Correlation with βT
    from scipy.stats import pearsonr, spearmanr

    has_both = has_ref_df.dropna(subset=["beta_T"])
    has_both = has_both[has_both["beta_T"].apply(lambda x: not math.isnan(x))]

    print("─" * 70)
    print(f"{'CORRELATION WITH βT':^70}")
    print("─" * 70)
    if len(has_both) >= 3:
        r_p, p_p = pearsonr(has_both["beta_T"], has_both["f1"])
        r_s, p_s = spearmanr(has_both["beta_T"], has_both["f1"])
        print(f"  n={len(has_both)} proteins with βT and φ-ref")
        print(f"  Pearson  r={r_p:+.3f}  p={p_p:.4f}")
        print(f"  Spearman r={r_s:+.3f}  p={p_s:.4f}")
        print()
        print(f"  {'Protein':<25} {'βT':>6}  {'F1':>6}")
        for _, r in has_both.sort_values("beta_T").iterrows():
            print(f"  {r['name'][:25]:<25} {r['beta_T']:>6.3f}  {r['f1']:>6.3f}")
    else:
        print(f"  Not enough data ({len(has_both)} proteins)")
    print()

    # Section 5 — Correlation with log(kf)
    has_kf = has_ref_df.dropna(subset=["ln_kf"])
    has_kf = has_kf[has_kf["ln_kf"].apply(lambda x: not math.isnan(x))]

    print("─" * 70)
    print(f"{'CORRELATION WITH ln(kf)':^70}")
    print("─" * 70)
    if len(has_kf) >= 3:
        r_p, p_p = pearsonr(has_kf["ln_kf"], has_kf["f1"])
        r_s, p_s = spearmanr(has_kf["ln_kf"], has_kf["f1"])
        print(f"  n={len(has_kf)} proteins with kf and φ-ref")
        print(f"  Pearson  r={r_p:+.3f}  p={p_p:.4f}")
        print(f"  Spearman r={r_s:+.3f}  p={p_s:.4f}")
        print()
        print(f"  {'Protein':<25} {'kf (s⁻¹)':>10}  {'ln(kf)':>7}  {'F1':>6}")
        for _, r in has_kf.sort_values("ln_kf").iterrows():
            print(f"  {r['name'][:25]:<25} {r['kf_val']:>10.0f}  {r['ln_kf']:>7.2f}  {r['f1']:>6.3f}")
    else:
        print(f"  Not enough data ({len(has_kf)} proteins)")
    print()

    # Section 6 — Correlation with protein size
    has_size = has_ref_df.dropna(subset=["n_res"])

    print("─" * 70)
    print(f"{'CORRELATION WITH PROTEIN SIZE':^70}")
    print("─" * 70)
    if len(has_size) >= 3:
        r_p, p_p = pearsonr(has_size["n_res"], has_size["f1"])
        r_s, p_s = spearmanr(has_size["n_res"], has_size["f1"])
        print(f"  n={len(has_size)} proteins")
        print(f"  Pearson  r={r_p:+.3f}  p={p_p:.4f}")
        print(f"  Spearman r={r_s:+.3f}  p={p_s:.4f}")
    else:
        print(f"  Not enough data ({len(has_size)} proteins)")
    print()

    # Section 7 — Proteins without a bundled comparison list
    print("─" * 70)
    print(f"{'PROTEINS WITHOUT A BUNDLED HIGH-Φ LIST':^70}")
    print("─" * 70)
    for _, r in no_ref_df.iterrows():
        print(f"  {r['name'][:30]:<30} {r['pdb']:>5}  "
              f"candidate_count={r['n_pred']:>3}  not evaluated")
    print()

    # Section 8 — Detailed per-protein analysis (evaluable only)
    print("─" * 70)
    print(f"{'DETAILED ANALYSIS (proteins with φ-ref)':^70}")
    print("─" * 70)
    for _, r in has_ref_df.sort_values("f1", ascending=False).iterrows():
        print(f"\n  {r['name']} ({r['pdb'].upper()})")
        print(f"    Thresholded candidates: {sorted(r['pred'])}")
        print(f"    Bundled high-Φ list:     {sorted(r['ref'])}")
        print(f"    TP={int(r['tp'])}  FP={int(r['fp'])}  FN={int(r['fn'])}  "
              f"P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.3f}")

        pred_s = sorted(r["pred"])
        ref_s  = sorted(r["ref"])
        matched_ref = []
        for ref_r in ref_s:
            close = [p for p in pred_s if abs(p - ref_r) <= TOLERANCE]
            if close:
                best = min(close, key=lambda p: abs(p - ref_r))
                matched_ref.append(f"{ref_r}→{best}")
        missed_ref = [str(rr) for rr in ref_s
                      if not any(abs(pp - rr) <= TOLERANCE for pp in pred_s)]
        if matched_ref:
            print(f"    Matched (ref→pred): {', '.join(matched_ref)}")
        if missed_ref:
            print(f"    Missed ref residues: {', '.join(missed_ref)}")
    print()

    # Section 9 — Summary and baseline comparison
    print("─" * 70)
    print(f"{'SUMMARY':^70}")
    print("─" * 70)

    baseline_records = []
    for _, row in results.iterrows():
        if not row["has_ref"]:
            continue
        n = int(row["n_residues"]) if "n_residues" in row.index else int(row["n_res"])
        pred_all = list(range(1, n + 1))
        ref = row["ref"]
        tp, fp, fn = match_with_tolerance(pred_all, ref)
        _, _, f1 = compute_f1(tp, fp, fn)
        baseline_records.append(f1)

    baseline_macro = np.mean(baseline_records) if baseline_records else float("nan")

    print(f"  Pipeline Macro-F1:  {macro_f1:.3f} ± {macro_f1_std:.3f}")
    print(f"  Baseline (all-res): {baseline_macro:.3f}  (select every residue)")
    print()
    print(f"  Best performers:   {', '.join(has_ref_df.nlargest(3, 'f1')['name'].tolist())}")
    print(f"  Worst performers:  {', '.join(has_ref_df.nsmallest(3, 'f1')['name'].tolist())}")
    print()

    comparison = []
    for _, row in has_ref_df.iterrows():
        n = int(row.get("n_residues", row["n_res"]))
        pred_all = list(range(1, n + 1))
        ref = row["ref"]
        tp_b, fp_b, fn_b = match_with_tolerance(pred_all, ref)
        _, _, f1_b = compute_f1(tp_b, fp_b, fn_b)
        comparison.append({
            "name": row["name"],
            "pipeline_f1": row["f1"],
            "baseline_f1": f1_b,
            "delta": row["f1"] - f1_b,
        })

    cmp_df = pd.DataFrame(comparison)
    beats = cmp_df[cmp_df["delta"] > 0]
    print(f"  Pipeline beats baseline: {len(beats)}/{len(cmp_df)} proteins")
    for _, r in cmp_df.sort_values("delta", ascending=False).iterrows():
        sign = "+" if r["delta"] >= 0 else ""
        print(f"    {r['name'][:30]:<30} pipeline={r['pipeline_f1']:.3f}  "
              f"baseline={r['baseline_f1']:.3f}  Δ={sign}{r['delta']:.3f}")

    print()
    print("=" * 70)
    print("Exploratory comparison complete.")
    print("=" * 70)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        stream=sys.stdout,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=("NucleoScan — exploratory comparison with approximate "
                     "bundled high-Φ residue lists.")
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(base_dir, "results", "dataset_summary.csv"),
        metavar="PATH",
        help="Path to dataset_summary.csv (default: results/dataset_summary.csv)",
    )
    parser.add_argument(
        "--kinetics",
        default=os.path.join(base_dir, "data", "two_state_folding_v3.xlsx"),
        metavar="PATH",
        help="Path to kinetics Excel database (default: data/two_state_folding_v3.xlsx)",
    )
    args = parser.parse_args()

    evaluate(args.summary, args.kinetics)


if __name__ == "__main__":
    main()
