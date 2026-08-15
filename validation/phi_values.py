"""
Exploratory comparison of FoldNucleus scores with residue-level Φ-values.

The legacy ``nucleus_score`` column is interpreted here as the
fragment-emergence score (FES). A correlation with Φ-values is useful as an
independent hypothesis-generating comparison, but it is not by itself a
validation of folding-nucleus membership. Prefix length is not folding time,
and unmeasured residues are retained as NaN rather than treated as negatives.

If an audited ``data/EVO3_Strict_Database.xlsx`` file is supplied, the adapter
can load mutation-level values by PDB identifier. The bundled hardcoded tables
are approximate literature-derived examples for software smoke tests only.
Before publication, verify the experimental construct, mutation identity,
residue-number mapping, ΔΔG eligibility, uncertainty and citation for every
included value.

Outputs:
  results/{protein}/phi_comparison.csv     — per-residue scores + Φ
  results/{protein}/phi_scatter.png        — exploratory score-versus-Φ plots
  results/{protein}/phi_correlation.txt    — Pearson & Spearman statistics

Usage:
  python validation/phi_values.py [--protein Villin_HP36] [--results-dir results]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Approximate Φ-value examples (1-indexed positions; smoke tests only)
# ---------------------------------------------------------------------------

# Villin HP36 — Kubelka et al. (2003), Brewer et al. (2005)
HP36_PHI_VALUES = {
    7:  0.40,   # Phe7  — hydrophobic core
    11: 0.40,   # Phe11 — hydrophobic core
    18: 0.60,   # Phe18 — high-Φ example
    20: 0.30,   # Leu20 — partially structured in TS
    28: 0.15,   # Leu28 — late folding
}

# CI2 (Chymotrypsin Inhibitor 2) — literature-derived smoke-test values
CI2_PHI_VALUES = {
    5:  0.03,   # Ala5  — N-terminus, late folding
    9:  0.10,   # Leu9  — β1, moderate
    12: 0.07,   # Phe12 — loop
    16: 0.47,   # Ala16 — helix
    20: 0.44,   # Ile20 — helix
    24: 0.31,   # Val24 — helix–β interface
    31: 0.44,   # Ile30 — β-strand
    49: 0.60,   # Leu49 — hydrophobic core
    57: 0.55,   # Ile57 — β-strand
    61: 0.30,   # Leu61 — C-terminal β
}

# Barnase — Serrano et al. (1992) J. Mol. Biol.; Fersht (1993)
# 110 aa, all-α/β; most comprehensive Φ dataset (~80 mutants)
BARNASE_PHI_VALUES = {
    5:  0.34,   # Ala5  — helix 1
    6:  0.33,   # Leu6  — helix 1
    8:  0.26,   # Arg8  — helix 1
    14: 0.24,   # Glu14 — helix 1-2 loop
    17: 0.16,   # Arg17 — helix 2
    25: 0.15,   # Ile25 — turn
    32: 0.47,   # Ile32 — β1
    36: 0.40,   # Phe36 — β1-β2
    59: 0.50,   # Ile59 — β2-β3 turn
    70: 0.60,   # Leu89 (renumbered 70 in clean PDB) — β3
    88: 0.34,   # Ile88 — β4
    96: 0.42,   # Tyr96 — β5
}

# ProteinG B1 domain — McCallister et al. (2000) Biochemistry; Nauli et al. (2001)
# 56 aa, α+β; central β-strand examples
PROTEIN_G_PHI_VALUES = {
    3:  0.11,   # Gln3  — β1, late
    20: 0.22,   # Thr20 — loop
    25: 0.30,   # Ala25 — β2
    29: 0.10,   # Lys29 — loop
    39: 0.57,   # Val39 — β3
    43: 0.55,   # Phe43 — β3 hydrophobic
    45: 0.80,   # Trp45 — β3 (highest value in this example table)
    46: 0.72,   # Asp46 — β3-β4 turn
    52: 0.18,   # Leu52 — α helix
}

# Engrailed homeodomain — Mayor et al. (2003) Nature
# 54 aa, all-α; fast two-state folder
ENGRAILED_PHI_VALUES = {
    4:  0.45,   # Ile4  — α1
    8:  0.30,   # Leu8  — α1
    12: 0.50,   # Arg12 — α1-α2
    16: 0.44,   # Val16 — α2
    20: 0.50,   # Met20 — α2 hydrophobic core
    28: 0.10,   # Lys28 — loop
    30: 0.28,   # Ala30 — α3
    34: 0.58,   # Ala34 — α3
    44: 0.35,   # Phe44 — α3
    50: 0.08,   # Arg50 — C-term
}

# Ubiquitin — Went & Jackson (2005); Briggs & Roder (1992)
# 76 aa, α+β; OB-fold-like with central β-sheet
UBIQUITIN_PHI_VALUES = {
    4:  0.57,   # Ile4  — β1
    6:  0.28,   # Leu6  — β1
    8:  0.43,   # Lys8  — β1 end
    11: 0.48,   # Lys11 — β1-β2
    17: 0.26,   # Glu17 — β2
    34: 0.28,   # Ile34 — β2-β3 turn
    44: 0.55,   # Ile44 — β-sheet core
    68: 0.42,   # Leu69 — β5
    73: 0.18,   # Arg73 — C-term
}

# Protein-to-table mapping (legacy smoke-test data, not a curated benchmark)
PHI_TABLES = {
    "Villin_HP36":    HP36_PHI_VALUES,
    "CI2":            CI2_PHI_VALUES,
    "Barnase":        BARNASE_PHI_VALUES,
    "ProteinG_B1":    PROTEIN_G_PHI_VALUES,
    "ProteinG":       PROTEIN_G_PHI_VALUES,   # alias
    "Engrailed":      ENGRAILED_PHI_VALUES,
    "Engrailed_HD":   ENGRAILED_PHI_VALUES,   # alias for dataset directory name
    "Ubiquitin":      UBIQUITIN_PHI_VALUES,
}

# Citation mapping for different proteins and sources
_CITATIONS = {
    "Villin_HP36":  "Kubelka et al. (2003) J Mol Biol 329:625-630; Brewer et al. (2005) PNAS 102:16662-16667",
    "CI2":          "Itzhaki et al. (1995) J Mol Biol 254:289-304",
    "Barnase":      "Serrano et al. (1992) J Mol Biol 224:847-859",
    "ProteinG_B1":  "McCallister et al. (2000) Biochemistry 39:11177-11190",
    "ProteinG":     "McCallister et al. (2000) Biochemistry 39:11177-11190",
    "Engrailed":    "Mayor et al. (2003) Nature 421:863-867",
    "Engrailed_HD": "Mayor et al. (2003) Nature 421:863-867",
    "Ubiquitin":    "Went & Jackson (2005) Protein Eng Des Sel 18:229-239",
    "_default":     "user-supplied EVO3_Strict_Database.xlsx",
}


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def load_predictions(results_dir, protein_name):
    """
    Load FES (legacy ``nucleus_score``), importance and cooperative scores.

    Returns a merged DataFrame with columns:
        residue, nucleus_score, is_nucleus, importance, [coop_score]
    """
    prot_dir = os.path.join(results_dir, protein_name)

    ns_path          = os.path.join(prot_dir, "residue_scores.csv")
    imp_path         = os.path.join(prot_dir, "structural_importance.csv")
    coop_path        = os.path.join(prot_dir, "cooperativity_scores.csv")
    committor_path   = os.path.join(prot_dir, "committor_coop_scores.csv")

    if not os.path.exists(ns_path):
        raise FileNotFoundError(f"residue_scores.csv not found: {ns_path}")

    df_ns = pd.read_csv(ns_path)   # legacy fields: nucleus_score, is_nucleus

    if os.path.exists(imp_path):
        df_imp = pd.read_csv(imp_path)
        imp_cols = [c for c in ["residue", "importance"] if c in df_imp.columns]
        df_ns = df_ns.merge(df_imp[imp_cols], on="residue", how="left")
    else:
        df_ns["importance"] = float("nan")

    # Load cooperative score if available
    if os.path.exists(coop_path):
        df_coop = pd.read_csv(coop_path)
        if "nucleus_score" in df_coop.columns:
            df_coop = df_coop[["residue", "nucleus_score"]].rename(
                columns={"nucleus_score": "coop_score"})
        elif "q_tse" in df_coop.columns:
            # cooperative results with detailed columns
            coop_cols = [c for c in ["residue", "q_tse"] if c in df_coop.columns]
            df_coop = df_coop[coop_cols].rename(columns={"q_tse": "coop_score"})
        if "coop_score" in df_coop.columns:
            df_ns = df_ns.merge(df_coop[["residue", "coop_score"]], on="residue", how="left")

    # Load committor-cooperative score if available
    if os.path.exists(committor_path):
        df_cc = pd.read_csv(committor_path)
        if "nucleus_score" in df_cc.columns:
            df_cc = df_cc[["residue", "nucleus_score"]].rename(
                columns={"nucleus_score": "committor_coop_score"})
        elif "q_tse" in df_cc.columns:
            df_cc = df_cc[["residue", "q_tse"]].rename(
                columns={"q_tse": "committor_coop_score"})
        if "committor_coop_score" in df_cc.columns:
            df_ns = df_ns.merge(df_cc[["residue", "committor_coop_score"]],
                                on="residue", how="left")

    return df_ns


def build_comparison_df(pred_df, phi_table):
    """
    Attach experimental Φ-values to per-residue prediction DataFrame.

    Returns a DataFrame extended with a 'phi_exp' column.
    """
    pred_df = pred_df.copy()
    pred_df["phi_exp"] = pred_df["residue"].map(phi_table)
    return pred_df


def compute_correlations(df, score_col, phi_col="phi_exp"):
    """
    Pearson and Spearman correlation between *score_col* and Φ-values.

    Only residues with non-NaN Φ and non-NaN score are used.
    Returns dict or None if fewer than 3 data points.
    """
    sub = df.dropna(subset=[score_col, phi_col])
    if len(sub) < 3:
        return None
    x = sub[score_col].values
    y = sub[phi_col].values
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "n":          len(sub),
        "pearson_r":  float(pr),
        "pearson_p":  float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phi_scatter(df, out_path, protein_name):
    """
    Exploratory score-versus-Φ panels for residues with measured Φ-values.
    """
    sub = df.dropna(subset=["phi_exp"])
    if sub.empty:
        print("  No Φ-value data available — skipping scatter plot.")
        return

    has_imp              = "importance"           in sub.columns and sub["importance"].notna().any()
    has_coop             = "coop_score"           in sub.columns and sub["coop_score"].notna().any()
    has_committor_coop   = "committor_coop_score" in sub.columns and sub["committor_coop_score"].notna().any()
    ncols = 1 + int(has_imp) + int(has_coop) + int(has_committor_coop)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 1, 4))
    if ncols == 1:
        axes = [axes]

    residue_labels = {int(r): str(int(r)) for r in sub["residue"].values}

    def _panel(ax, xcol, xlabel):
        sub2 = sub.dropna(subset=[xcol])
        if sub2.empty:
            ax.set_visible(False)
            return
        x = sub2[xcol].values
        y = sub2["phi_exp"].values
        res = sub2["residue"].values

        colors = sub2["is_nucleus"].map(
            {True: "#e74c3c", False: "#3498db"}
        ).values if "is_nucleus" in sub2.columns else ["#3498db"] * len(sub2)

        ax.scatter(x, y, c=colors, s=80, zorder=3)
        for xi, yi, ri in zip(x, y, res):
            label = residue_labels.get(ri, str(ri))
            ax.annotate(label, (xi, yi),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color="black")

        if len(x) >= 3:
            m, b, r, p, _ = stats.linregress(x, y)
            xlin = np.linspace(x.min(), x.max(), 100)
            ax.plot(xlin, m * xlin + b, "k--", linewidth=1, alpha=0.6)
            pr, pp2 = stats.pearsonr(x, y)
            ax.text(0.05, 0.92,
                    f"Pearson r = {pr:.2f}  (p={pp2:.2g}, n={len(x)})",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Experimental Φ-value")
        ax.set_title(f"{protein_name}: {xlabel} vs Φ")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="High-score candidate"),
            Patch(color="#3498db", label="Other residue"),
        ], fontsize=7, loc="lower right")

    panel_idx = 0
    _panel(axes[panel_idx], "nucleus_score", "Fragment-emergence score")
    panel_idx += 1
    if has_imp:
        _panel(axes[panel_idx], "importance", "Structural importance")
        panel_idx += 1
    if has_coop:
        _panel(axes[panel_idx], "coop_score", "Cooperative score")
        panel_idx += 1
    if has_committor_coop:
        _panel(axes[panel_idx], "committor_coop_score", "Committor-cooperative score")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Phi scatter plot saved → {out_path}")


def plot_phi_profile(df, out_path, protein_name):
    """
    Bar chart of FES (legacy ``nucleus_score``) with Φ-value overlay.
    """
    fig, ax1 = plt.subplots(figsize=(12, 4))

    x  = df["residue"].values
    ns = df["nucleus_score"].values
    nucleus_mask = df["is_nucleus"].values if "is_nucleus" in df.columns else np.zeros(len(df), dtype=bool)

    colors = ["#e74c3c" if n else "#3498db" for n in nucleus_mask]
    ax1.bar(x, ns, color=colors, alpha=0.75, width=0.8,
            label="Fragment-emergence score")
    ax1.set_xlabel("Residue position")
    ax1.set_ylabel("Fragment-emergence score", color="#3498db")
    ax1.set_title(f"{protein_name}: FES profile with Φ-values overlay")

    # Φ-value overlay (secondary y-axis, scatter)
    sub = df.dropna(subset=["phi_exp"])
    if not sub.empty:
        ax2 = ax1.twinx()
        ax2.scatter(sub["residue"], sub["phi_exp"],
                    color="black", s=60, zorder=5, marker="D",
                    label="Experimental Φ-value")
        for _, row in sub.iterrows():
            ax2.annotate(f"Φ={row['phi_exp']:.2f}",
                         (row["residue"], row["phi_exp"]),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=7, ha="center")
        ax2.set_ylabel("Φ-value", color="black")
        ax2.set_ylim(0, 1.2)
        ax2.legend(loc="upper right", fontsize=8)

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color="#e74c3c", label="High-score candidate"),
        Patch(color="#3498db", label="Other residue"),
    ], fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Phi profile plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(df, corr_nucleus, corr_imp, protein_name, out_path,
                 corr_coop=None, corr_committor_coop=None):
    lines = []
    lines.append(f"Φ-value Comparison Report — {protein_name}")
    lines.append("=" * 60)
    citation = _CITATIONS.get(protein_name, _CITATIONS["_default"])
    lines.append(f"\nExperimental Φ-values ({citation}):")
    sub = df.dropna(subset=["phi_exp"])
    for _, row in sub.iterrows():
        flag = " [HIGH-SCORE CANDIDATE]" if row.get("is_nucleus") else ""
        imp_str = (f"  importance={row['importance']:.3f}"
                   if "importance" in row and not np.isnan(row.get("importance", float("nan")))
                   else "")
        coop_str = (f"  coop_score={row['coop_score']:.3f}"
                    if "coop_score" in row and not np.isnan(row.get("coop_score", float("nan")))
                    else "")
        lines.append(
            f"  Res {int(row['residue']):3d} : Φ_exp={row['phi_exp']:.2f}  "
            f"nucleus_score={row['nucleus_score']:.3f}{imp_str}{coop_str}{flag}"
        )

    lines.append("\n--- Correlations ---")

    def _fmt(corr, label):
        if corr is None:
            return f"  {label}: insufficient data (need ≥ 3 points)"
        return (
            f"  {label}:\n"
            f"    Pearson  r = {corr['pearson_r']:+.3f}  (p = {corr['pearson_p']:.3g})\n"
            f"    Spearman r = {corr['spearman_r']:+.3f}  (p = {corr['spearman_p']:.3g})\n"
            f"    n = {corr['n']}"
        )

    lines.append(_fmt(corr_nucleus, "FES (legacy nucleus_score) vs Φ_exp"))
    lines.append(_fmt(corr_imp,     "importance    vs Φ_exp"))
    if corr_coop is not None:
        lines.append(_fmt(corr_coop, "coop_score    vs Φ_exp"))
    if corr_committor_coop is not None:
        lines.append(_fmt(corr_committor_coop, "committor_coop vs Φ_exp"))

    lines.append("\n--- Interpretation ---")
    lines.append(
        "  This is an exploratory association analysis, not a nucleus validation.\n"
        "  The legacy 'is_nucleus' field denotes a thresholded score candidate.\n"
        "  A positive correlation does not establish folding order or causality.\n"
        "  Interpret effect size together with n, uncertainty, mapping quality and p-values.\n"
    )

    text = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"  Phi correlation report → {out_path}")
    print("\n" + text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(protein_name, results_dir, pdb_id=None):
    print(f"\n=== Exploratory Φ-value comparison for {protein_name} ===")
    print("  WARNING: bundled tables are smoke-test data, not a curated benchmark.")

    phi_table = None
    citation = _CITATIONS.get(protein_name, _CITATIONS["_default"])

    # Priority 1: load a user-supplied audited workbook using pdb_id.
    if pdb_id is not None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xlsx_path = os.path.join(base_dir, "data", "EVO3_Strict_Database.xlsx")
        if os.path.exists(xlsx_path):
            try:
                from data.kprodb_adapter import extract_phi_tables
                all_tables = extract_phi_tables(xlsx_path)
                phi_table = all_tables.get(pdb_id)
                if phi_table is not None:
                    citation = _CITATIONS["_default"]
            except Exception:
                pass

    # Priority 2: fallback to approximate hardcoded smoke-test tables.
    if phi_table is None:
        phi_table = PHI_TABLES.get(protein_name)

    if phi_table is None:
        print(f"  No Φ-value table for {protein_name} (pdb_id={pdb_id}). "
              f"Hardcoded proteins: {list(PHI_TABLES.keys())}")
        return

    pred_df = load_predictions(results_dir, protein_name)
    df      = build_comparison_df(pred_df, phi_table)

    out_dir = os.path.join(results_dir, protein_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save merged CSV
    csv_path = os.path.join(out_dir, "phi_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Comparison CSV → {csv_path}")

    # Correlations
    corr_ns            = compute_correlations(df, "nucleus_score")
    corr_imp           = compute_correlations(df, "importance")           if "importance"           in df.columns else None
    corr_coop          = compute_correlations(df, "coop_score")           if "coop_score"           in df.columns else None
    corr_committor_coop = compute_correlations(df, "committor_coop_score") if "committor_coop_score" in df.columns else None

    # Plots
    plot_phi_scatter(df,
        out_path=os.path.join(out_dir, "phi_scatter.png"),
        protein_name=protein_name)
    plot_phi_profile(df,
        out_path=os.path.join(out_dir, "phi_profile.png"),
        protein_name=protein_name)

    # Text report
    write_report(df, corr_ns, corr_imp, protein_name,
                 out_path=os.path.join(out_dir, "phi_correlation.txt"),
                 corr_coop=corr_coop,
                 corr_committor_coop=corr_committor_coop)

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein",      default="Villin_HP36")
    ap.add_argument("--results-dir",  default="results")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(
        protein_name=args.protein,
        results_dir=(args.results_dir if os.path.isabs(args.results_dir)
                     else os.path.join(base, args.results_dir)),
    )


if __name__ == "__main__":
    main()
