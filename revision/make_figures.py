"""Figures for the revised manuscript from revision_outputs_mapped/ (and _positional for Fig. S6).

Usage: python make_figures.py
Writes figures/ (TIFF 300 dpi for the main text, PNG for Online Resource 1).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent
M = ROOT / "revision_outputs_mapped"
P = ROOT / "revision_outputs_positional"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, RED, GRAY = "#3a76b8", "#e0735f", "#9a9a9a"
LABELS = {"A0": "A0\nFES", "B1": "B1\nRandom", "B2": "B2\nAll\nresidue", "B3": "B3\nConf.\nonly",
          "B4": "B4\nContact\ndegree", "A1": "A1\nSliding\nwindow", "A2": "A2\nMarginal\nonly",
          "A3": "A3\nNo conf.\ngate/term", "A4": "A4\nNo conf.\nterm", "A5": "A5\nTopology\nonly"}
ORDER = list(LABELS)
LEVEL_NAME = {"early": "EARLY (kinetic HDX protection)", "strong": "STRONG (equilibrium HDX protection)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})

long = pd.read_csv(M / "metrics_long.csv")
paired = pd.read_csv(M / "paired_comparisons.csv")


def save(fig, name, main=False):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    if main:
        fig.savefig(FIG / f"{name}.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# Fig. 2: AUC-ROC distributions, EARLY and STRONG
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, level, color in zip(axes, ("early", "strong"), (BLUE, RED)):
    data = [long[(long.level == level) & (long.method == m)].auc_roc.dropna() for m in ORDER]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "black", "linewidth": 1.6},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none", "markeredgecolor": GRAY})
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=color, alpha=1.0 if i == 0 else 0.55, edgecolor="#333333", linewidth=0.8)
    ax.axhline(0.5, ls="--", lw=0.8, color=GRAY)
    sub = paired[(paired.level == level) & (paired.metric == "auc_roc")].set_index("comparator")
    for i, m in enumerate(ORDER[1:], start=2):
        ax.text(i, 1.02, stars(sub.loc[m, "holm_p_two_sided"]), ha="center", fontsize=8)
    ax.set_xticks(range(1, len(ORDER) + 1), [LABELS[m] for m in ORDER], fontsize=7)
    ax.set_title(f"{LEVEL_NAME[level]}  (n = {len(data[0])})", fontsize=10)
    ax.set_ylim(0, 1.07)
axes[0].set_ylabel("Per-protein AUC-ROC")
save(fig, "Figure_2", main=True)

# Fig. 3: EARLY P@K, FES vs confidence-only
piv = long[long.level == "early"].pivot_table(index="name", columns="method", values="precision_at_k")[["A0", "B3"]].dropna()
d = piv.A0 - piv.B3
fig, ax = plt.subplots(figsize=(4.6, 4.3))
for mask, c, mk, lab in ((d > 1e-12, BLUE, "o", "FES higher"), (d < -1e-12, RED, "^", "Confidence higher"),
                         (d.abs() <= 1e-12, GRAY, "s", "Tied")):
    ax.scatter(piv.B3[mask], piv.A0[mask], s=36, c=c, marker=mk, edgecolor="white", linewidth=0.8,
               label=f"{lab} ({int(mask.sum())})", zorder=3)
lim = max(piv.max()) * 1.05 + 0.02
ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color=GRAY)
ax.set(xlim=(-0.02, lim), ylim=(-0.02, lim), xlabel="B3, confidence-only P@K", ylabel="A0, FES P@K")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"EARLY protection, n = {len(piv)} proteins", fontsize=10)
save(fig, "Figure_3", main=True)

# Fig. 4: EARLY AUC vs length
sub = long[(long.level == "early") & (long.method == "A0")]
rho, p = spearmanr(sub.length, sub.auc_roc)
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(sub.length, sub.auc_roc, s=34, c=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
z = np.polyfit(sub.length, sub.auc_roc, 1)
xs = np.linspace(sub.length.min(), sub.length.max(), 50)
ax.plot(xs, np.polyval(z, xs), color="#555555", lw=1.2, label=f"Linear trend (Spearman rho = {rho:.2f}, P = {p:.3f})")
ax.axhline(0.5, ls=":", lw=0.8, color=GRAY)
for (_, r), dy in zip(sub.nlargest(2, "auc_roc").iterrows(), (3, -9)):
    ax.annotate(r.pdb_id, (r.length, r.auc_roc), xytext=(6, dy), textcoords="offset points", fontsize=7)
for _, r in sub.nsmallest(2, "auc_roc").iterrows():
    ax.annotate(r.pdb_id, (r.length, r.auc_roc), xytext=(4, -8), textcoords="offset points", fontsize=7)
ax.set(xlabel="Protein length (residues)", ylabel="EARLY AUC-ROC (FES)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
save(fig, "Figure_4", main=True)

# Fig. S1: FES AUC across all six classes
levels = ["early", "intermediate", "late", "strong", "medium", "weak"]
fig, ax = plt.subplots(figsize=(7, 3.8))
data = [long[(long.level == lv) & (long.method == "A0")].auc_roc for lv in levels]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "black"})
for i, box in enumerate(bp["boxes"]):
    box.set(facecolor=BLUE if i < 3 else RED, alpha=0.7)
ax.axhline(0.5, ls="--", lw=0.8, color=GRAY)
ax.set_xticks(range(1, 7), [f"{lv.upper()}\n(n = {len(x)})" for lv, x in zip(levels, data)])
ax.set_ylabel("Per-protein AUC-ROC (FES)")
save(fig, "Figure_S1")

# Fig. S2: paired differences A0 - comparator with bootstrap 95% CI (EARLY)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, metric, title in zip(axes, ("auc_roc", "precision_at_k"), ("AUC-ROC", "P@K")):
    sub = paired[(paired.level == "early") & (paired.metric == metric)].set_index("comparator").loc[ORDER[1:]]
    ci = sub.median_diff_CI.apply(lambda s: [float(x) for x in s.strip("()").split(",")])
    y = np.arange(len(sub))[::-1]
    ax.errorbar(sub.median_paired_diff, y, xerr=[sub.median_paired_diff - ci.str[0], ci.str[1] - sub.median_paired_diff],
                fmt="o", color=BLUE, ecolor="#555555", capsize=3, ms=5)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_yticks(y, [LABELS[c].replace("\n", " ") for c in sub.index], fontsize=8)
    ax.set_xlabel(f"Median paired difference, A0 minus comparator ({title})")
save(fig, "Figure_S2")

# Fig. S3: protein-by-method EARLY AUC heatmap
hm = long[long.level == "early"].pivot_table(index=["pdb_id", "length"], columns="method", values="auc_roc")[ORDER]
hm = hm.sort_index(level="length")
fig, ax = plt.subplots(figsize=(6, 9))
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ORDER)), [o for o in ORDER])
ax.set_yticks(range(len(hm)), [f"{p} ({l})" for p, l in hm.index], fontsize=6.5)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.5, label="EARLY AUC-ROC")
save(fig, "Figure_S3")

# Fig. S4: ROC curves, six representative proteins (spanning the observed range)
res = pd.read_csv(M / "residue_scores_all_methods.csv.gz")
early = sub_early = long[(long.level == "early") & (long.method == "A0")].sort_values("auc_roc")
pick = early.iloc[np.linspace(0, len(early) - 1, 6).round().astype(int)][::-1]
fig, axes = plt.subplots(2, 3, figsize=(9, 6.4), gridspec_kw={"hspace": 0.45})
for ax, (_, r) in zip(axes.flat, pick.iterrows()):
    g = res[res.name == r["name"]]
    for m, c, ls in (("A0", BLUE, "-"), ("B3", RED, "--")):
        ok = g[m].notna()
        fpr, tpr, _ = roc_curve(g.label_early[ok], g[m][ok])
        a = long[(long.name == r["name"]) & (long.level == "early") & (long.method == m)].auc_roc.iloc[0]
        ax.plot(fpr, tpr, color=c, ls=ls, lw=1.5, label=f"{m} AUC {a:.2f}")
    ax.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=0.8)
    ax.set_title(f"{r.pdb_id} ({r.length} residues)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
for ax in axes[-1]:
    ax.set_xlabel("False positive rate")
for ax in axes[:, 0]:
    ax.set_ylabel("True positive rate")
save(fig, "Figure_S4")

# Fig. S5: sensitivity to the axis weight and to the other heuristics
sens = pd.read_csv(M / "sensitivity_summary.csv")
sweep = sens[sens.variant.str.startswith("w_marg=")].copy()
sweep["w"] = sweep.variant.str.split("=").str[1].astype(float)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75})
ax = axes[0]
for col, c, ls, mk, lab in (("early_median_auc", BLUE, "-", "o", "EARLY AUC-ROC"),
                            ("strong_median_auc", RED, "-", "o", "STRONG AUC-ROC"),
                            ("early_median_p_at_k", BLUE, "--", "^", "EARLY P@K"),
                            ("strong_median_p_at_k", RED, "--", "^", "STRONG P@K")):
    ax.plot(sweep.w, sweep[col], color=c, ls=ls, marker=mk, ms=4.5, lw=1.5, label=lab)
w0 = 0.35 / 0.60
ax.axvline(w0, color="#555555", lw=0.8, ls=":")
ax.text(w0 - 0.02, 0.52, "default\n(0.583)", fontsize=7.5, ha="right")
ax.set(xlabel="Effective marginal-axis weight w (topology weight 1 - w)", ylabel="Median across proteins")
ax.legend(frameon=False, fontsize=7.5, loc="center left")
ax = axes[1]
NICE = {"A0": "FES default (A0)", "A3": "No confidence gate or term (A3)", "A4": "No confidence term (A4)",
        "A5": "Topology axis only (A5)", "nominal_equal_1/3": "Equal nominal weights (1/3)",
        "Q_weight=1": "Q coefficient 1", "Q_weight=3": "Q coefficient 3",
        "gate-10": "Gate thresholds -10", "gate-5": "Gate thresholds -5",
        "gate+5": "Gate thresholds +5", "gate+10": "Gate thresholds +10",
        "reduced_factor=0.25": "Reduced-weight factor 0.25", "reduced_factor=0.75": "Reduced-weight factor 0.75",
        "confidence_term=0.5": "Confidence term x0.5", "confidence_term=2": "Confidence term x2"}
other = sens[~sens.variant.str.startswith("w_marg=") & ~sens.variant.isin(["A1", "A2"])].iloc[::-1]
y = np.arange(len(other))
ax.scatter(other.early_median_auc, y, color=BLUE, marker="o", label="EARLY AUC-ROC", zorder=3)
ax.scatter(other.early_median_p_at_k, y, color=BLUE, marker="^", label="EARLY P@K", zorder=3)
ax.scatter(other.strong_median_auc, y, color=RED, marker="o", label="STRONG AUC-ROC", zorder=3)
ax.set_yticks(y, [NICE.get(v, v) for v in other.variant], fontsize=8)
ax.set_xlabel("Median across proteins")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
save(fig, "Figure_S5")

# Fig. S6: effect of the residue-mapping correction on per-protein EARLY AUC
if (P / "metrics_long.csv").exists():
    pos = pd.read_csv(P / "metrics_long.csv")
    a = pos[(pos.level == "early") & (pos.method == "A0")].set_index("name").auc_roc
    b = long[(long.level == "early") & (long.method == "A0")].set_index("name").auc_roc
    diag = pd.read_csv(M / "protein_diagnostics.csv").set_index("name")
    affected = (diag.positional_identity < 0.9) | (diag.label_offset != 0) | (diag.native_identity < 0.9)
    both = pd.concat([a.rename("pos"), b.rename("map")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    for flag, c, mk, lab in ((False, GRAY, "o", "mapping unchanged"), (True, RED, "^", "mapping corrected")):
        s = both[affected.reindex(both.index).fillna(False) == flag]
        ax.scatter(s["pos"], s["map"], c=c, marker=mk, s=34, edgecolor="white", linewidth=0.8, label=f"{lab} ({len(s)})", zorder=3)
    ax.plot([0.2, 1], [0.2, 1], ls="--", color=GRAY, lw=0.8)
    ax.set(xlabel="EARLY AUC-ROC, v1.1.0 positional mapping", ylabel="EARLY AUC-ROC, sequence-aligned mapping")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "Figure_S6")

# Fig. S7: computational cost
t = pd.read_csv(ROOT / "logs" / "esmfold_timing.tsv", sep="\t")
t["input_len"] = t["length"].astype(int) + 30
cost = pd.read_csv(M / "esmfold_cost_per_protein.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].scatter(t.input_len, t.seconds, s=3, color=BLUE, alpha=0.4)
axes[0].set(xscale="log", yscale="log", xlabel="ESMFold input length (prefix + 30 Gly)", ylabel="Wall-clock time per prefix (s)")
axes[1].scatter(cost.length, cost.total_s / 60, s=26, color=BLUE, edgecolor="white")
axes[1].set(xlabel="Protein length (residues)", ylabel="Total scan time (min)")
save(fig, "Figure_S7")
print(sorted(p.name for p in FIG.iterdir()))
