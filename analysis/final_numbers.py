"""Collect every number cited in the documents into FINAL_NUMBERS.json.

Primary analysis: revision_outputs_mapped (sequence-aligned). Index-paired comparison:
revision_outputs_positional. Flank control: revision_outputs_flank0 and the same
proteins in the primary analysis.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
M, P, F = (ROOT / f"revision_outputs_{m}" for m in ("mapped", "positional", "flank0"))
LEVELS = ["early", "intermediate", "late", "strong", "medium", "weak"]


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def block(out_dir):
    t2 = pd.read_csv(out_dir / "table2_fes_by_class.csv").set_index("level")
    pc = pd.read_csv(out_dir / "paired_comparisons.csv")
    long = pd.read_csv(out_dir / "metrics_long.csv")
    res = {"table2": {lv: {k: r3(v) if k != "n" else int(v) for k, v in t2.loc[lv].items()} for lv in t2.index},
           "n_evaluations": int(t2.n.sum())}
    res["medians"] = {lv: {m: {k: r3(g[k].median()) for k in ("auc_roc", "precision_at_k", "p_at_5", "p_at_10",
                                                                 "p_at_10pct", "f1", "mcc")}
                           for m, g in long[long.level == lv].groupby("method")} for lv in LEVELS}
    res["paired"] = {}
    for _, r in pc.iterrows():
        lo, hi = (float(x) for x in r.median_diff_CI.strip("()").split(","))
        res["paired"].setdefault(r.level, {}).setdefault(r.metric, {})[r.comparator] = {
            "median_A0": r3(r.median_A0), "median_comp": r3(r.median_comparator),
            "median_diff": r3(r.median_paired_diff), "ci": [r3(lo), r3(hi)],
            "mean_diff": r3(r.mean_paired_diff), "wins": int(r.wins), "losses": int(r.losses), "ties": int(r.ties),
            "p_greater": float(r.wilcoxon_p_greater), "p_two": float(r.wilcoxon_p_two_sided),
            "holm_two": float(r.holm_p_two_sided), "holm_greater": float(r.holm_p_greater),
            "sign_p": float(r.sign_test_p_two_sided), "cliff": r3(r.cliffs_delta)}
    la = pd.read_csv(out_dir / "length_association.csv").set_index("method")
    res["length"] = {m: {"rho": r3(la.loc[m, "spearman_rho"]), "p": float(la.loc[m, "p"]), "n": int(la.loc[m, "n"])}
                     for m in la.index}
    return res, long


def main():
    out = {}
    out["mapped"], long_m = block(M)
    out["positional"], long_p = block(P)

    sens = pd.read_csv(M / "sensitivity_summary.csv").set_index("variant")
    out["sensitivity"] = {v: {k: r3(x) for k, x in row.items()} for v, row in sens.iterrows()}
    cons = pd.read_csv(M / "early_patk_consistency.csv").set_index("comparator")
    out["consistency"] = {c: {k: r3(x) for k, x in row.items()} for c, row in cons.iterrows()}
    out["composition"] = pd.read_csv(M / "benchmark_composition.csv").set_index("level").to_dict(orient="index")
    out["cost"] = json.loads((M / "esmfold_cost.json").read_text())
    cpp = pd.read_csv(M / "esmfold_cost_per_protein.csv")
    near = cpp[(cpp.length >= 90) & (cpp.length <= 110)]
    out["cost"]["median_min_90_110"] = r3(np.median(near.total_s) / 60)
    out["cost"]["range_min_90_110"] = [r3(near.total_s.min() / 60), r3(near.total_s.max() / 60)]
    out["cost"]["longest"] = {"length": int(cpp.length.max()), "min": r3(cpp.loc[cpp.length.idxmax(), "total_s"] / 60)}

    diag = pd.read_csv(M / "protein_diagnostics.csv")
    out["mapping"] = {
        "native_positional_identity_lt_0.9": sorted(diag.loc[diag.positional_identity < 0.9, "pdb_id"]),
        "label_shifted": {r.pdb_id: int(r.label_offset) for r in diag.itertuples() if r.label_offset != 0},
        "label_letter_match_min": r3(diag.label_checksum.min()),
        "labels_dropped_total": int(diag.labels_dropped.sum()),
        "native_identity_min": r3(diag.native_identity.min()),
        "effective_weights": sorted({(r3(a), r3(b), r3(c)) for a, b, c in
                                     diag[["w_marginal", "w_intermediate", "w_topology"]].itertuples(index=False)}),
    }
    pdiag = pd.read_csv(P / "protein_diagnostics.csv")
    out["reproduction_max_abs_diff"] = float(pdiag.max_abs_diff_rescore_vs_published.max())
    out["lengths"] = {"min": int(diag.n_res.min()), "max": int(diag.n_res.max()), "median": r3(diag.n_res.median())}

    # per-protein extremes, EARLY, primary analysis
    e = long_m[(long_m.level == "early") & (long_m.method == "A0")].sort_values("auc_roc")
    out["early_extremes"] = {"top": e.tail(4)[["pdb_id", "length", "auc_roc"]].round(3).values.tolist()[::-1],
                             "bottom": e.head(3)[["pdb_id", "length", "auc_roc"]].round(3).values.tolist()}

    # flank control: same proteins, 15-Gly (primary) versus no flanks
    if (F / "metrics_long.csv").exists():
        lf = pd.read_csv(F / "metrics_long.csv")
        fc = {}
        for lv in ("early", "strong"):
            for m in ("A0", "B3"):
                a = lf[(lf.level == lv) & (lf.method == m)].set_index("name")
                b = long_m[(long_m.level == lv) & (long_m.method == m)].set_index("name").loc[a.index]
                for metric in ("auc_roc", "precision_at_k"):
                    d = b[metric] - a[metric]
                    fc[f"{lv}_{m}_{metric}"] = {"n": len(d), "median_15gly": r3(b[metric].median()),
                                                "median_0gly": r3(a[metric].median()), "median_diff": r3(d.median()),
                                                "p_two": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0}
        tf = pd.read_csv(ROOT / "logs" / "flank0_timing.tsv", sep="\t")
        t15 = pd.read_csv(ROOT / "logs" / "esmfold_timing.tsv", sep="\t")
        t15 = t15[t15.protein.isin(tf.protein.unique())]
        fc["rejected_0gly"] = int(tf.status.str.startswith("rejected").sum())
        fc["rejected_15gly"] = int(t15.status.str.startswith("rejected").sum())
        fc["prefixes"] = len(tf)
        fc["proteins"] = sorted(n.split("_")[-1] for n in tf.protein.unique())
        out["flank_control"] = fc

    (ROOT.parent / "FINAL_NUMBERS.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("mapping", "cost", "lengths", "reproduction_max_abs_diff")}, indent=1, default=str))


if __name__ == "__main__":
    main()
