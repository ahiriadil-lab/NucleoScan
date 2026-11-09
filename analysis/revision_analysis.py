"""Benchmark analyses for the NucleoScan manuscript.

Reads the v1.1.0 predictions (results_rerun/, structures_rerun/), rescores every protein
under the published configuration, the four ablations and a sensitivity grid,
and writes tables to revision_outputs/.

Rescoring reuses the cached per-fragment metrics (fragment_metrics.csv) and the
v1.1.0 scoring functions; only module constants are changed per variant. The
published configuration is checked to reproduce residue_scores.csv exactly.

Usage: python revision_analysis.py [positional|mapped]
  positional  index-based residue pairing and raw Start2Fold numbering
  mapped      sequence-aligned native reference and labels (mapping.py; reported analysis)
  flank0      as mapped, on the flank-control predictions (structures_flank0/, no glycine flanks)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import core.score as cs  # noqa: E402
from modules.importance import compute_importance_from_native  # noqa: E402
from validation.advanced_metrics import bootstrap_metric_ci, compute_mcc  # noqa: E402
from validation.baselines import (_stable_random_scores, cliffs_delta,  # noqa: E402
                                  confidence_baseline, evaluate_scores,
                                  native_contact_degree, precision_at_k_tie_aware)
from validation.start2fold import ALL_LEVELS, load_start2fold  # noqa: E402

RESULTS = ROOT / "results_rerun"
MODE = sys.argv[1] if len(sys.argv) > 1 else "mapped"
if MODE == "flank0":
    os.environ["NS_STRUCTURES"] = "structures_flank0"
STRUCTURES = ROOT / os.environ.get("NS_STRUCTURES", "structures_rerun")
PDB_DIR = ROOT / "data" / "pdb_cleaned"
OUT = ROOT / f"revision_outputs_{MODE}"
SEED = 42
N_BOOT = 10000

GATE = ["PLDDT_REJECT_THRESHOLD", "PLDDT_REDUCED_WEIGHT_THRESHOLD",
        "PLDDT_BETA_REJECT_THRESHOLD", "PLDDT_BETA_REDUCED_THRESHOLD",
        "PLDDT_ALPHA_REJECT_THRESHOLD", "PLDDT_ALPHA_REDUCED_THRESHOLD",
        "PLDDT_COIL_REJECT_THRESHOLD", "PLDDT_COIL_REDUCED_THRESHOLD"]


def shifted_gate(delta):
    return {name: getattr(cs, name) + delta for name in GATE}


# variant -> (scoring method, core.score constants to override)
VARIANTS = {
    "A0": ("unified", {}),
    "A1": ("sliding_window", {}),
    "A2": ("marginal", {}),
    "A3": ("unified", {"PLDDT_WEIGHT": 0.0, **{n: -np.inf for n in GATE}}),
    "A4": ("unified", {"PLDDT_WEIGHT": 0.0}),
    # native-topology axis alone (no prediction input)
    "A5": ("unified", {"UNIFIED_W_KINETIC": 0.0, "UNIFIED_W_COOPERATIVE": 0.0, "UNIFIED_W_TOPOLOGICAL": 1.0}),
}
# Effective axis weights: marginal w, topology 1 - w (intermediate-contact axis is empty
# with one prediction per prefix). w = 0.5 is the effective result of equal nominal weights.
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    VARIANTS[f"w_marg={w:.1f}"] = ("unified", {"UNIFIED_W_KINETIC": w, "UNIFIED_W_COOPERATIVE": 0.0,
                                               "UNIFIED_W_TOPOLOGICAL": 1 - w})
VARIANTS["nominal_equal_1/3"] = ("unified", {"UNIFIED_W_KINETIC": 1 / 3, "UNIFIED_W_COOPERATIVE": 1 / 3,
                                             "UNIFIED_W_TOPOLOGICAL": 1 / 3})
for q in (1.0, 3.0):
    VARIANTS[f"Q_weight={q:g}"] = ("unified", {"Q_WEIGHT": q})
for d in (-10, -5, 5, 10):
    VARIANTS[f"gate{d:+d}"] = ("unified", shifted_gate(d))
for f in (0.25, 0.75):
    VARIANTS[f"reduced_factor={f}"] = ("unified", {"PLDDT_REDUCED_WEIGHT_FACTOR": f})
for p in (0.5, 2.0):
    VARIANTS[f"confidence_term={p:g}"] = ("unified", {"PLDDT_WEIGHT": p})

SCORE_COLS = ["length", "rmsd_mean", "q_mean", "q_local_mean", "hydro_rg_mean", "plddt_mean",
              "ss_dominant", "packing_mean", "hydro_burial_mean"]


@contextmanager
def patched(overrides):
    old = {k: getattr(cs, k) for k in overrides}
    for k, v in overrides.items():
        setattr(cs, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(cs, k, v)


def rescore(metrics, n_res, importance, variant):
    method, overrides = VARIANTS[variant]
    with patched(overrides), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fragment_df = cs.compute_fragment_scores(metrics)
        res = cs.score_residues(fragment_df, max_length=n_res, method=method,
                                importance_df=importance, ref_pdb_path=None)
    return res.sort_values("residue")


def protein_job(name):
    """All score vectors for one protein: variants and baselines, plus labels and evaluable mask."""
    pdir = RESULTS / name
    pdb_id = (next(pdir.glob("*.fasta")).stem).lower()
    published = pd.read_csv(pdir / "residue_scores.csv").sort_values("residue")
    n_res = len(published)
    native = PDB_DIR / f"{pdb_id}_A.pdb"
    s2f = load_start2fold(str(ROOT / "data" / "start2fold_data.json"))[pdb_id]
    if MODE in ("mapped", "flank0"):
        import mapping as mp
        fm = mp.mapped_fragment_metrics(name)
        importance = mp.mapped_importance(name)
        labels = {lv: mp.mapped_labels(name, lv) for lv in ALL_LEVELS}
        mask = mp.evaluable_mask(name)
        b4 = mp.mapped_contact_degree(name)
        mp_info = {k: v for k, v in mp.mapping(name).items() if k in (
            "label_offset", "label_checksum", "labels_dropped", "native_identity", "native_mapped", "native_len")}
    else:
        fm = pd.read_csv(pdir / "fragment_metrics.csv")
        try:
            importance = compute_importance_from_native(str(native))
        except ValueError:  # multi-model NMR file (1yyx): v1.1.0 leaves the topology axis at zero
            importance = None
        labels = {lv: s2f[lv] for lv in ALL_LEVELS}
        mask = np.ones(n_res, dtype=bool)
        b4 = native_contact_degree(native, n_res)
        mp_info = {}
    metrics = fm[[c for c in SCORE_COLS if c in fm]].to_dict("records")

    scores, flags = {}, {}
    for v in VARIANTS:
        res = rescore(metrics, n_res, importance, v)
        scores[v] = res["nucleus_score"].to_numpy(float)
        flags[v] = res["is_nucleus"].to_numpy(bool)
    check = float(np.max(np.abs(scores["A0"] - published["nucleus_score"].to_numpy(float))))

    scores["B1"] = _stable_random_scores(pdb_id, n_res, SEED)
    scores["B2"] = np.ones(n_res)
    scores["B3"] = confidence_baseline(STRUCTURES / name, n_res)
    scores["B4"] = b4
    for v in scores:
        scores[v] = np.where(mask, np.asarray(scores[v], float), np.nan)

    # native-reference diagnostics (residue mapping is positional in v1.1.0)
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
    chain = next(PDBParser(QUIET=True).get_structure(pdb_id, str(native))[0].get_chains())
    nat_seq = "".join(seq1(r.get_resname()) for r in chain if is_aa(r, standard=True))
    seq = "".join(l.strip() for l in (pdir / f"{pdb_id}.fasta").read_text().splitlines()[1:])
    m = min(len(seq), len(nat_seq))
    diag = {
        "name": name, "pdb_id": pdb_id, "n_res": n_res, "native_res": len(nat_seq),
        "positional_identity": float(np.mean([a == b for a, b in zip(seq[:m], nat_seq[:m])])) if m else np.nan,
        "n_prefixes_expected": n_res - 2, "n_prefixes_scored": len(fm),
        "w_marginal": float(published["w_kinetic"].iloc[0]),
        "w_intermediate": float(published["w_cooperative"].iloc[0]),
        "w_topology": float(published["w_topological"].iloc[0]),
        "max_abs_diff_rescore_vs_published": check, "evaluable_positions": int(mask.sum()), **mp_info,
    }
    return name, pdb_id, scores, flags, diag, labels


def evaluate_all(jobs, s2f):
    rows = []
    for name, pdb_id, scores, flags, diag, labels in jobs:
        n_res = diag["n_res"]
        for level in ALL_LEVELS:
            protected = labels[level]
            if not protected:
                continue
            for method, sc in scores.items():
                m = evaluate_scores(np.asarray(sc, float), protected)
                if m is None:
                    continue
                lab = np.array([int(r in protected) for r in range(1, n_res + 1)])
                valid = np.isfinite(sc)
                extra = {f"p_at_{k}": precision_at_k_tie_aware(sc[valid], lab[valid], k) for k in (5, 10)}
                extra["p_at_10pct"] = precision_at_k_tie_aware(sc[valid], lab[valid], max(1, round(0.1 * n_res)))
                if method in flags:
                    pred = set(np.flatnonzero(flags[method] & valid) + 1)
                    prot = {r for r in protected if 1 <= r <= n_res and valid[r - 1]}
                    tp = len(pred & prot)
                    prec = tp / len(pred) if pred else 0.0
                    rec = tp / len(prot) if prot else 0.0
                    extra["f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                    extra["mcc"] = compute_mcc(pred, prot, n_res)
                rows.append({"name": name, "pdb_id": pdb_id, "level": level, "method": method,
                             "length": n_res, **m, **extra})
    return pd.DataFrame(rows)


def boot_ci(diff, stat=np.median, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    b = stat(diff[idx], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_table(long, level, metric, comparators, ref="A0"):
    piv = long[long.level == level].pivot_table(index="name", columns="method", values=metric)
    if ref not in piv:
        return pd.DataFrame()
    rows = []
    for c in comparators:
        pair = piv[[ref, c]].dropna()
        x, y = pair[ref].to_numpy(), pair[c].to_numpy()
        d = x - y
        nz = d[np.abs(d) > 1e-12]
        p_gt = float(wilcoxon(x, y, alternative="greater").pvalue) if len(nz) else 1.0
        p_2s = float(wilcoxon(x, y).pvalue) if len(nz) else 1.0
        wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        rows.append({
            "level": level, "metric": metric, "comparator": c, "n": len(pair),
            "median_A0": np.median(x), "median_comparator": np.median(y),
            "median_paired_diff": np.median(d), "median_diff_CI": boot_ci(d),
            "mean_paired_diff": d.mean(), "mean_diff_CI": boot_ci(d, np.mean),
            "wins": wins, "losses": losses, "ties": len(d) - wins - losses,
            "sign_test_p_two_sided": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0,
            "wilcoxon_p_greater": p_gt, "wilcoxon_p_two_sided": p_2s,
            "cliffs_delta": cliffs_delta(x, y),
        })
    df = pd.DataFrame(rows)
    df["holm_p_greater"] = holm(df["wilcoxon_p_greater"])
    df["holm_p_two_sided"] = holm(df["wilcoxon_p_two_sided"])
    return df


def main():
    OUT.mkdir(exist_ok=True)
    s2f = load_start2fold(str(ROOT / "data" / "start2fold_data.json"))
    names = sorted(d.name for d in RESULTS.iterdir() if (d / "residue_scores.csv").exists()
                   and (STRUCTURES / d.name).exists())
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(protein_job, names))

    diag = pd.DataFrame([j[4] for j in jobs])
    diag.to_csv(OUT / "protein_diagnostics.csv", index=False)
    long = evaluate_all(jobs, s2f)
    long.to_csv(OUT / "metrics_long.csv", index=False)
    pd.DataFrame([{"name": n, "residue": i + 1, **{m: s[i] for m, s in sc.items()},
                   **{f"label_{lv}": int(i + 1 in lab[lv]) for lv in ALL_LEVELS}}
                  for n, _, sc, _, _, lab in jobs for i in range(len(sc["A0"]))]).to_csv(
        OUT / "residue_scores_all_methods.csv.gz", index=False)

    # Table: benchmark composition per class
    comp = []
    for level in ALL_LEVELS:
        a0 = long[(long.level == level) & (long.method == "A0")]
        comp.append({"level": level, "proteins": len(a0), "protected_residues": int(a0.n_protected.sum()),
                     "median_protected_per_protein": float(a0.n_protected.median()),
                     "residues_scored": int(a0.n_scored.sum()),
                     "median_prevalence": float((a0.n_protected / a0.n_scored).median())})
    pd.DataFrame(comp).to_csv(OUT / "benchmark_composition.csv", index=False)

    # Table 2: FES per class
    t2 = []
    for level in ALL_LEVELS:
        a0 = long[(long.level == level) & (long.method == "A0")]
        lo, hi, med = bootstrap_metric_ci(a0.auc_roc.tolist())
        t2.append({"level": level, "n": len(a0), "median_auc": med, "auc_ci_lo": lo, "auc_ci_hi": hi,
                   "median_p_at_k": a0.precision_at_k.median(), "median_f1": a0.f1.median(),
                   "median_mcc": a0.mcc.median()})
    pd.DataFrame(t2).to_csv(OUT / "table2_fes_by_class.csv", index=False)

    # Table 3 and P@K comparisons: EARLY primary, all classes exploratory
    comps = ["B1", "B2", "B3", "B4", "A1", "A2", "A3", "A4", "A5"]
    paired = pd.concat([paired_table(long, lv, m, comps) for lv in ALL_LEVELS
                        for m in ("auc_roc", "precision_at_k", "p_at_5", "p_at_10", "p_at_10pct")])
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)

    # Head-of-ranking consistency, EARLY, FES vs B3 and B4
    cons = []
    for c in ("B3", "B4", "A5"):
        piv = long[long.level == "early"].pivot_table(index="name", columns="method",
                                                       values=["precision_at_k", "length"])
        d = (piv["precision_at_k"]["A0"] - piv["precision_at_k"][c]).dropna().sort_values()
        lengths = piv["length"]["A0"].loc[d.index]
        loo = [np.median(d.drop(i)) for i in d.index]
        gains = d[d > 0].sort_values(ascending=False)
        rho, p = spearmanr(lengths, d)
        cons.append({"comparator": c, "n": len(d), "median_diff": d.median(),
                     "loo_median_min": min(loo), "loo_median_max": max(loo),
                     "share_of_positive_gain_top5": gains.head(5).sum() / gains.sum() if len(gains) else np.nan,
                     "median_diff_without_top5_gains": np.median(d.drop(gains.head(5).index)),
                     "spearman_diff_vs_length": rho, "spearman_p": p,
                     **{f"median_diff_len_tertile{i + 1}": g.median()
                        for i, (_, g) in enumerate(d.groupby(pd.qcut(lengths, 3)))}})
        d.rename("p_at_k_diff").to_frame().join(lengths.rename("length")).to_csv(
            OUT / f"early_patk_diff_A0_vs_{c}.csv")
    pd.DataFrame(cons).to_csv(OUT / "early_patk_consistency.csv", index=False)

    # Sensitivity summary (all variants vs published A0)
    sens = []
    a0_scores = {n: sc["A0"] for n, _, sc, _, _, _ in jobs}
    for v in VARIANTS:
        row = {"variant": v}
        for level in ("early", "strong"):
            sub = long[(long.level == level) & (long.method == v)]
            row[f"{level}_median_auc"] = sub.auc_roc.median()
            row[f"{level}_median_p_at_k"] = sub.precision_at_k.median()
        rhos = [spearmanr(sc[v], a0_scores[n])[0] for n, _, sc, _, _, _ in jobs
                if np.std(sc[v]) > 0 and np.std(a0_scores[n]) > 0]
        row["median_rank_corr_with_A0"] = float(np.median(rhos))
        sens.append(row)
    pd.DataFrame(sens).to_csv(OUT / "sensitivity_summary.csv", index=False)

    # Length association (EARLY AUC), FES and baselines
    la = []
    for m in ("A0", "B3", "B4"):
        sub = long[(long.level == "early") & (long.method == m)]
        rho, p = spearmanr(sub.length, sub.auc_roc)
        la.append({"method": m, "n": len(sub), "spearman_rho": rho, "p": p})
    pd.DataFrame(la).to_csv(OUT / "length_association.csv", index=False)

    # Computational cost from the prediction log
    t = pd.read_csv(ROOT / "logs" / "esmfold_timing.tsv", sep="\t")
    t["input_len"] = t["length"].astype(int) + 30
    fit = t[(t.input_len >= 100) & (t.seconds > 0)]  # one negative wall-clock reading (2vil prefix 121)
    b, a = np.polyfit(np.log(fit.input_len), np.log(fit.seconds), 1)
    per_protein = t.groupby("protein").agg(prefixes=("seconds", "size"), total_s=("seconds", "sum"),
                                           rejected=("status", lambda s: int(s.str.startswith("rejected").sum())))
    per_protein["length"] = per_protein.prefixes + 2
    per_protein.to_csv(OUT / "esmfold_cost_per_protein.csv")
    cost = {"total_prefixes": len(t), "total_hours": t.seconds.sum() / 3600,
            "rejected_prefixes": int(t.status.str.startswith("rejected").sum()),
            "chunked_retries": int(t.status.str.endswith("chunk64").sum()),
            "power_law_exponent_input_ge_100": b,
            "median_s_per_prefix_by_input_len": t.groupby(pd.cut(t.input_len, [0, 60, 100, 150, 200, 300, 450]),
                                                          observed=True).seconds.median().round(2).rename(str).to_dict()}
    near100 = per_protein[(per_protein.length >= 90) & (per_protein.length <= 110)]
    cost["proteins_90_110_res_total_min"] = (near100.total_s / 60).round(1).to_dict()
    (OUT / "esmfold_cost.json").write_text(json.dumps(cost, indent=2, default=str))

    print(diag[["max_abs_diff_rescore_vs_published"]].describe().T)
    print(pd.DataFrame(t2).round(3).to_string(index=False))
    print(paired[(paired.level == "early") & paired.metric.isin(["auc_roc", "precision_at_k"])][
        ["metric", "comparator", "n", "median_A0", "median_comparator", "median_paired_diff",
         "wins", "losses", "ties", "wilcoxon_p_greater", "holm_p_greater", "sign_test_p_two_sided",
         "cliffs_delta"]].round(4).to_string(index=False))
    print(pd.DataFrame(cons).round(3).T)
    print(pd.DataFrame(sens).round(3).to_string(index=False))
    print(pd.DataFrame(la).round(4))
    print(json.dumps(cost, indent=1, default=str))


if __name__ == "__main__":
    main()
