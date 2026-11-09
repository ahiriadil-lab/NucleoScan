#!/usr/bin/env python3
"""Evaluate any residue-level score against the NucleoScan benchmark labels.

Input: a CSV with columns name,residue,score (name as in residue_scores_all_methods.csv.gz).
Output: <out>_metrics.csv (per protein and class) and <out>_paired.csv (score minus each method: median
difference, bootstrap CI, wins/losses/ties, two-sided Wilcoxon, Holm within class and metric).

    python evaluate_table.py my_scores.csv --residues residue_scores_all_methods.csv.gz --out my_scores
    python evaluate_table.py --check --residues residue_scores_all_methods.csv.gz --metrics metrics_long.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

LEVELS = ["early", "intermediate", "late", "strong", "medium", "weak"]
METHODS = ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4"]
METRICS = ["auc_roc", "precision_at_k"]
SEED, N_BOOT = 42, 10000


def precision_at_k(scores, labels):
    """P@K with K = number of positives; ties at the cutoff use the expected precision over random orders."""
    k = int(labels.sum())
    cutoff = np.sort(scores)[::-1][k - 1]
    above, tied = scores > cutoff, scores == cutoff
    return (labels[above].sum() + (k - above.sum()) * labels[tied].mean()) / k


def evaluate(table, column):
    """Per-protein metrics for one score column; positions without a finite score are omitted."""
    rows = []
    for name, group in table.groupby("name"):
        for level in LEVELS:
            part = group[np.isfinite(group[column])]
            labels, scores = part[f"label_{level}"].to_numpy(int), part[column].to_numpy(float)
            if len(labels) < 2 or labels.sum() in (0, len(labels)):
                continue
            rows.append({"name": name, "level": level, "method": column, "n_scored": len(labels),
                         "n_protected": int(labels.sum()), "auc_roc": roc_auc_score(labels, scores),
                         "precision_at_k": precision_at_k(scores, labels)})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def paired(long, reference):
    rows = []
    for (level, metric), piv in ((k, long[long.level == k[0]].pivot_table(index="name", columns="method", values=k[1]))
                                 for k in ((lv, m) for lv in LEVELS for m in METRICS)):
        family = []
        for method in [m for m in METHODS if m in piv and m != reference]:
            pair = piv[[reference, method]].dropna()
            d = (pair[reference] - pair[method]).to_numpy()
            if not np.abs(d).max(initial=0) > 1e-12:
                continue
            idx = np.random.default_rng(SEED).integers(0, len(d), size=(N_BOOT, len(d)))
            lo, hi = np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])
            family.append({"level": level, "metric": metric, "comparator": method, "n": len(d),
                           "median_diff": np.median(d), "ci_low": lo, "ci_high": hi, "wins": int((d > 1e-12).sum()),
                           "losses": int((d < -1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
                           "p_two_sided": wilcoxon(pair[reference], pair[method]).pvalue})
        if family:
            for row, p in zip(family, holm([r["p_two_sided"] for r in family])):
                row["holm_p"] = p
            rows += family
    return pd.DataFrame(rows)


def check(table, metrics_path):
    """Every published method must be reproduced from the per-residue table."""
    published = pd.read_csv(metrics_path)
    for method in METHODS:
        merged = published[published.method == method].merge(evaluate(table, method), on=["name", "level", "method"],
                                                             how="outer", suffixes=("", "_new"), indicator=True)
        assert (merged._merge == "both").all(), f"{method}: different protein-class pairs"
        for metric in METRICS:
            assert np.allclose(merged[metric], merged[f"{metric}_new"], atol=1e-9), f"{method}: {metric} differs"
    print(f"OK: {len(METHODS)} methods reproduced from {len(table)} residues")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", nargs="?")
    ap.add_argument("--residues", required=True, help="residue_scores_all_methods.csv.gz from Online Resource 2")
    ap.add_argument("--out", default="evaluated")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--metrics", help="metrics_long.csv, needed with --check")
    args = ap.parse_args()
    table = pd.read_csv(args.residues)
    if args.check:
        return check(table, args.metrics)
    if not args.scores:
        ap.error("a score table is required unless --check is used")
    user = pd.read_csv(args.scores, usecols=["name", "residue", "score"]).rename(columns={"score": "SCORE"})
    if user.duplicated(["name", "residue"]).any():
        raise SystemExit("duplicate name/residue rows in the score table")
    if not set(user.name) & set(table.name):
        raise SystemExit("no protein name in the score table matches the benchmark")
    table = table.merge(user, on=["name", "residue"], how="left")
    table.loc[table["A0"].isna(), "SCORE"] = np.nan  # same evaluable positions as FES
    long = pd.concat([evaluate(table, m) for m in ["SCORE", *METHODS]])
    long[long.method == "SCORE"].to_csv(f"{args.out}_metrics.csv", index=False)
    paired(long, "SCORE").to_csv(f"{args.out}_paired.csv", index=False)
    print(f"wrote {args.out}_metrics.csv and {args.out}_paired.csv")


if __name__ == "__main__":
    main()
