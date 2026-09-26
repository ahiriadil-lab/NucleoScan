"""
Module 4 — Structural Importance.

Assigns a composite structural importance score to each residue, combining
Three metrics derived from the reference structure:

  1. Contact degree    — number of contacts above threshold in native contact map
  2. SS persistence    — secondary structure stability (from native DSSP)
  3. Betweenness      — graph-theoretic centrality in the contact network

RMSF (requires ensemble fluctuations) and GNM (Gaussian Network Model, requires
dynamic ensemble) have been removed — both are meaningless with a single structure.

Outputs
-------
results/structural_importance.csv
results/importance_vs_nucleus.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md

from config import (
    RESULTS_DIR,
    IMPORTANCE_WEIGHTS, IMPORTANCE_CONTACT_THRESHOLD,
)
from core.score import normalize_vector


# Sub-metrics

def compute_contact_degree(contact_prob_map, threshold=IMPORTANCE_CONTACT_THRESHOLD):
    """
    Per-residue contact degree from a contact map.

    Returns
    -------
    degree : (n_res,) int array
    """
    return (contact_prob_map >= threshold).sum(axis=1)


def compute_ss_persistence(traj):
    """
    Secondary structure persistence from a single structure.

    For a single structure, DSSP returns the SS class per residue.
    We convert to a binary 'has_secondary_structure' signal:
      H (helix) or E (sheet) → 1.0; C (coil) → 0.0.

    Returns
    -------
    ss_binary : (n_res,) float array in {0.0, 1.0}
    """
    n_res = traj.topology.n_residues
    ss_binary = np.zeros(n_res, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dssp = md.compute_dssp(traj, simplified=True)  # (n_frames, n_res)
        # Use first frame
        for i, ss in enumerate(dssp[0]):
            if ss in ("H", "E"):
                ss_binary[i] = 1.0
    except Exception:
        pass
    return ss_binary


def compute_betweenness(contact_map, threshold=IMPORTANCE_CONTACT_THRESHOLD):
    """
    Node betweenness centrality in the contact graph.

    Nodes  = residues
    Edges  = pairs with contact value >= threshold
    (Unweighted for binary contact maps)

    Returns
    -------
    between : (n_res,) float array
    """
    import networkx as nx

    n_res = contact_map.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_res))
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if contact_map[i, j] >= threshold:
                G.add_edge(i, j)

    bc = nx.betweenness_centrality(G, normalized=True)
    return np.array([bc.get(i, 0.0) for i in range(n_res)])


# Composite importance (3 metrics)

def compute_importance(degree, ss_persist, between, weights=None):
    """
    Combine 3 sub-metrics into a composite importance score.

    Each metric is min-max normalised to [0, 1].

    Parameters
    ----------
    degree, ss_persist, between : (n_res,) float arrays
    weights : tuple of 3 floats summing to 1 (default: IMPORTANCE_WEIGHTS).

    Returns
    -------
    importance : (n_res,) float array in [0, 1]
    """
    if weights is None:
        weights = IMPORTANCE_WEIGHTS

    w_deg, w_ss, w_bt = weights[0], weights[1], weights[2]

    norm_deg = normalize_vector(degree.astype(float))
    norm_ss  = normalize_vector(ss_persist.astype(float))
    norm_bt  = normalize_vector(between.astype(float))

    composite = w_deg * norm_deg + w_ss * norm_ss + w_bt * norm_bt
    return normalize_vector(composite)


# Correlation with FES

def correlate_with_nucleus(importance_df):
    """
    Pearson and Spearman correlation between importance and FES
    (``nucleus_score`` column).
    """
    from scipy import stats

    df = importance_df.dropna(subset=["importance", "nucleus_score"])
    if len(df) < 3:
        return {}

    imp = df["importance"].values
    ns  = df["nucleus_score"].values
    pr, pp = stats.pearsonr(imp, ns)
    sr, sp = stats.spearmanr(imp, ns)
    return {
        "pearson_r":  float(pr),
        "pearson_p":  float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# Plotting

def plot_importance_vs_nucleus(importance_df, out_path=None):
    """Scatter of importance versus FES."""
    if importance_df.empty or "nucleus_score" not in importance_df.columns:
        return
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "importance_vs_nucleus.png")

    df = importance_df.dropna(subset=["importance", "nucleus_score"])
    if df.empty:
        return

    corr = correlate_with_nucleus(importance_df)

    fig, ax = plt.subplots(figsize=(6, 5))
    nucleus_color = df["is_nucleus"].map({True: "#e74c3c", False: "#3498db"}) if "is_nucleus" in df.columns else "#3498db"
    ax.scatter(df["nucleus_score"], df["importance"],
               c=nucleus_color, s=30, alpha=0.7)
    ax.set_xlabel("Fragment-emergence score")
    ax.set_ylabel("Structural importance")
    ax.set_title("Structural Importance vs Fragment-emergence Score")

    if corr:
        ax.text(0.05, 0.92,
                f"Pearson r = {corr['pearson_r']:.3f}  (p={corr['pearson_p']:.2e})\n"
                f"Spearman r = {corr['spearman_r']:.3f}  (p={corr['spearman_p']:.2e})",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Importance vs FES scatter saved to {out_path}")


# Top-level entry: compute from native PDB (no decoys required)

def compute_importance_from_native(ref_pdb_path: str) -> pd.DataFrame:
    """
    Compute structural importance directly from the native PDB (no decoys needed).

    Derives 3 metrics from the native structure:
      - Contact degree: from binary native contact map
      - SS persistence: from DSSP on the native structure
      - Betweenness: graph centrality from native contact map

    Parameters
    ----------
    ref_pdb_path : path to the native/reference PDB

    Returns
    -------
    DataFrame with columns: residue (1-indexed), importance
    """
    from core.analyze import compute_native_contacts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_traj = md.load(ref_pdb_path)

    n_res = ref_traj.topology.n_residues

    # Native contact map (binary)
    native_pairs, _ = compute_native_contacts(ref_traj)
    contact_map = np.zeros((n_res, n_res), dtype=float)
    for i, j in native_pairs:
        if i < n_res and j < n_res:
            contact_map[i, j] = 1.0
            contact_map[j, i] = 1.0

    degree  = compute_contact_degree(contact_map, threshold=0.5)
    ss_pers = compute_ss_persistence(ref_traj)
    between = compute_betweenness(contact_map, threshold=0.5)

    composite = compute_importance(degree.astype(float), ss_pers, between.astype(float))

    return pd.DataFrame({
        "residue":    np.arange(1, n_res + 1),
        "importance": composite,
    })


def run_importance_analysis(ref_traj, cmaps, residue_df=None,
                            out_dir=None, lengths=None, pdb_path=None):
    """
    Structural importance analysis from native structure.

    Uses compute_importance_from_native() which requires only the reference PDB.
    The cmaps argument is unused; the contact map is derived directly from the
    native structure.

    Returns importance_df.
    """
    if out_dir is None:
        out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    from config import REFERENCE_PDB
    ref_pdb = pdb_path or REFERENCE_PDB

    if not os.path.exists(ref_pdb):
        print("  [Importance] Reference PDB not found — skipping.")
        return pd.DataFrame()

    print("  [Importance] Computing structural importance from native structure …")
    importance_df = compute_importance_from_native(ref_pdb)

    if importance_df.empty:
        print("  [Importance] No data — skipping.")
        return importance_df

    if residue_df is not None and not residue_df.empty:
        importance_df = importance_df.merge(
            residue_df[["residue", "nucleus_score", "is_nucleus"]],
            on="residue", how="left",
        )

    csv_path = os.path.join(out_dir, "structural_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"  [Importance] CSV saved to {csv_path}")

    plot_importance_vs_nucleus(importance_df,
                               out_path=os.path.join(out_dir, "importance_vs_nucleus.png"))

    if "nucleus_score" in importance_df.columns:
        corr = correlate_with_nucleus(importance_df)
        if corr:
            print(f"  [Importance] Pearson r={corr['pearson_r']:.3f}, "
                  f"Spearman r={corr['spearman_r']:.3f}")

    print("  [Importance] Done.")
    return importance_df
