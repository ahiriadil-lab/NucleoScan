"""Matplotlib visualizations for fragment-emergence results."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import RESULTS_DIR, SCORE_THRESHOLD, PDB_ID


def plot_nucleus_scores(residue_df: pd.DataFrame, out_path: str = None):
    """Plot FES from the legacy ``nucleus_score`` output field."""
    if residue_df.empty:
        print("No residue data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [
        "#e74c3c" if row.is_nucleus else "#3498db"
        for _, row in residue_df.iterrows()
    ]
    ax.bar(residue_df["residue"], residue_df["nucleus_score"], color=colors)
    ax.axhline(SCORE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Fragment-emergence score")
    ax.set_title(f"Fragment-emergence candidates — {PDB_ID}")
    ax.set_xticks(residue_df["residue"])
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="High-score candidate"),
        mpatches.Patch(color="#3498db", label="Other residue"),
        plt.Line2D([0], [0], color="black", linestyle="--",
                   label=f"Threshold {SCORE_THRESHOLD}"),
    ])
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "folding_nucleus.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Fragment-emergence bar chart saved to {out_path}")


def plot_rmsd_convergence(fragment_df: pd.DataFrame, out_path: str = None):
    """RMSD across fragment lengths (single prediction per fragment)."""
    if fragment_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    lengths = fragment_df["length"].values
    means   = fragment_df["rmsd_mean"].values

    ax.plot(lengths, means, marker="o", color="#2ecc71", label="RMSD")
    ax.set_xlabel("Fragment length (residues)")
    ax.set_ylabel("RMSD to native (nm)")
    ax.set_title(f"RMSD by Fragment Length — {PDB_ID}")
    ax.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "rmsd_convergence.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RMSD plot saved to {out_path}")


def plot_q_convergence(fragment_df: pd.DataFrame, out_path: str = None):
    """Q-value across fragment lengths (single prediction per fragment)."""
    if fragment_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    lengths = fragment_df["length"].values
    means   = fragment_df["q_mean"].values

    ax.plot(lengths, means, marker="s", color="#9b59b6", label="Q-value")
    ax.set_xlabel("Fragment length (residues)")
    ax.set_ylabel("Q-value (fraction native contacts)")
    ax.set_title(f"Native Contact Q-Value by Fragment Length — {PDB_ID}")
    ax.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "q_convergence.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Q-value plot saved to {out_path}")


def plot_hydro_rg_convergence(fragment_df: pd.DataFrame, out_path: str = None):
    """Hydrophobic Rg across fragment lengths (single prediction per fragment)."""
    if fragment_df.empty:
        return
    if "hydro_rg_mean" not in fragment_df.columns:
        return

    hydro = fragment_df["hydro_rg_mean"].values
    if np.all(np.isnan(hydro)):
        print("hydro_rg_convergence: all NaN — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    lengths = fragment_df["length"].values

    ax.plot(lengths, hydro, marker="^", color="#e67e22", label="Hydrophobic Rg")
    ax.set_xlabel("Fragment length (residues)")
    ax.set_ylabel("Hydrophobic Rg (nm)")
    ax.set_title(f"Hydrophobic Core Radius of Gyration — {PDB_ID}")
    ax.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "hydro_rg_convergence.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Hydrophobic Rg plot saved to {out_path}")


def plot_nucleus_score_distribution(residue_df: pd.DataFrame, out_path: str = None):
    """
    Histogram of FES with thresholded candidate overlay.
    """
    if residue_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    scores  = residue_df["nucleus_score"].values
    is_nuc  = residue_df["is_nucleus"].values

    bins = np.linspace(scores.min(), scores.max(), 20)
    ax.hist(scores, bins=bins, color="#3498db", alpha=0.7, label="All residues")
    ax.hist(scores[is_nuc], bins=bins, color="#e74c3c", alpha=0.8,
            label="High-score candidates")
    ax.axvline(SCORE_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"Threshold {SCORE_THRESHOLD:.2f}")
    ax.set_xlabel("Fragment-emergence score")
    ax.set_ylabel("Number of residues")
    ax.set_title(f"Fragment-emergence Score Distribution — {PDB_ID}")
    ax.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "nucleus_score_distribution.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"FES distribution saved to {out_path}")


def plot_metric_contribution_heatmap(fragment_df: pd.DataFrame, out_path: str = None):
    """
    Heatmap: fragment (Y-axis) × metric (X-axis) showing normalised metric values.
    Reveals which metric dominates scoring for each fragment.
    """
    if fragment_df.empty:
        return

    metric_cols = [c for c in ("norm_rmsd", "norm_q", "norm_hydro_rg")
                   if c in fragment_df.columns]
    if not metric_cols:
        return

    data = fragment_df[metric_cols].values
    lengths = fragment_df["length"].values

    fig, ax = plt.subplots(figsize=(max(5, len(metric_cols) * 2),
                                    max(4, len(lengths) * 0.25 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([c.replace("norm_", "") for c in metric_cols])
    # Y-ticks: show every 5th fragment to avoid crowding
    step = max(1, len(lengths) // 20)
    ax.set_yticks(range(0, len(lengths), step))
    ax.set_yticklabels(lengths[::step])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Fragment length (residues)")
    ax.set_title(f"Metric Contribution Heatmap — {PDB_ID}")
    plt.colorbar(im, ax=ax, label="Normalised value [0,1]")
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "metric_contribution_heatmap.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Metric contribution heatmap saved to {out_path}")


def plot_composite_summary(residue_df: pd.DataFrame, fragment_df: pd.DataFrame,
                           importance_df: pd.DataFrame = None,
                           out_path: str = None):
    """
    2×2 summary figure:
      [0,0] Fragment-emergence score bar chart
      [0,1] RMSD convergence
      [1,0] Structural importance (if available) or Q convergence
      [1,1] Metric contribution heatmap
    """
    if residue_df.empty and fragment_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # [0,0] FES bar chart
    ax = axes[0, 0]
    if not residue_df.empty:
        colors = ["#e74c3c" if n else "#3498db"
                  for n in residue_df["is_nucleus"]]
        ax.bar(residue_df["residue"], residue_df["nucleus_score"], color=colors)
        ax.axhline(SCORE_THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_title("Fragment-emergence Scores")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Score")

    # [0,1] RMSD by fragment length
    ax = axes[0, 1]
    if not fragment_df.empty and "rmsd_mean" in fragment_df.columns:
        lengths = fragment_df["length"].values
        ax.plot(lengths, fragment_df["rmsd_mean"].values,
                marker="o", color="#2ecc71", label="RMSD")
        ax.set_title("RMSD by Fragment Length")
        ax.set_xlabel("Fragment length")
        ax.set_ylabel("RMSD (nm)")

    # [1,0] Importance or Q by fragment length
    ax = axes[1, 0]
    if importance_df is not None and not importance_df.empty and "importance" in importance_df.columns:
        ax.bar(importance_df["residue"], importance_df["importance"],
               color="#9b59b6", alpha=0.8)
        ax.set_title("Structural Importance")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Importance")
    elif not fragment_df.empty and "q_mean" in fragment_df.columns:
        lengths = fragment_df["length"].values
        ax.plot(lengths, fragment_df["q_mean"].values,
                marker="s", color="#9b59b6", label="Q-value")
        ax.set_title("Q-Value by Fragment Length")
        ax.set_xlabel("Fragment length")
        ax.set_ylabel("Q-value")

    # [1,1] Metric heatmap (inline)
    ax = axes[1, 1]
    metric_cols = [c for c in ("norm_rmsd", "norm_q", "norm_hydro_rg")
                   if c in fragment_df.columns]
    if not fragment_df.empty and metric_cols:
        data = fragment_df[metric_cols].values
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels([c.replace("norm_", "") for c in metric_cols])
        ax.set_title("Metric Contributions")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Fragment index")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"FoldNucleus Summary — {PDB_ID}", fontsize=13)
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "composite_summary.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Composite summary saved to {out_path}")


def plot_construction_dashboard(structures_dir: str,
                                ref_pdb: str,
                                fragment_df: pd.DataFrame,
                                out_path: str = None,
                                n_panels: int = 8) -> None:
    """
    2-row multi-panel construction dashboard.

    Row 1 — For each sampled fragment length: 2D PCA projection of Cα coordinates
             comparing the best decoy (coloured) vs the native (grey).
    Row 2 — RMSD and Q-value of the best decoy at each sampled step (markers
             on top of the full convergence curve).

    Parameters
    ----------
    structures_dir : directory containing frag{length}/ subdirectories
    ref_pdb        : native PDB path
    fragment_df    : DataFrame from compute_fragment_scores (for RMSD/Q curves)
    out_path       : output PNG (default: RESULTS_DIR/construction_dashboard.png)
    n_panels       : number of fragment lengths to sample (default 8)
    """
    import warnings as _w
    import mdtraj as md
    try:
        from sklearn.decomposition import PCA
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    from modules.trajectory import select_best_decoy

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "construction_dashboard.png")

    # ---- Load native ----
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        ref_traj = md.load(ref_pdb)

    ref_ca_idx = ref_traj.topology.select("name CA")
    ref_ca     = ref_traj.xyz[0, ref_ca_idx, :]  # (N, 3) in nm

    # ---- Sample fragment lengths ----
    import glob as _glob
    frag_dirs = sorted(_glob.glob(os.path.join(structures_dir, "frag[0-9]*")))
    if not frag_dirs:
        print("  [dashboard] No fragment directories found — skipping.")
        return

    step = max(1, len(frag_dirs) // n_panels)
    sampled_dirs = frag_dirs[::step][:n_panels]
    n_col = len(sampled_dirs)
    if n_col == 0:
        return

    # ---- Fit PCA on native Cα coordinates ----
    if _has_sklearn and len(ref_ca) >= 3:
        pca = PCA(n_components=2)
        pca.fit(ref_ca)
        ref_2d = pca.transform(ref_ca)
    else:
        # Fallback: use x, y coordinates directly
        ref_2d = ref_ca[:, :2]
        pca    = None

    # ---- Build figure ----
    fig, axes = plt.subplots(2, n_col, figsize=(3 * n_col, 7))
    if n_col == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing

    rmsd_col = "rmsd_mean" if "rmsd_mean" in fragment_df.columns else None
    q_col    = ("q_local_mean" if "q_local_mean" in fragment_df.columns
                else "q_mean" if "q_mean" in fragment_df.columns else None)

    for col_idx, frag_dir in enumerate(sampled_dirs):
        frag_name = os.path.basename(frag_dir)
        # try to parse length from directory name (frag003 → 3)
        try:
            frag_len = int("".join(c for c in frag_name if c.isdigit()))
        except ValueError:
            frag_len = col_idx + 1

        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # ---- Row 1: 2D PCA of Cα ----
        best_path = select_best_decoy(frag_dir, ref_traj)
        if best_path is not None:
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    dec_traj = md.load(best_path)
                dec_ca_idx = dec_traj.topology.select("name CA")
                dec_ca     = dec_traj.xyz[0, dec_ca_idx, :]  # (M, 3)

                # Align decoy to native
                n_common = min(len(ref_ca_idx), len(dec_ca_idx))
                sub_dec = dec_traj.atom_slice(dec_ca_idx[:n_common])
                sub_ref = ref_traj.atom_slice(ref_ca_idx[:n_common])
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    sub_dec.superpose(sub_ref)
                dec_ca_aligned = sub_dec.xyz[0]  # (n_common, 3)

                # Project
                n_proj = min(len(dec_ca_aligned), len(ref_ca))
                if pca is not None:
                    ref_plot = pca.transform(ref_ca[:n_proj])
                    dec_plot = pca.transform(dec_ca_aligned[:n_proj])
                else:
                    ref_plot = ref_ca[:n_proj, :2]
                    dec_plot = dec_ca_aligned[:n_proj, :2]

                ax_top.scatter(ref_plot[:, 0], ref_plot[:, 1],
                               c="grey", s=30, alpha=0.6, label="Native", zorder=2)
                cmap = plt.cm.plasma
                colors = cmap(np.linspace(0, 1, n_proj))
                ax_top.scatter(dec_plot[:, 0], dec_plot[:, 1],
                               c=colors, s=25, alpha=0.8, zorder=3)
            except Exception as e:
                ax_top.text(0.5, 0.5, f"Error\n{e}", transform=ax_top.transAxes,
                            ha="center", va="center", fontsize=6)
        else:
            ax_top.text(0.5, 0.5, "No decoy", transform=ax_top.transAxes,
                        ha="center", va="center", fontsize=8, color="grey")

        ax_top.set_title(frag_name, fontsize=8)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        if col_idx == 0:
            ax_top.set_ylabel("PC2", fontsize=7)

        # ---- Row 2: RMSD / Q curve with marker at this fragment ----
        if not fragment_df.empty:
            lengths = fragment_df["length"].values
            if rmsd_col:
                ax_bot.plot(lengths, fragment_df[rmsd_col].values,
                            color="#2ecc71", alpha=0.5, linewidth=1)
            if q_col:
                ax2 = ax_bot.twinx()
                ax2.plot(lengths, fragment_df[q_col].values,
                         color="#9b59b6", alpha=0.5, linewidth=1)
                ax2.set_yticks([])

            # Marker at current fragment length
            row = fragment_df[fragment_df["length"] == frag_len]
            if not row.empty:
                if rmsd_col:
                    ax_bot.scatter([frag_len], row[rmsd_col].values,
                                   color="#e74c3c", zorder=5, s=40)

        ax_bot.set_xlabel("Frag len", fontsize=7)
        if col_idx == 0 and rmsd_col:
            ax_bot.set_ylabel("RMSD (nm)", fontsize=7)
        ax_bot.tick_params(labelsize=6)

    fig.suptitle(f"Prefix-length Structure Series — {PDB_ID}", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [dashboard] Construction dashboard saved to {out_path}")


def plot_contact_formation_heatmap(profile: np.ndarray,
                                   lengths: list,
                                   residue_df: pd.DataFrame = None,
                                   phi_values: dict = None,
                                   out_path: str = None):
    """
    Heatmap of per-residue contact formation q_residue(i, L).

    X-axis: fragment lengths.
    Y-axis: residue indices.
    Colour: q_residue(i, L) — fraction of residue i's native contacts formed.

    Overlays:
      - Horizontal markers for thresholded FES candidates (red dashes).
      - Vertical markers for contact onset points (stars on diagonal).
      - If phi_values provided, residue labels coloured by Phi magnitude.

    Parameters
    ----------
    profile    : (n_res, n_lengths) array from build_contact_formation_profile
    lengths    : list[int] — matching length values for columns of profile
    residue_df : DataFrame with 'residue', 'is_nucleus' columns (optional)
    phi_values : dict {residue_1indexed: phi_value} (optional)
    out_path   : output file (default: RESULTS_DIR/contact_formation_heatmap.png)
    """
    if profile.size == 0 or len(lengths) == 0:
        print("contact_formation_heatmap: no profile data — skipping.")
        return

    n_res, n_len = profile.shape

    fig, ax = plt.subplots(figsize=(max(8, n_len * 0.5), max(6, n_res * 0.25)))

    im = ax.imshow(profile, aspect="auto", cmap="RdYlBu_r",
                   vmin=0.0, vmax=1.0, origin="lower",
                   extent=[lengths[0] - 0.5, lengths[-1] + 0.5,
                           0.5, n_res + 0.5])

    plt.colorbar(im, ax=ax, label="q_residue (fraction native contacts formed)")

    # Mark thresholded FES candidates (red horizontal lines)
    if residue_df is not None and not residue_df.empty and "is_nucleus" in residue_df.columns:
        nucleus_res = residue_df[residue_df["is_nucleus"]]["residue"].values
        for r in nucleus_res:
            ax.axhline(r, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7)

    # Mark contact onset points (star where q first exceeds 0.3)
    from config import CONTACT_ONSET_THRESHOLD
    for res_idx in range(n_res):
        for col_idx, L in enumerate(lengths):
            if profile[res_idx, col_idx] > CONTACT_ONSET_THRESHOLD:
                ax.scatter(L, res_idx + 1, marker="*", color="gold",
                           s=30, zorder=5, linewidths=0.3, edgecolors="black")
                break

    ax.set_xlabel("Fragment length (residues)")
    ax.set_ylabel("Residue index")
    ax.set_title(f"Contact Formation Profile — {PDB_ID} (native-reference diagnostic)\n"
                 f"(gold ★ = contact onset, red dashes = FES candidates)")

    # Y-tick labels: colour by Phi if provided
    ytick_positions = list(range(1, n_res + 1))
    ax.set_yticks(ytick_positions)
    if phi_values:
        labels = []
        for r in ytick_positions:
            phi = phi_values.get(r, None)
            labels.append(f"{r}" if phi is None else f"{r} (Φ={phi:.2f})")
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_yticklabels([str(r) for r in ytick_positions], fontsize=7)

    ax.set_xticks(lengths)
    ax.set_xticklabels([str(l) for l in lengths], fontsize=7, rotation=45)

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "contact_formation_heatmap.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Contact formation heatmap saved to {out_path}")



def save_results(fragment_df: pd.DataFrame,
                 residue_df: pd.DataFrame,
                 out_dir: str = None):
    """Save CSVs of fragment metrics and residue scores."""
    if out_dir is None:
        out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not fragment_df.empty:
        frag_path = os.path.join(out_dir, "fragment_metrics.csv")
        fragment_df.to_csv(frag_path, index=False)
        print(f"Fragment metrics saved to {frag_path}")

    if not residue_df.empty:
        res_path = os.path.join(out_dir, "residue_scores.csv")
        residue_df.to_csv(res_path, index=False)
        print(f"Residue scores saved to {res_path}")
