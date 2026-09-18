"""
Sensitivity analysis for fragment-emergence candidate thresholds.

Sweeps SCORE_THRESHOLD from 1.0 to 2.5 (step 0.1) and shows how the number
and identity of thresholded candidates change with the parameter.

Outputs
-------
results/sensitivity_heatmap.png  — heatmap: residue (Y) × threshold (X)
results/sensitivity_nb_nucleus.png — curve: nb_nucleus vs threshold
results/sensitivity_results.csv  — raw sweep data

Usage
-----
    from sensitivity import run_sensitivity_analysis
    run_sensitivity_analysis(fragment_df, residue_df, out_dir="results/")
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.score import score_residues


def run_sensitivity_analysis(fragment_df: pd.DataFrame,
                              residue_df: pd.DataFrame,
                              out_dir: str = "results/",
                              threshold_min: float = 1.0,
                              threshold_max: float = 2.5,
                              threshold_step: float = 0.1,
                              method: str = None) -> pd.DataFrame:
    """
    Sweep the appropriate parameter based on scoring method:
    - For 'marginal': sweep MARGINAL_SIGMA_FACTOR (0.0 to 3.0)
    - For others: sweep SCORE_THRESHOLD (1.0 to 2.5)

    Parameters
    ----------
    fragment_df    : DataFrame from compute_fragment_scores
    residue_df     : baseline residue scores (used for residue list)
    out_dir        : directory for output files
    threshold_min  : lower bound of sweep (for non-marginal methods)
    threshold_max  : upper bound of sweep (for non-marginal methods)
    threshold_step : increment (for non-marginal methods)
    method         : scoring method to use (defaults to config.SCORING_METHOD)

    Returns
    -------
    pd.DataFrame with columns: param_value, n_nucleus, nucleus_fraction, nucleus_residues
    """
    import config as _cfg

    if fragment_df.empty:
        print("sensitivity: fragment_df is empty — skipping.")
        return pd.DataFrame()

    if method is None:
        method = _cfg.SCORING_METHOD

    max_len = int(fragment_df["length"].max())
    n_res   = max_len

    # Determine which parameter to sweep based on scoring method
    is_marginal = (method == "marginal")

    if is_marginal:
        param_values = np.arange(0.0, 3.0 + 0.125, 0.25)  # 0.0 to 3.0, step 0.25
        param_name = "MARGINAL_SIGMA_FACTOR"
        original_val = _cfg.MARGINAL_SIGMA_FACTOR
        param_label = "MARGINAL_SIGMA_FACTOR"
    else:
        param_values = np.arange(threshold_min, threshold_max + threshold_step / 2,
                                threshold_step)
        param_values = np.round(param_values, 3)
        param_name = "SCORE_THRESHOLD"
        original_val = _cfg.SCORE_THRESHOLD
        param_label = "SCORE_THRESHOLD"

    sweep_rows  = []
    # One indicates a selected candidate.
    nucleus_matrix = np.zeros((n_res, len(param_values)), dtype=np.uint8)

    for v_idx, param_val in enumerate(param_values):
        setattr(_cfg, param_name, float(param_val))
        rdf = score_residues(fragment_df, max_length=max_len, method=method)
        if rdf.empty:
            continue
        is_nuc = rdf["is_nucleus"].values
        n_nuc  = int(is_nuc.sum())
        nucleus_matrix[:len(is_nuc), v_idx] = is_nuc.astype(np.uint8)
        nuc_res = rdf[rdf["is_nucleus"]]["residue"].tolist()
        sweep_rows.append({
            param_label:        float(param_val),
            "n_nucleus":        n_nuc,
            "nucleus_fraction": n_nuc / max(len(rdf), 1),
            "nucleus_residues": str(nuc_res),
        })

    # Restore original value
    setattr(_cfg, param_name, original_val)

    sweep_df = pd.DataFrame(sweep_rows)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_results.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"Sensitivity sweep saved to {csv_path}")

    # Candidate heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(param_values) * 0.5),
                                    max(4, n_res * 0.15 + 1)))
    im = ax.imshow(nucleus_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in param_values], rotation=90, fontsize=7)
    step = max(1, n_res // 20)
    ax.set_yticks(range(0, n_res, step))
    ax.set_yticklabels(range(1, n_res + 1, step), fontsize=7)
    ax.set_xlabel(param_label)
    ax.set_ylabel("Residue position")
    ax.set_title(f"Candidate membership vs {param_label} — method={method}")
    plt.colorbar(im, ax=ax, label="Selected candidate (1=yes)")
    plt.tight_layout()
    hm_path = os.path.join(out_dir, "sensitivity_heatmap.png")
    plt.savefig(hm_path, dpi=150)
    plt.close(fig)
    print(f"Sensitivity heatmap saved to {hm_path}")

    # Candidate count by parameter
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df[param_label], sweep_df["n_nucleus"],
            marker="o", color="#e74c3c")
    ax.set_xlabel(param_label)
    ax.set_ylabel("Number of selected candidates")
    ax.set_title(f"Candidate-set size sensitivity to {param_label}")
    ax.axhline(sweep_df["n_nucleus"].median(), color="gray",
               linestyle="--", linewidth=0.8, label="Median")
    ax.legend()
    plt.tight_layout()
    nb_path = os.path.join(out_dir, "sensitivity_nb_nucleus.png")
    plt.savefig(nb_path, dpi=150)
    plt.close(fig)
    print(f"Candidate-count sensitivity curve saved to {nb_path}")

    return sweep_df
