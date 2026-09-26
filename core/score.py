"""
Per-residue structural-emergence scoring.

Each residue i is scored by a pure sliding-window signal:
  nucleus_score(i) = mean composite quality score over all fragments
                     of length >= i (i.e., all fragments that contain residue i).

The composite quality score per fragment is derived from three structural
metrics — backbone RMSD, Q-value (fraction of native contacts), and
hydrophobic core radius of gyration — each min-max normalised across all
fragment lengths before combination (each in [0,1], total in [0,3]).

The per-residue ``nucleus_score`` is then normalized to [0,1] and
rescaled to [0,3] to match SCORE_THRESHOLD. It is interpreted as a
fragment-emergence score; the thresholded ``is_nucleus`` field is an
exploratory candidate flag, not a calibrated biological classification.
"""
import numpy as np
import pandas as pd

from config import (
    MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH,
    SCORE_THRESHOLD, SCORING_METHOD, SCORE_THRESHOLD_MARGINAL,
    MARGINAL_SIGMA_FACTOR, Q_WEIGHT, MIN_Q_SIGNAL,
    adaptive_sigma_factor,
    SCORING_COMBINED_W_MARGINAL, SCORING_COMBINED_W_IMPORTANCE,
    CONTACT_ONSET_THRESHOLD, EMERGENT_CONTACT_DEGREE_ONSET,
    LOCAL_PACKING_WEIGHT, HYDRO_BURIAL_WEIGHT,
    PLDDT_WEIGHT, PLDDT_GRADIENT_WEIGHT,
    CONTACT_RATE_WEIGHT,
    PLDDT_REJECT_THRESHOLD, PLDDT_REDUCED_WEIGHT_THRESHOLD, PLDDT_REDUCED_WEIGHT_FACTOR,
    PLDDT_BETA_REJECT_THRESHOLD, PLDDT_BETA_REDUCED_THRESHOLD,
    PLDDT_ALPHA_REJECT_THRESHOLD, PLDDT_ALPHA_REDUCED_THRESHOLD,
    PLDDT_COIL_REJECT_THRESHOLD, PLDDT_COIL_REDUCED_THRESHOLD,
    FDR_TARGET,
    UNIFIED_W_KINETIC, UNIFIED_W_COOPERATIVE, UNIFIED_W_TOPOLOGICAL,
    UNIFIED_COOP_W_TSE, UNIFIED_COOP_W_LR, UNIFIED_COOP_W_COOP,
    TSE_MIN_FRAMES, TSE_MIN_FRAMES_PARTIAL,
)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array to [0, 1].
    If all values are equal, returns array of 0.5.
    """
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi == lo:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def _effective_q_weight(fragment_length: int, n_residues: int) -> float:
    """Return Q_WEIGHT (constant — Q_LOCAL_DAMPEN removed)."""
    return Q_WEIGHT


def compute_fragment_scores(metrics: list) -> pd.DataFrame:
    """
    Given a list of per-fragment metric dicts (from analyze_structures),
    compute a composite quality score for each fragment.

    Score = Q_WEIGHT * norm_q  +  (1 - norm_rmsd)  +  (1 - norm_hydro_rg)
    norm_q is derived from q_local_mean (contacts formed / contacts available in
    fragment window) when present, falling back to q_mean (q_global) otherwise.
    q_local is length-independent, avoiding the monotone length-bias of q_global.

    Returns a DataFrame with columns:
        length, rmsd_mean, q_mean, hydro_rg_mean, score, rmsd_std, q_std, hydro_rg_std,
        norm_rmsd, norm_q, norm_hydro_rg
    """
    if not metrics:
        return pd.DataFrame()

    df = pd.DataFrame(metrics)
    lengths = df["length"].values.astype(int)

    norm_rmsd = normalize_vector(df["rmsd_mean"].values)
    # Local Q avoids the fragment-length bias of global Q.
    q_col = "q_local_mean" if "q_local_mean" in df.columns else "q_mean"
    norm_q = normalize_vector(df[q_col].values)

    hydro = df["hydro_rg_mean"].values.astype(float)
    if np.all(np.isnan(hydro)):
        norm_hydro = np.full(len(hydro), 0.5)
    else:
        norm_hydro = normalize_vector(np.where(np.isnan(hydro),
                                                np.nanmean(hydro), hydro))

    # Invert metrics where lower values are better.
    n_res = int(lengths.max())  # total protein length (proxy)
    q_weights = np.array([_effective_q_weight(int(l), n_res) for l in lengths])
    df["score"] = q_weights * norm_q + (1.0 - norm_rmsd) + (1.0 - norm_hydro)
    df["norm_rmsd"] = norm_rmsd
    df["norm_q"] = norm_q
    df["norm_hydro_rg"] = norm_hydro

    # Apply secondary-structure-aware thresholds only to fragments ≥ 8 residues.
    _MIN_SS_AWARE_LEN = 8
    if "plddt_mean" in df.columns:
        plddt = df["plddt_mean"].values.astype(float)
        lengths_arr = df["length"].values.astype(int) if "length" in df.columns else np.full(len(plddt), 20)
        if "ss_dominant" in df.columns:
            ss = df["ss_dominant"].values
            reject_thr = np.where(
                lengths_arr < _MIN_SS_AWARE_LEN, PLDDT_REJECT_THRESHOLD,
                np.where(ss == "E", PLDDT_BETA_REJECT_THRESHOLD,
                np.where(ss == "H", PLDDT_ALPHA_REJECT_THRESHOLD,
                         PLDDT_COIL_REJECT_THRESHOLD)))
            reduced_thr = np.where(
                lengths_arr < _MIN_SS_AWARE_LEN, PLDDT_REDUCED_WEIGHT_THRESHOLD,
                np.where(ss == "E", PLDDT_BETA_REDUCED_THRESHOLD,
                np.where(ss == "H", PLDDT_ALPHA_REDUCED_THRESHOLD,
                         PLDDT_COIL_REDUCED_THRESHOLD)))
        else:
            reject_thr = np.full(len(plddt), PLDDT_REJECT_THRESHOLD)
            reduced_thr = np.full(len(plddt), PLDDT_REDUCED_WEIGHT_THRESHOLD)
        plddt_weight = np.where(
            plddt < reject_thr, 0.0,
            np.where(plddt < reduced_thr, PLDDT_REDUCED_WEIGHT_FACTOR, 1.0))
        df["plddt_confidence_weight"] = plddt_weight
        df["score"] *= plddt_weight

    if PLDDT_WEIGHT > 0 and "plddt_mean" in df.columns:
        plddt_vals = df["plddt_mean"].values.astype(float)
        if not np.all(np.isnan(plddt_vals)):
            norm_plddt = normalize_vector(np.where(np.isnan(plddt_vals),
                                                    np.nanmean(plddt_vals), plddt_vals))
            df["score"] += PLDDT_WEIGHT * norm_plddt
            df["norm_plddt"] = norm_plddt

    if LOCAL_PACKING_WEIGHT > 0 and "packing_mean" in df.columns:
        packing_vals = df["packing_mean"].values.astype(float)
        if np.any(packing_vals > 0):
            norm_packing = normalize_vector(packing_vals)
            df["score"] += LOCAL_PACKING_WEIGHT * norm_packing
            df["norm_packing"] = norm_packing

    if HYDRO_BURIAL_WEIGHT > 0 and "hydro_burial_mean" in df.columns:
        burial_vals = df["hydro_burial_mean"].values.astype(float)
        if np.any(burial_vals > 0):
            norm_burial = normalize_vector(burial_vals)
            df["score"] += HYDRO_BURIAL_WEIGHT * norm_burial
            df["norm_hydro_burial"] = norm_burial

    return df


def score_residues_sliding_window(fragment_df: pd.DataFrame,
                                  max_length: int) -> pd.DataFrame:
    """
    Sliding-window FES stored in the ``nucleus_score`` column.
    It is normalized to [0,1] then rescaled to [0,3].
    """
    fdf = fragment_df.set_index("length").sort_index()
    scores = fdf["score"]

    records = []
    for res_i in range(1, max_length + 1):
        containing = scores[scores.index >= res_i]
        nucleus_score = float(containing.mean()) if len(containing) > 0 else 0.0
        records.append({"residue": res_i, "nucleus_score": nucleus_score})

    res_df = pd.DataFrame(records)
    ns_norm = normalize_vector(res_df["nucleus_score"].values) * 3.0
    res_df["nucleus_score"] = ns_norm
    res_df["is_nucleus"] = res_df["nucleus_score"] >= SCORE_THRESHOLD
    return res_df


def score_residues_marginal(fragment_df: pd.DataFrame,
                            max_length: int,
                            plddt_gradient: np.ndarray = None,
                            contact_formation_rate: np.ndarray = None) -> pd.DataFrame:
    """
    Marginal (delta) scoring: delta(i) = score(fragment_i) - score(fragment_{i-1}).

    delta > 0: adding residue i is associated with improved fragment score.
    delta < 0: adding residue i is associated with a reduced fragment score.

    Large positive deltas identify fragment-emergence candidates. Because
    fragment structures are predicted independently, delta is not a time step
    or direct evidence of folding-nucleus membership.
    The delta is normalised to [0, 1] then rescaled to [0, 3].
    """
    fdf = fragment_df.set_index("length").sort_index()
    scores = fdf["score"]

    lengths = sorted(set(scores.index.tolist()))
    def _to_scalar(v):
        return float(v.mean()) if hasattr(v, "mean") else float(v)
    score_by_len = {l: _to_scalar(scores.loc[l]) for l in lengths}

    # Local Q determines where meaningful marginal scoring begins.
    q_signal_col = "q_local_mean" if "q_local_mean" in fdf.columns else "q_mean"
    q_by_len = fdf[q_signal_col] if q_signal_col in fdf.columns else None
    if q_by_len is not None:
        def _q_scalar(l):
            v = q_by_len.get(l, 0)
            return float(v.mean()) if hasattr(v, "mean") else float(v)
        first_q_lengths = [l for l in lengths if _q_scalar(l) > MIN_Q_SIGNAL]
        if first_q_lengths:
            effective_min = first_q_lengths[0] - 1  # last length with q=0
            min_len = max(lengths[0], effective_min)
        else:
            # Preserve RMSD/Rg information when local Q is uniformly weak.
            min_len = lengths[0]  # first fragment sets the reference baseline
    else:
        min_len = lengths[0]  # fallback: no q_mean column

    prev_score = score_by_len.get(min_len, score_by_len[lengths[0]])
    if prev_score != prev_score:  # NaN check
        valid_scores = [(l, s) for l, s in score_by_len.items()
                        if s == s and l >= min_len]  # s==s is NaN-safe
        if valid_scores:
            prev_score = min(valid_scores, key=lambda x: x[0])[1]
        else:
            prev_score = 0.0

    records = []
    for res_i in range(1, min_len + 1):
        records.append({"residue": res_i, "nucleus_score": 0.0})

    for res_i in range(min_len + 1, max_length + 1):
        if res_i in score_by_len:
            cur_score = score_by_len[res_i]
            if cur_score != cur_score:  # NaN check
                records.append({"residue": res_i, "nucleus_score": 0.0})
                continue
            delta = cur_score - prev_score
            prev_score = cur_score
        else:
            available = [l for l in lengths if l >= res_i and
                         score_by_len[l] == score_by_len[l]]  # skip NaN
            if available:
                cur_score = score_by_len[available[0]]
                delta = cur_score - prev_score
                prev_score = cur_score
            else:
                delta = 0.0
        records.append({"residue": res_i, "nucleus_score": delta})

    res_df = pd.DataFrame(records)
    marginal_mask = res_df["residue"] > min_len
    raw = res_df["nucleus_score"].values.copy()
    if marginal_mask.any():
        marginal_raw = raw[marginal_mask]
        ns_norm_marginal = normalize_vector(marginal_raw) * 3.0
        raw[marginal_mask] = ns_norm_marginal
    raw[~marginal_mask] = 0.0  # baseline residues: explicitly zero after normalization
    res_df["nucleus_score"] = raw

    if PLDDT_GRADIENT_WEIGHT > 0 and plddt_gradient is not None:
        grad_aligned = np.zeros(max_length, dtype=float)
        n = min(len(plddt_gradient), max_length)
        grad_aligned[:n] = plddt_gradient[:n]
        norm_grad = normalize_vector(grad_aligned) * 3.0
        raw = res_df["nucleus_score"].values.copy()
        raw = (1.0 - PLDDT_GRADIENT_WEIGHT) * raw + PLDDT_GRADIENT_WEIGHT * norm_grad
        raw[~marginal_mask] = 0.0
        res_df["nucleus_score"] = raw

    if CONTACT_RATE_WEIGHT > 0 and contact_formation_rate is not None:
        rate_aligned = np.zeros(max_length, dtype=float)
        n = min(len(contact_formation_rate), max_length)
        rate_aligned[:n] = contact_formation_rate[:n]
        norm_rate = normalize_vector(rate_aligned) * 3.0
        raw = res_df["nucleus_score"].values.copy()
        raw = (1.0 - CONTACT_RATE_WEIGHT) * raw + CONTACT_RATE_WEIGHT * norm_rate
        raw[~marginal_mask] = 0.0
        res_df["nucleus_score"] = raw

    # Threshold marginal residues with a length-adaptive sigma factor.
    marginal_scores = res_df.loc[marginal_mask, "nucleus_score"].values
    if len(marginal_scores) > 0:
        mu = np.mean(marginal_scores)
        sigma = np.std(marginal_scores)
        threshold = mu + adaptive_sigma_factor(max_length) * sigma
    else:
        threshold = SCORE_THRESHOLD_MARGINAL * 3.0
    res_df["is_nucleus"] = res_df["nucleus_score"] >= threshold
    return res_df


def score_residues_zscore(fragment_df: pd.DataFrame,
                          max_length: int) -> pd.DataFrame:
    """
    Z-score variant of FES stored in the ``nucleus_score`` column.
    Threshold exceedance defines a candidate flag, not a biological class.
    """
    fdf = fragment_df.set_index("length").sort_index()
    scores = fdf["score"]

    all_scores = scores.values
    mu = np.mean(all_scores)
    sigma = np.std(all_scores)

    records = []
    for res_i in range(1, max_length + 1):
        if res_i in scores.index:
            z = (scores[res_i] - mu) / sigma if sigma > 1e-9 else 0.0
        else:
            # Use nearest fragment
            available = [l for l in sorted(scores.index) if l >= res_i]
            if available:
                z = (scores[available[0]] - mu) / sigma if sigma > 1e-9 else 0.0
            else:
                z = 0.0
        records.append({"residue": res_i, "nucleus_score": z})

    res_df = pd.DataFrame(records)
    ns_norm = normalize_vector(res_df["nucleus_score"].values) * 3.0
    res_df["nucleus_score"] = ns_norm
    res_df["is_nucleus"] = res_df["nucleus_score"] >= SCORE_THRESHOLD
    return res_df


def score_residues_combined(fragment_df: pd.DataFrame,
                            max_length: int,
                            importance_df: pd.DataFrame = None,
                            w_marginal: float = None,
                            w_importance: float = None) -> pd.DataFrame:
    """
    Combined scoring: weighted fusion of ab initio score and structural importance.

    Uses score_residues_ab_initio() as the primary signal (truly ab initio:
    emergent contacts, self-referenced RMSF, SS onset timing — no native
    structure reference). Falls back to marginal scoring when prediction-
    ensemble data is unavailable.

    Adaptive weighting: when prediction quality is poor,
    the structural importance signal dominates.

      q_full < 0.20  → w_imp=0.90, w_marg=0.10
      q_full < 0.40  → w_imp=0.75, w_marg=0.25
      q_full ≥ 0.40  → w_imp=0.50, w_marg=0.50

    When importance_df is None or empty, falls back to pure ab initio / marginal.

    Parameters
    ----------
    fragment_df   : DataFrame from compute_fragment_scores
    max_length    : highest residue index to score
    importance_df : DataFrame with columns 'residue', 'importance'
    w_marginal    : explicit weight override (disables adaptive logic)
    w_importance  : explicit weight override (disables adaptive logic)

    Returns a DataFrame with columns: residue, nucleus_score, is_nucleus.
    """
    # Adjust weights when the primary signal is uninformative.
    if w_marginal is None and w_importance is None:
        q_full = 0.30  # neutral default
        if not fragment_df.empty:
            full_rows = fragment_df[fragment_df["length"] == max_length]
            if not full_rows.empty:
                q_col = ("q_local_mean" if "q_local_mean" in fragment_df.columns
                         else "q_mean")
                if q_col in full_rows.columns:
                    q_full = float(full_rows[q_col].iloc[0])
        if q_full < 0.20:
            w_marginal, w_importance = 0.10, 0.90
        elif q_full < 0.40:
            w_marginal, w_importance = 0.25, 0.75
        else:
            w_marginal, w_importance = 0.50, 0.50
    else:
        if w_marginal is None:
            w_marginal = SCORING_COMBINED_W_MARGINAL
        if w_importance is None:
            w_importance = SCORING_COMBINED_W_IMPORTANCE

    primary_df = score_residues_ab_initio(max_length)
    if primary_df.empty:
        primary_df = score_residues_marginal(fragment_df, max_length)

    if importance_df is None or importance_df.empty or "importance" not in importance_df.columns:
        return primary_df

    merged = primary_df.merge(
        importance_df[["residue", "importance"]], on="residue", how="left"
    )
    merged["importance"] = merged["importance"].fillna(0.0)

    ns_norm  = normalize_vector(merged["nucleus_score"].values)
    imp_norm = normalize_vector(merged["importance"].values)

    combined = (w_marginal * ns_norm + w_importance * imp_norm) * 3.0
    merged["nucleus_score"] = combined

    sigma_factor = adaptive_sigma_factor(max_length)
    mu    = float(np.mean(combined))
    sigma = float(np.std(combined))
    threshold = mu + sigma_factor * sigma
    merged["is_nucleus"] = merged["nucleus_score"] >= threshold

    return merged[["residue", "nucleus_score", "is_nucleus"]]


def score_residues_bidirectional(n_fragment_df: pd.DataFrame,
                                 c_fragment_df: pd.DataFrame,
                                 max_length: int,
                                 relative_contact_order: float = None) -> pd.DataFrame:
    """
    Bidirectional marginal scoring: fuse N-terminal and C-terminal fragment signals.

    Fusion is weighted by relative contact order.
    RCO ∈ [0,1]: low → MAX fusion; high → mean fusion of terminal scans.
    fused = RCO * mean(N,C) + (1-RCO) * max(N,C)

    Parameters
    ----------
    n_fragment_df            : fragment scores for N-terminal fragments
    c_fragment_df            : fragment scores for C-terminal fragments
    max_length               : protein length (number of residues)
    relative_contact_order   : RCO float from compute_relative_contact_order (optional)

    Returns a DataFrame with columns: residue, nucleus_score, is_nucleus.
    """
    n_df = score_residues_marginal(n_fragment_df, max_length) if not n_fragment_df.empty else pd.DataFrame()
    c_df = score_residues_marginal(c_fragment_df, max_length) if not c_fragment_df.empty else pd.DataFrame()

    if n_df.empty and c_df.empty:
        return pd.DataFrame()
    if n_df.empty:
        return c_df
    if c_df.empty:
        return n_df

    merged = n_df.rename(columns={"nucleus_score": "n_score", "is_nucleus": "n_nucleus"}).merge(
        c_df.rename(columns={"nucleus_score": "c_score", "is_nucleus": "c_nucleus"}),
        on="residue", how="outer"
    ).fillna(0.0)

    n_arr = merged["n_score"].values
    c_arr = merged["c_score"].values
    # RCO interpolates between local maximum and global mean evidence.
    alpha = float(np.clip(relative_contact_order, 0.0, 1.0)) if relative_contact_order is not None else 0.0
    fused = alpha * (n_arr + c_arr) / 2.0 + (1.0 - alpha) * np.maximum(n_arr, c_arr)
    merged["nucleus_score"] = fused

    scores = merged["nucleus_score"].values
    mu    = float(np.mean(scores))
    sigma = float(np.std(scores))
    threshold = mu + adaptive_sigma_factor(max_length) * sigma
    merged["is_nucleus"] = merged["nucleus_score"] >= threshold

    return merged[["residue", "nucleus_score", "is_nucleus"]]


def score_residues_ab_initio(max_length: int,
                              w_contact: float = 1/3,
                              w_rmsf: float = 1/3,
                              w_ss: float = 1/3) -> pd.DataFrame:
    """
    No-native-reference structural-emergence scoring.

    For each residue i, three signals are extracted from prediction ensembles
    without any knowledge of the native/target structure:

      1. emergent_contact_onset(i): smallest fragment length L where the
         residue develops >= EMERGENT_CONTACT_DEGREE_ONSET contacts that
         form consistently (probability >= EMERGENT_CONTACT_CONSISTENCY)
         in the ensemble. Earlier onset gives a higher score.

      2. rmsf_onset(i): smallest L where self-referenced RMSF(i,L) drops
         below the global median RMSF.  RMSF is computed relative to the
         mean ensemble coordinates without native superposition.
         Residues that rigidify early score higher.

      3. ss_onset(i): smallest L where SS_persistence(i,L) > 0.70.
         DSSP is applied per-frame — purely geometric, no native reference.
         Residues whose secondary structure stabilises early score higher.

    Combined score = w_contact * norm(1/onset_c)
                   + w_rmsf   * norm(1/onset_r)
                   + w_ss     * norm(1/onset_s)

    Parameters
    ----------
    max_length            : int — protein length (highest residue to score)
    w_contact, w_rmsf, w_ss : float — signal weights (should sum to 1.0)

    Returns
    -------
    DataFrame with columns: residue, nucleus_score, is_nucleus
    """
    from modules.contact_profile import (
        build_emergent_contact_profile,
        compute_rmsf_trajectory_ab_initio,
        compute_ss_trajectory,
        compute_onset_from_matrix,
    )

    n_res = max_length
    lengths = list(range(MIN_FRAGMENT_LENGTH, max_length + 1))
    fallback = float(max_length + 1)  # worst-case onset: never reached

    # Contact onset without a native reference
    profile, valid_lengths_c = build_emergent_contact_profile(n_res, lengths=lengths)
    if len(valid_lengths_c) == 0:
        return pd.DataFrame()
    contact_onset = compute_onset_from_matrix(
        profile.astype(float), valid_lengths_c,
        threshold=EMERGENT_CONTACT_DEGREE_ONSET - 1, above=True,
    )

    # Self-referenced RMSF onset
    rmsf_matrix, valid_lengths_r = compute_rmsf_trajectory_ab_initio(n_res, lengths=lengths)
    if len(valid_lengths_r) > 0 and rmsf_matrix.size > 0:
        rmsf_median = float(np.nanmedian(rmsf_matrix))
        rmsf_onset = compute_onset_from_matrix(rmsf_matrix, valid_lengths_r,
                                                threshold=rmsf_median, above=False)
    else:
        rmsf_onset = np.full(n_res, np.inf)

    # Geometric secondary-structure onset
    ss_matrix, valid_lengths_s = compute_ss_trajectory(n_res, lengths=lengths)
    if len(valid_lengths_s) > 0 and ss_matrix.size > 0:
        ss_onset = compute_onset_from_matrix(ss_matrix, valid_lengths_s,
                                              threshold=0.70, above=True)
    else:
        ss_onset = np.full(n_res, np.inf)

    co = np.where(np.isfinite(contact_onset[:n_res]), contact_onset[:n_res], fallback)
    ro = np.where(np.isfinite(rmsf_onset[:n_res]),    rmsf_onset[:n_res],    fallback)
    so = np.where(np.isfinite(ss_onset[:n_res]),       ss_onset[:n_res],      fallback)

    score_c = normalize_vector(1.0 / co)
    score_r = normalize_vector(1.0 / ro)
    score_s = normalize_vector(1.0 / so)

    combined = (w_contact * score_c + w_rmsf * score_r + w_ss * score_s) * 3.0

    sigma_factor = adaptive_sigma_factor(max_length)
    mu    = float(np.mean(combined))
    sigma = float(np.std(combined))
    threshold = mu + sigma_factor * sigma

    residues = np.arange(1, n_res + 1)
    return pd.DataFrame({
        "residue":       residues,
        "nucleus_score": combined,
        "is_nucleus":    combined >= threshold,
    })


def score_residues_native(ref_pdb_path: str) -> pd.DataFrame:
    """
    Score residues from the native PDB structure only.

    Uses compute_importance_from_native() to derive all five importance metrics
    from the native coordinates, then applies the standard adaptive threshold.

    Accessible via SCORING_METHOD = "native" in config, or called directly.

    Parameters
    ----------
    ref_pdb_path : path to the native/reference PDB

    Returns
    -------
    DataFrame with columns: residue, nucleus_score, is_nucleus
    """
    from modules.importance import compute_importance_from_native

    importance_df = compute_importance_from_native(ref_pdb_path)
    if importance_df.empty:
        return pd.DataFrame()

    n_res  = len(importance_df)
    scores = normalize_vector(importance_df["importance"].values) * 3.0

    sigma_factor = adaptive_sigma_factor(n_res)
    mu    = float(np.mean(scores))
    sigma = float(np.std(scores))
    threshold = mu + sigma_factor * sigma

    return pd.DataFrame({
        "residue":      importance_df["residue"].values,
        "nucleus_score": scores,
        "is_nucleus":   scores >= threshold,
    })


def score_residues_unified(fragment_df: pd.DataFrame,
                          max_length: int,
                          importance_df: pd.DataFrame = None,
                          ref_pdb_path: str = None,
                          plddt_gradient: np.ndarray = None,
                          contact_formation_rate: np.ndarray = None) -> pd.DataFrame:
    """
    Unified fragment-emergence scoring with adaptive weights.

    Axis A — Fragment-addition marginal: delta(i) = score(L=i) - score(L=i-1)
    Axis B — Intermediate-contact: q_tse, lr_fraction and coop_corr from an
             operational Q window (q_tse; not a validated TSE)
    Axis C — Topological (structural): contact degree, SS persistence, betweenness

    Weights adjust adaptively based on intermediate-window sampling quality.

    Parameters
    ----------
    fragment_df : DataFrame from compute_fragment_scores (N-terminal fragments)
    max_length : int
        Protein length
    importance_df : DataFrame with 'residue' and 'importance' columns (optional)
    ref_pdb_path : str
        Path to native PDB (needed to compute importance if not provided)
    plddt_gradient : (n_res,) array (optional, forwarded to marginal scoring)
    contact_formation_rate : (n_res,) array (optional, forwarded to marginal scoring)

    Returns
    -------
    DataFrame with columns:
        residue, nucleus_score, is_nucleus, axis_kinetic, axis_cooperative,
        axis_topological, w_kinetic, w_cooperative, w_topological
    """
    # Axis A: fragment-addition marginal score
    if fragment_df.empty:
        return pd.DataFrame()

    marginal_df = score_residues_marginal(
        fragment_df, max_length,
        plddt_gradient=plddt_gradient,
        contact_formation_rate=contact_formation_rate
    )
    if marginal_df.empty:
        return pd.DataFrame()

    axis_a_scores = normalize_vector(marginal_df["nucleus_score"].values)

    # Axis B: operational intermediate-contact subset
    from core.analyze import compute_native_contacts, _load_traj
    import warnings

    axis_b_scores = np.zeros(max_length, dtype=float)
    n_tse_frames = 0

    if ref_pdb_path is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_traj = _load_traj(ref_pdb_path)
            native_pairs, _ = compute_native_contacts(ref_traj)

            from modules.cooperativity import compute_cooperative_signals

            coop_dict = compute_cooperative_signals(max_length, native_pairs, max_length)

            if coop_dict["available"]:
                q_tse = coop_dict["q_tse"]
                lr_frac = coop_dict["lr_fraction"]
                coop_corr = coop_dict["coop_corr"]
                n_tse_frames = coop_dict["n_tse_frames"]

                q_tse_n = normalize_vector(q_tse)
                lr_n = normalize_vector(lr_frac)
                coop_n = normalize_vector(coop_corr)

                axis_b_scores = (
                    UNIFIED_COOP_W_TSE * q_tse_n +
                    UNIFIED_COOP_W_LR * lr_n +
                    UNIFIED_COOP_W_COOP * coop_n
                )
        except Exception as e:
            import logging as _logging
            _logging.getLogger("dataset").warning(f"[unified] Cooperative axis unavailable: {e}")

    axis_b_scores_norm = normalize_vector(axis_b_scores)

    # Axis C: Topological (structural importance)
    axis_c_scores = np.zeros(max_length, dtype=float)

    if importance_df is not None and not importance_df.empty:
        imp_vals = importance_df["importance"].values
        n_copy = min(len(imp_vals), max_length)
        axis_c_scores[:n_copy] = imp_vals[:n_copy]
    elif ref_pdb_path is not None:
        try:
            from modules.importance import compute_importance_from_native
            imp_df = compute_importance_from_native(ref_pdb_path)
            if not imp_df.empty:
                imp_vals = imp_df["importance"].values
                n_copy = min(len(imp_vals), max_length)
                axis_c_scores[:n_copy] = imp_vals[:n_copy]
        except Exception as e:
            import logging as _logging
            _logging.getLogger("dataset").warning(f"[unified] Topological axis unavailable: {e}")

    axis_c_scores_norm = normalize_vector(axis_c_scores)

    alpha = UNIFIED_W_KINETIC
    beta = UNIFIED_W_COOPERATIVE
    gamma = UNIFIED_W_TOPOLOGICAL

    if n_tse_frames < TSE_MIN_FRAMES_PARTIAL:
        surplus = beta
        beta = 0.0
        ratio_a = alpha / (alpha + gamma) if (alpha + gamma) > 0 else 0.5
        alpha += surplus * ratio_a
        gamma += surplus * (1.0 - ratio_a)
    elif n_tse_frames < TSE_MIN_FRAMES:
        beta = UNIFIED_W_COOPERATIVE * 0.5
        surplus = UNIFIED_W_COOPERATIVE - beta
        ratio_a = alpha / (alpha + gamma) if (alpha + gamma) > 0 else 0.5
        alpha += surplus * ratio_a
        gamma += surplus * (1.0 - ratio_a)

    # Downweight a failed full-length prediction.
    q_local_full = None
    if "q_local_mean" in fragment_df.columns:
        full_len_rows = fragment_df[fragment_df["length"] == max_length]
        if not full_len_rows.empty:
            q_local_full = float(full_len_rows["q_local_mean"].iloc[0])

    if q_local_full is not None and q_local_full < 0.01:
        # Reserve this penalty for near-zero, not merely weak, local Q.
        reduction = alpha * 0.7
        alpha -= reduction
        gamma += reduction

    weight_sum = alpha + beta + gamma
    if weight_sum > 0:
        alpha /= weight_sum
        beta /= weight_sum
        gamma /= weight_sum
    else:
        alpha = beta = gamma = 1.0 / 3.0

    unified_scores = (alpha * axis_a_scores +
                      beta * axis_b_scores_norm +
                      gamma * axis_c_scores_norm)

    unified_rescaled = normalize_vector(unified_scores) * 3.0

    sigma_factor = adaptive_sigma_factor(max_length)
    mu = float(np.mean(unified_rescaled))
    sigma = float(np.std(unified_rescaled))
    threshold = mu + sigma_factor * sigma
    is_nucleus = unified_rescaled >= threshold

    # Bimodality is diagnostic only and does not change candidate selection.
    has_clear_nucleus = True  # denotes score bimodality only
    dip_stat, dip_pval = float("nan"), float("nan")
    if len(unified_rescaled) >= 8:
        try:
            from diptest import diptest as _diptest
            dip_stat, dip_pval = _diptest(unified_rescaled)
            has_clear_nucleus = (dip_pval <= 0.10)
        except ImportError:
            pass

    residues = np.arange(1, max_length + 1)
    return pd.DataFrame({
        "residue":            residues,
        "nucleus_score":      unified_rescaled,
        "is_nucleus":         is_nucleus,
        "axis_kinetic":       axis_a_scores * 3.0,
        "axis_cooperative":   axis_b_scores_norm * 3.0,
        "axis_topological":   axis_c_scores_norm * 3.0,
        "w_kinetic":          alpha,
        "w_cooperative":      beta,
        "w_topological":      gamma,
        "has_clear_nucleus":  has_clear_nucleus,
        "has_score_bimodality": has_clear_nucleus,
        "dip_statistic":      dip_stat,
        "dip_pvalue":         dip_pval,
    })


def compute_plddt_gradient(metrics: list, n_residues: int) -> np.ndarray:
    """
    Compute per-residue pLDDT gradient: how much each residue's pLDDT changes
    when the fragment length increases by 1.

    For residue i, the gradient is the average of (pLDDT[i, L] - pLDDT[i, L-1])
    across all consecutive length pairs where residue i is present in both.

    Parameters
    ----------
    metrics    : list of dicts from analyze_fragment (must contain 'plddt_per_residue')
    n_residues : int — full protein length

    Returns
    -------
    gradient : (n_residues,) float array — mean ∂pLDDT/∂L per residue
    """
    have_plddt = [(m["length"], m["plddt_per_residue"])
                  for m in metrics
                  if "plddt_per_residue" in m and m["plddt_per_residue"] is not None]
    if len(have_plddt) < 2:
        return np.zeros(n_residues, dtype=float)

    have_plddt.sort(key=lambda x: x[0])
    gradient_acc  = np.zeros(n_residues, dtype=float)
    gradient_cnt  = np.zeros(n_residues, dtype=int)

    for idx in range(1, len(have_plddt)):
        l_prev, plddt_prev = have_plddt[idx - 1]
        l_curr, plddt_curr = have_plddt[idx]
        if l_curr != l_prev + 1:
            continue
        n_prev = len(plddt_prev)
        n_curr = len(plddt_curr)
        n_shared = min(n_prev, n_curr, n_residues)
        delta = plddt_curr[:n_shared] - plddt_prev[:n_shared]
        gradient_acc[:n_shared] += delta
        gradient_cnt[:n_shared] += 1

    with np.errstate(invalid="ignore"):
        gradient = np.where(gradient_cnt > 0,
                            gradient_acc / gradient_cnt,
                            0.0)
    return gradient


def score_residues(fragment_df: pd.DataFrame,
                   max_length: int = MAX_FRAGMENT_LENGTH,
                   method: str = None,
                   importance_df: pd.DataFrame = None,
                   c_fragment_df: pd.DataFrame = None,
                   plddt_gradient: np.ndarray = None,
                   contact_formation_rate: np.ndarray = None,
                   ref_pdb_path: str = None,
                   relative_contact_order: float = None) -> pd.DataFrame:
    """
    Assign a fragment-emergence score to each residue 1..max_length.

    Parameters
    ----------
    fragment_df   : DataFrame from compute_fragment_scores (N-terminal fragments)
    max_length    : highest residue index to score
    method        : 'sliding_window' | 'marginal' | 'zscore' | 'combined' |
                    'bidirectional' | 'ab_initio' | 'native' | 'unified'
                    Defaults to SCORING_METHOD from config.
    importance_df : for method='combined'/'unified', DataFrame with 'residue' + 'importance' columns
    c_fragment_df : for method='bidirectional', fragment scores for C-terminal fragments
    plddt_gradient  : (n_res,) array of ∂pLDDT/∂L per residue
    contact_formation_rate : (n_res,) array of contact-formation rates
    ref_pdb_path  : str, path to reference PDB (needed for unified and native methods)

    Returns columns ``nucleus_score`` (FES) and ``is_nucleus``
    (thresholded candidate flag), plus method-specific diagnostics.
    """
    if fragment_df.empty and method not in ("ab_initio", "native", "unified"):
        return pd.DataFrame()

    if method is None:
        method = SCORING_METHOD

    if method == "unified":
        return score_residues_unified(fragment_df, max_length, importance_df=importance_df,
                                      ref_pdb_path=ref_pdb_path,
                                      plddt_gradient=plddt_gradient,
                                      contact_formation_rate=contact_formation_rate)
    elif method == "combined":
        return score_residues_combined(fragment_df, max_length, importance_df)
    elif method == "ab_initio":
        result = score_residues_ab_initio(max_length)
        if result.empty:
            return score_residues_marginal(fragment_df, max_length,
                                           plddt_gradient=plddt_gradient,
                                           contact_formation_rate=contact_formation_rate)
        return result
    elif method == "bidirectional":
        return score_residues_bidirectional(
            fragment_df,
            c_fragment_df if c_fragment_df is not None else pd.DataFrame(),
            max_length,
            relative_contact_order=relative_contact_order,
        )
    elif method == "marginal":
        return score_residues_marginal(fragment_df, max_length,
                                       plddt_gradient=plddt_gradient,
                                       contact_formation_rate=contact_formation_rate)
    elif method == "zscore":
        return score_residues_zscore(fragment_df, max_length)
    elif method == "native":
        # Native-only scoring requires a PDB path; fragment_df is ignored.
        if ref_pdb_path is not None:
            return score_residues_native(ref_pdb_path)
        return pd.DataFrame()
    else:
        return score_residues_sliding_window(fragment_df, max_length)
