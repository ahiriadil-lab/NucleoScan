"""
Cooperativity-aware scoring in an operational intermediate-contact window.

Computes three metrics from the full-length fragment ensemble:
  1. q_tse — per-residue contact probability in an intermediate-Q subset
  2. lr_fraction — long-range contact fraction
  3. coop_corr — cooperativity correlation (Pearson r between local q and global Q)

Fuses these signals into the cooperative score axis. ``q_tse`` and related TSE
identifiers are legacy names retained for compatibility. Selecting frames by a
Q window does not establish a kinetic transition-state ensemble.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import (
    TSE_Q_LO, TSE_Q_HI, TSE_Q_LO_WIDE, TSE_Q_HI_WIDE,
    SEQ_SEP_LONG_RANGE, TSE_MIN_FRAMES, TSE_MIN_FRAMES_PARTIAL,
    UNIFIED_COOP_W_TSE, UNIFIED_COOP_W_LR, UNIFIED_COOP_W_COOP,
    adaptive_sigma_factor,
)
from core.analyze import (
    load_ensemble, load_raw_metrics, compute_native_contacts,
    compute_per_residue_q, compute_per_frame_per_residue_contacts,
    _load_traj,
)
from core.score import normalize_vector


# Intermediate-contact frame selection

def extract_tse_frames(length, use_wide=False):
    """
    Select frames from a full-length fragment ensemble using a Q window.

    This is an operational intermediate-contact subset, not a transition-state
    ensemble unless independently supported by a kinetic committor analysis.

    Parameters
    ----------
    length : int
        Fragment length (typically = max_length for full-length)
    use_wide : bool
        If True, use wide Q-window [TSE_Q_LO_WIDE, TSE_Q_HI_WIDE]
        Otherwise, use standard [TSE_Q_LO, TSE_Q_HI]

    Returns
    -------
    tse_mask : (n_decoys,) bool array
        Legacy name for the intermediate-window selection mask.
    n_tse : int
        Number of selected intermediate-window frames.
    """
    try:
        _rmsds, qs, _hydros = load_raw_metrics(length)
    except Exception:
        return np.array([], dtype=bool), 0

    if use_wide:
        tse_mask = (qs >= TSE_Q_LO_WIDE) & (qs <= TSE_Q_HI_WIDE)
    else:
        tse_mask = (qs >= TSE_Q_LO) & (qs <= TSE_Q_HI)

    return tse_mask, int(tse_mask.sum())


def _join_compatible_frames(frames):
    """
    Join a list of single-frame mdtraj trajectories.
    Keeps only those with the same n_atoms as the first frame.
    """
    import mdtraj as md
    if not frames:
        return None
    ref_n = frames[0].n_atoms
    compatible = [f for f in frames if f.n_atoms == ref_n]
    if not compatible:
        return None
    if len(compatible) == 1:
        return compatible[0]
    return md.join(compatible, check_topology=False)


# Per-residue Q in the intermediate-contact subset

def compute_per_residue_q_tse(length, tse_mask, native_pairs, n_res_full):
    """
    Per-residue contact probability restricted to intermediate-Q frames.

    Parameters
    ----------
    length : int
        Fragment length
    tse_mask : (n_decoys,) bool
        Mask selecting intermediate-Q frames
    native_pairs : (n_pairs, 2) int array
        Native contact pairs
    n_res_full : int
        Full protein length (for padding)

    Returns
    -------
    q_tse : (n_res_full,) float array
    """
    trajs = load_ensemble(length)
    if not trajs:
        return np.zeros(n_res_full, dtype=float)

    import mdtraj as md
    frames = []
    for t in trajs:
        for fi in range(t.n_frames):
            frames.append(t.slice(fi))

    tse_indices = np.where(tse_mask)[0]
    valid_indices = tse_indices[tse_indices < len(frames)]
    if len(valid_indices) == 0:
        return np.zeros(n_res_full, dtype=float)

    tse_trajs = [frames[i] for i in valid_indices]
    combined = _join_compatible_frames(tse_trajs)
    if combined is None:
        return np.zeros(n_res_full, dtype=float)

    n_res_frag = combined.topology.n_residues
    q_tse_frag = compute_per_residue_q(combined, native_pairs)

    # Pad to full-length protein size
    q_tse = np.zeros(n_res_full, dtype=float)
    n_copy = min(n_res_frag, n_res_full)
    q_tse[:n_copy] = q_tse_frag[:n_copy]
    return q_tse


# Long-range contact fraction in the intermediate-contact subset

def compute_long_range_contact_fraction_tse(length, tse_mask, native_pairs, n_res_full,
                                            seq_sep=None):
    """
    Per-residue fraction of selected-window contacts that are long-range.

    Parameters
    ----------
    length : int
        Fragment length
    tse_mask : (n_decoys,) bool
        Mask selecting intermediate-Q frames
    native_pairs : (n_pairs, 2) int array
        Native contact pairs
    n_res_full : int
        Full protein length
    seq_sep : int or None
        Sequence separation threshold (default: SEQ_SEP_LONG_RANGE)

    Returns
    -------
    lr_fraction : (n_res_full,) float array in [0, 1]
    """
    if seq_sep is None:
        seq_sep = SEQ_SEP_LONG_RANGE

    trajs = load_ensemble(length)
    if not trajs:
        return np.zeros(n_res_full, dtype=float)

    import mdtraj as md
    frames = []
    for t in trajs:
        for fi in range(t.n_frames):
            frames.append(t.slice(fi))

    tse_indices = np.where(tse_mask)[0]
    valid_indices = tse_indices[tse_indices < len(frames)]
    if len(valid_indices) == 0:
        return np.zeros(n_res_full, dtype=float)

    tse_trajs = [frames[i] for i in valid_indices]
    combined = _join_compatible_frames(tse_trajs)
    if combined is None:
        return np.zeros(n_res_full, dtype=float)

    # Separate native_pairs into long-range and all
    sep = np.abs(native_pairs[:, 0].astype(int) - native_pairs[:, 1].astype(int))
    lr_pairs = native_pairs[sep > seq_sep]

    q_all = compute_per_residue_q(combined, native_pairs)
    if len(lr_pairs) == 0:
        lr_fraction_frag = np.zeros(combined.topology.n_residues, dtype=float)
    else:
        q_lr = compute_per_residue_q(combined, lr_pairs)
        with np.errstate(invalid="ignore", divide="ignore"):
            lr_fraction_frag = np.clip(q_lr / (q_all + 1e-9), 0.0, 1.0)

    n_res_frag = combined.topology.n_residues
    lr_fraction = np.zeros(n_res_full, dtype=float)
    n_copy = min(n_res_frag, n_res_full)
    lr_fraction[:n_copy] = lr_fraction_frag[:n_copy]
    return lr_fraction


# Cooperativity Correlation

def compute_cooperativity_correlation(length, native_pairs, n_res_full):
    """
    Per-residue Pearson correlation between local q and global Q.

    Uses all frames, not only the selected intermediate-Q subset.

    Parameters
    ----------
    length : int
        Fragment length
    native_pairs : (n_pairs, 2) int array
        Native contact pairs
    n_res_full : int
        Full protein length

    Returns
    -------
    coop_corr : (n_res_full,) float array in [-1, 1]
    """
    trajs = load_ensemble(length)
    if not trajs:
        return np.zeros(n_res_full, dtype=float)

    combined = _join_compatible_frames(trajs)
    if combined is None:
        return np.zeros(n_res_full, dtype=float)

    q_matrix, global_q = compute_per_frame_per_residue_contacts(combined, native_pairs)
    # q_matrix : (n_frames, n_res_frag)
    # global_q : (n_frames,)

    n_frames, n_res_frag = q_matrix.shape
    coop_corr_frag = np.zeros(n_res_frag, dtype=float)

    if n_frames < 3:
        # Cannot compute meaningful correlation
        coop_corr = np.zeros(n_res_full, dtype=float)
        n_copy = min(n_res_frag, n_res_full)
        coop_corr[:n_copy] = coop_corr_frag[:n_copy]
        return coop_corr

    std_global = float(np.std(global_q))
    for i in range(n_res_frag):
        col = q_matrix[:, i]
        if np.std(col) < 1e-9 or std_global < 1e-9:
            coop_corr_frag[i] = 0.0
        else:
            try:
                r, _ = pearsonr(col, global_q)
                coop_corr_frag[i] = r if np.isfinite(r) else 0.0
            except Exception:
                coop_corr_frag[i] = 0.0

    coop_corr = np.zeros(n_res_full, dtype=float)
    n_copy = min(n_res_frag, n_res_full)
    coop_corr[:n_copy] = coop_corr_frag[:n_copy]
    return coop_corr


# Orchestrator

def compute_cooperative_signals(max_length, native_pairs, n_res):
    """
    Compute cooperative signals from a full-length fragment ensemble.

    The dictionary keys retain their historical TSE terminology, but the
    selected frames are defined only by an operational Q window.

    Parameters
    ----------
    max_length : int
        Fragment length (typically = protein length)
    native_pairs : (n_pairs, 2) int array
        Native contact pairs
    n_res : int
        Protein length (for output consistency)

    Returns
    -------
    dict with keys:
        'q_tse': (n_res,) array (legacy name for intermediate-window contact score)
        'lr_fraction': (n_res,) array
        'coop_corr': (n_res,) array
        'n_tse_frames': int (legacy name for selected-window frame count)
        'available': bool (True if n_tse >= TSE_MIN_FRAMES)
    """
    # Try the standard intermediate-Q window.
    tse_mask, n_tse = extract_tse_frames(max_length, use_wide=False)

    # Fall back to the wide window if sampling is insufficient.
    if n_tse < TSE_MIN_FRAMES:
        tse_mask, n_tse = extract_tse_frames(max_length, use_wide=True)

    # Check availability
    available = n_tse >= TSE_MIN_FRAMES

    if available:
        q_tse = compute_per_residue_q_tse(max_length, tse_mask, native_pairs, n_res)
        lr_fraction = compute_long_range_contact_fraction_tse(
            max_length, tse_mask, native_pairs, n_res
        )
        coop_corr = compute_cooperativity_correlation(max_length, native_pairs, n_res)
    else:
        q_tse = np.zeros(n_res, dtype=float)
        lr_fraction = np.zeros(n_res, dtype=float)
        coop_corr = np.zeros(n_res, dtype=float)

    return {
        'q_tse': q_tse,
        'lr_fraction': lr_fraction,
        'coop_corr': coop_corr,
        'n_tse_frames': int(n_tse),
        'available': available,
    }
