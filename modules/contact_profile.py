"""
Contact formation profile — per-residue contact, RMSF and SS persistence
trajectories across progressive fragment lengths.

Used by score_residues_ab_initio() to construct an onset-based structural-
emergence score. "Early" refers to smaller sequence prefixes, not elapsed
folding time; the result is not direct evidence of folding-nucleus membership.
"""
import warnings
import numpy as np

from config import (
    STRUCTURES_DIR, MIN_FRAGMENT_LENGTH, CONTACT_CUTOFF_NM,
    CONTACT_ONSET_THRESHOLD, EMERGENT_CONTACT_CONSISTENCY,
    EMERGENT_CONTACT_DEGREE_ONSET,
)
from core.analyze import (
    load_ensemble, compute_per_residue_q, compute_emergent_contact_degree,
)
from modules.importance import compute_ss_persistence


# Contact formation profile

def build_contact_formation_profile(ref_traj, native_pairs, lengths=None):
    """
    For each residue i and each fragment length L, compute q_residue(i, L):
    the fraction of residue i's native contacts formed in the prediction ensemble at L.

    Parameters
    ----------
    ref_traj     : md.Trajectory — native/reference structure
    native_pairs : (n_pairs, 2) int array of native contact pair indices
    lengths      : list[int] — fragment lengths to evaluate
                   (default: MIN_FRAGMENT_LENGTH … n_res)

    Returns
    -------
    profile       : np.ndarray (n_res, n_valid_lengths) — per-residue Q matrix
    valid_lengths : list[int] — lengths for which ensembles were found
    """
    import mdtraj as md

    n_res = ref_traj.topology.n_residues
    if lengths is None:
        lengths = list(range(MIN_FRAGMENT_LENGTH, n_res + 1))

    valid_lengths = []
    profiles = []

    for length in lengths:
        trajs = load_ensemble(length)
        if not trajs:
            continue

        # Concatenate all frames (same topology for a given length)
        try:
            xyz_all = np.concatenate([t.xyz for t in trajs], axis=0)
            combined = md.Trajectory(xyz_all, trajs[0].topology)
        except Exception:
            combined = trajs[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q_per_res = compute_per_residue_q(combined, native_pairs)

        # Pad to n_res (fragment has fewer residues than full protein)
        full = np.zeros(n_res, dtype=float)
        n = min(len(q_per_res), n_res)
        full[:n] = q_per_res[:n]

        profiles.append(full)
        valid_lengths.append(length)

    if not profiles:
        return np.zeros((n_res, 0)), []

    profile = np.column_stack(profiles)  # (n_res, n_valid_lengths)
    return profile, valid_lengths


# Ab initio contact profile (no native reference)

def build_emergent_contact_profile(n_res, lengths=None):
    """
    For each residue i and each fragment length L, compute the emergent contact
    degree: the number of contacts that form with probability >=
    EMERGENT_CONTACT_CONSISTENCY in the prediction ensemble at L.

    No native structure reference is used — contacts are discovered purely from
    the prediction-ensemble geometry.

    Parameters
    ----------
    n_res   : int — number of residues in the protein
    lengths : list[int] — fragment lengths to evaluate
              (default: MIN_FRAGMENT_LENGTH … n_res)

    Returns
    -------
    profile       : np.ndarray (n_res, n_valid_lengths) — emergent contact degree
    valid_lengths : list[int]
    """
    import mdtraj as md

    if lengths is None:
        lengths = list(range(MIN_FRAGMENT_LENGTH, n_res + 1))

    valid_lengths = []
    profiles = []

    for length in lengths:
        trajs = load_ensemble(length)
        if not trajs:
            continue

        try:
            xyz_all = np.concatenate([t.xyz for t in trajs], axis=0)
            combined = md.Trajectory(xyz_all, trajs[0].topology)
        except Exception:
            combined = trajs[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degree = compute_emergent_contact_degree(combined)

        # Pad to n_res (fragment may be shorter than full protein)
        full = np.zeros(n_res, dtype=int)
        n = min(len(degree), n_res)
        full[:n] = degree[:n]

        profiles.append(full)
        valid_lengths.append(length)

    if not profiles:
        return np.zeros((n_res, 0), dtype=int), []

    profile = np.column_stack(profiles)  # (n_res, n_valid_lengths)
    return profile, valid_lengths


# Ab initio RMSF trajectory (self-referenced, no native)

def compute_rmsf_trajectory_ab_initio(n_res, lengths=None):
    """
    Per-residue RMSF from prediction ensembles, without a native reference.

    For each fragment length L, the concatenated trajectory is superposed on
    frame 0 (internal reference), then md.rmsf(reference=None) computes RMSF
    relative to the mean coordinates — the standard MD definition (Hess 2002).
    No native structure is used anywhere.

    Parameters
    ----------
    n_res   : int — number of residues in the protein
    lengths : list[int] — fragment lengths to evaluate

    Returns
    -------
    rmsf_matrix   : np.ndarray (n_res, n_valid_lengths)
    valid_lengths : list[int]
    """
    import mdtraj as md

    if lengths is None:
        lengths = list(range(MIN_FRAGMENT_LENGTH, n_res + 1))

    valid_lengths = []
    rmsf_cols = []

    for length in lengths:
        trajs = load_ensemble(length)
        if not trajs:
            continue

        try:
            xyz_all = np.concatenate([t.xyz for t in trajs], axis=0)
            combined = md.Trajectory(xyz_all, trajs[0].topology)
        except Exception:
            combined = trajs[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ca_idx = combined.topology.select("name CA")
            if len(ca_idx) == 0:
                continue
            ca_sub = combined.atom_slice(ca_idx)
            # Superpose on frame 0 — purely internal reference, no native used
            ca_sub.superpose(ca_sub, frame=0)
            # reference=None → uses mean coordinates (standard RMSF definition)
            rmsf_vals = md.rmsf(ca_sub, reference=None)

        # Pad to n_res with NaN for residues not in this fragment
        full = np.full(n_res, np.nan)
        n = min(len(rmsf_vals), n_res)
        full[:n] = rmsf_vals[:n]

        rmsf_cols.append(full)
        valid_lengths.append(length)

    if not rmsf_cols:
        return np.full((n_res, 0), np.nan), []

    return np.column_stack(rmsf_cols), valid_lengths


# SS persistence trajectory

def compute_ss_trajectory(n_res_or_ref_traj, lengths=None):
    """
    For each residue i and each fragment length L, compute SS_persistence(i, L).

    Uses the single predicted structure for each fragment length (binary H/E=1 or C=0).

    Parameters
    ----------
    n_res_or_ref_traj : int or md.Trajectory
        Either the number of residues (int) or a reference trajectory
        (backward-compatible; only n_residues is used from it).

    Returns
    -------
    ss_matrix     : np.ndarray (n_res, n_valid_lengths) — SS persistence
    valid_lengths : list[int]
    """
    if isinstance(n_res_or_ref_traj, int):
        n_res = n_res_or_ref_traj
    else:
        n_res = n_res_or_ref_traj.topology.n_residues
    if lengths is None:
        lengths = list(range(MIN_FRAGMENT_LENGTH, n_res + 1))

    valid_lengths = []
    ss_cols = []

    for length in lengths:
        trajs = load_ensemble(length)
        if not trajs:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # compute_ss_persistence takes a single trajectory
            ss_vals = compute_ss_persistence(trajs[0])

        # Pad to n_res (fragment may be shorter than full protein)
        full = np.zeros(n_res, dtype=float)
        n = min(len(ss_vals), n_res)
        full[:n] = ss_vals[:n]

        ss_cols.append(full)
        valid_lengths.append(length)

    if not ss_cols:
        return np.zeros((n_res, 0)), []

    return np.column_stack(ss_cols), valid_lengths


# Onset detection

def compute_contact_onset(profile, lengths, threshold=None):
    """
    For each residue, find the smallest length L where q_residue(i, L) > threshold.

    Parameters
    ----------
    profile   : (n_res, n_lengths) float array
    lengths   : list[int]
    threshold : float (default: CONTACT_ONSET_THRESHOLD from config)

    Returns
    -------
    onset_lengths : (n_res,) float array
                    Length at which contact threshold is first exceeded.
                    np.inf if the threshold is never reached.
    """
    if threshold is None:
        threshold = CONTACT_ONSET_THRESHOLD
    n_res = profile.shape[0]
    onset = np.full(n_res, np.inf)
    for col_idx, L in enumerate(lengths):
        q = profile[:, col_idx]
        triggered = (onset == np.inf) & (q > threshold)
        onset[triggered] = float(L)
    return onset


def compute_onset_from_matrix(matrix, lengths, threshold, above=True):
    """
    Generic onset finder: for each residue, find the first length where
    value > threshold (above=True) or value < threshold (above=False).

    NaN values are treated as "not triggered" (residue not yet in the fragment).

    Returns
    -------
    onset_lengths : (n_res,) float array — onset lengths (np.inf if never triggered)
    """
    n_res = matrix.shape[0]
    onset = np.full(n_res, np.inf)
    for col_idx, L in enumerate(lengths):
        vals = matrix[:, col_idx]
        valid = ~np.isnan(vals)
        if above:
            triggered = valid & (onset == np.inf) & (vals > threshold)
        else:
            triggered = valid & (onset == np.inf) & (vals < threshold)
        onset[triggered] = float(L)
    return onset
