"""
Module 3 — Contact Maps.

Computes residue-residue contact maps for each fragment's single prediction,
compares with the native contact map, and tracks when each native contact
forms along the fragment length axis.

With one structure per fragment, contact maps are binary (contact present/absent)
rather than probability maps.

Outputs
-------
results/contact_maps/contact_map_native.png
results/contact_maps/contact_map_frag{NN:03d}.png  (selected)
results/contact_formation.png
results/contact_formation.csv
"""
import os
import glob
import warnings
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md

from config import (
    RESULTS_DIR, STRUCTURES_DIR,
    CONTACT_CUTOFF_NM, CONTACT_FORMATION_THRESHOLD,
    MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH,
    LONG_RANGE_CONTACT_SEPARATION, LONG_RANGE_CONTACT_WEIGHT,
)
from core.analyze import _load_traj, load_ensemble, get_pdb_paths

_CMAP_DIR = os.path.join(RESULTS_DIR, "contact_maps")


# Native contact map

def compute_native_contact_map(ref_traj, cutoff=CONTACT_CUTOFF_NM):
    """
    Binary (n_res × n_res) contact map for the reference structure.

    Returns
    -------
    native_map : (n_res, n_res) bool array
    """
    n_res = ref_traj.topology.n_residues
    native_map = np.zeros((n_res, n_res), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distances, pairs = md.compute_contacts(ref_traj, scheme="closest-heavy")
    d0 = distances[0]
    for (i, j), d in zip(pairs, d0):
        if d <= cutoff:
            native_map[i, j] = True
            native_map[j, i] = True
    return native_map


# Contact map for a single structure

def compute_contact_map_single(traj, n_res, cutoff=CONTACT_CUTOFF_NM):
    """
    Binary contact map from a single structure.

    Parameters
    ----------
    traj  : md.Trajectory (single frame)
    n_res : int — size of the square matrix (reference residue count)

    Returns
    -------
    cmap : (n_res, n_res) float array with values 0.0 or 1.0
    """
    cmap = np.zeros((n_res, n_res), dtype=float)
    frag_res = traj.topology.n_residues
    if frag_res < 2:
        return cmap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            distances, pairs = md.compute_contacts(traj, scheme="closest-heavy")
        except Exception:
            return cmap
    d0 = distances[0]
    for (i, j), d in zip(pairs, d0):
        if i < n_res and j < n_res and d <= cutoff:
            cmap[i, j] = 1.0
            cmap[j, i] = 1.0
    return cmap


# Compute & cache maps for all fragments

def compute_all_contact_maps(ref_traj, lengths=None, chunk_size=10):
    """
    Compute binary contact maps for all fragment lengths and cache them.

    Returns
    -------
    cmaps : dict {length: (n_res, n_res) float array}
    """
    if lengths is None:
        lengths = range(MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH + 1)
    os.makedirs(_CMAP_DIR, exist_ok=True)

    n_res = ref_traj.topology.n_residues
    cmaps = {}
    lengths = list(lengths)

    h5_path = os.path.join(_CMAP_DIR, "contact_maps.h5")
    for chunk_start in range(0, len(lengths), chunk_size):
        chunk = lengths[chunk_start: chunk_start + chunk_size]
        for length in chunk:
            key = f"frag{length:03d}"
            with h5py.File(h5_path, "a") as f:
                if key in f:
                    cached_cutoff = f[key].attrs.get("cutoff_nm", None)
                    if cached_cutoff is not None and float(cached_cutoff) != CONTACT_CUTOFF_NM:
                        del f[key]
                    else:
                        cmaps[length] = np.array(f[key]["prob_map"])
                        continue

            # Load single prediction
            pdbs = get_pdb_paths(length)
            if not pdbs:
                continue
            try:
                traj = _load_traj(pdbs[0])
            except Exception:
                continue

            cmap = compute_contact_map_single(traj, n_res)
            with h5py.File(h5_path, "a") as f:
                g = f.create_group(key)
                g.create_dataset("prob_map", data=cmap, compression="gzip")
                g.attrs["cutoff_nm"] = CONTACT_CUTOFF_NM
            cmaps[length] = cmap

    return cmaps


def load_contact_map(length, cmap_dir=None):
    """Load a cached contact probability map."""
    if cmap_dir is None:
        cmap_dir = _CMAP_DIR
    h5_path = os.path.join(cmap_dir, "contact_maps.h5")
    key = f"frag{length:03d}"
    with h5py.File(h5_path, "r") as f:
        return np.array(f[key]["prob_map"])


# Contact formation tracking

def compute_contact_formation(cmaps, native_map, lengths):
    """
    For each native contact pair (i, j), find the smallest fragment length
    where prob > CONTACT_FORMATION_THRESHOLD.

    Returns
    -------
    DataFrame with columns: res_i, res_j, separation, formation_length
    """
    rows = []
    sorted_lengths = sorted(lengths)
    native_pairs = list(zip(*np.where(np.triu(native_map, k=1))))

    for (i, j) in native_pairs:
        separation = j - i
        formation_length = None
        for length in sorted_lengths:
            pmap = cmaps.get(length)
            if pmap is None:
                continue
            if i < pmap.shape[0] and j < pmap.shape[1]:
                if pmap[i, j] >= CONTACT_FORMATION_THRESHOLD:
                    formation_length = length
                    break
        rows.append({
            "res_i":            int(i),
            "res_j":            int(j),
            "separation":       int(separation),
            "formation_length": int(formation_length) if formation_length is not None else np.nan,
        })

    return pd.DataFrame(rows)


# Plotting

def plot_contact_map(matrix, title, out_path, cmap="Blues"):
    """Heatmap of a contact map (native binary or probability)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, origin="lower", cmap=cmap,
                   vmin=0, vmax=1 if matrix.max() <= 1 else None,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, label="Contact probability")
    ax.set_xlabel("Residue j")
    ax.set_ylabel("Residue i")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Contact map saved to {out_path}")


def plot_contact_formation(formation_df, out_path=None):
    """Scatter: sequence separation vs formation length for native contacts."""
    if formation_df.empty:
        return
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "contact_formation.png")

    df = formation_df.dropna(subset=["formation_length"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["separation"], df["formation_length"],
                    c=df["separation"], cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Sequence separation |i−j|")
    ax.set_xlabel("Sequence separation |i−j|")
    ax.set_ylabel("Formation length (residues)")
    ax.set_title("Native Contact Formation vs Sequence Separation")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Contact formation plot saved to {out_path}")


# Per-residue contact-formation rate

def compute_per_residue_contact_formation_rate(
        cmaps: dict,
        native_map: np.ndarray,
        lengths: list,
        long_range_sep: int = None,
        long_range_weight: float = None,
) -> np.ndarray:
    """
    For each residue i, compute the weighted-average fraction of its native
    contacts that are formed across all available fragment lengths.

    Long-range contacts (|i-j| > long_range_sep) are given extra weight
    (long_range_weight) because they provide a more selective topology signal.

    Parameters
    ----------
    cmaps          : dict {length: (n_res, n_res) float array} from compute_all_contact_maps
    native_map     : (n_res, n_res) bool array from compute_native_contact_map
    lengths        : list[int] of fragment lengths to include
    long_range_sep : int — |i-j| > this → long-range (default: LONG_RANGE_CONTACT_SEPARATION)
    long_range_weight : float — extra weight for long-range contacts (default: LONG_RANGE_CONTACT_WEIGHT)

    Returns
    -------
    rate : (n_res,) float array in [0, 1] — weighted contact formation rate per residue
    """
    if long_range_sep is None:
        long_range_sep = LONG_RANGE_CONTACT_SEPARATION
    if long_range_weight is None:
        long_range_weight = LONG_RANGE_CONTACT_WEIGHT

    n_res = native_map.shape[0]
    native_pairs = list(zip(*np.where(np.triu(native_map, k=1))))

    if not native_pairs or not cmaps:
        return np.zeros(n_res, dtype=float)

    sorted_lengths = sorted(l for l in lengths if l in cmaps)
    n_lengths = len(sorted_lengths)
    if n_lengths == 0:
        return np.zeros(n_res, dtype=float)

    # Accumulate weighted formation rate per residue
    rate_acc  = np.zeros(n_res, dtype=float)
    weight_acc = np.zeros(n_res, dtype=float)

    for (i, j) in native_pairs:
        w = long_range_weight if abs(j - i) > long_range_sep else 1.0
        # Count how many fragment lengths have this contact formed
        formed = sum(
            1 for l in sorted_lengths
            if i < cmaps[l].shape[0] and j < cmaps[l].shape[1]
            and cmaps[l][i, j] >= CONTACT_FORMATION_THRESHOLD
        )
        formation_fraction = formed / n_lengths
        rate_acc[i]   += w * formation_fraction
        rate_acc[j]   += w * formation_fraction
        weight_acc[i] += w
        weight_acc[j] += w

    with np.errstate(invalid="ignore"):
        rate = np.where(weight_acc > 0, rate_acc / weight_acc, 0.0)
    return rate


# Top-level entry point

def run_contact_map_analysis(ref_traj, lengths=None, out_dir=None,
                             selected=None):
    """
    Compute and save all contact-map outputs.

    Returns
    -------
    cmaps      : dict {length: prob_map}
    native_map : (n_res, n_res) bool array
    """
    if out_dir is None:
        out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(_CMAP_DIR, exist_ok=True)

    print("  [ContactMaps] Computing native contact map …")
    native_map = compute_native_contact_map(ref_traj)
    native_path = os.path.join(_CMAP_DIR, "contact_map_native.png")
    plot_contact_map(native_map.astype(float), "Native Contact Map",
                     native_path, cmap="Greys")

    print("  [ContactMaps] Computing fragment contact probability maps …")
    if lengths is None:
        lengths = list(range(MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH + 1))
    cmaps = compute_all_contact_maps(ref_traj, lengths=lengths)
    if not cmaps:
        print("  [ContactMaps] No ensembles found — skipping.")
        return {}, native_map

    # Plot selected fragment maps
    if selected is None:
        all_lengths = sorted(cmaps.keys())
        n = len(all_lengths)
        indices = np.linspace(0, n - 1, min(6, n), dtype=int)
        selected = [all_lengths[i] for i in indices]

    for length in selected:
        pmap = cmaps.get(length)
        if pmap is None:
            continue
        plot_contact_map(pmap,
                         f"Contact Probability — frag{length:03d}",
                         os.path.join(_CMAP_DIR, f"contact_map_frag{length:03d}.png"))

    # Contact formation
    print("  [ContactMaps] Tracking contact formation …")
    formation_df = compute_contact_formation(cmaps, native_map, sorted(cmaps.keys()))
    csv_path = os.path.join(out_dir, "contact_formation.csv")
    formation_df.to_csv(csv_path, index=False)
    print(f"  [ContactMaps] Formation CSV saved to {csv_path}")

    plot_contact_formation(formation_df,
                           out_path=os.path.join(out_dir, "contact_formation.png"))

    print(f"  [ContactMaps] Done — {len(cmaps)} maps computed.")
    return cmaps, native_map
