"""
Structural analysis: RMSD, Q-value (native contacts), hydrophobic core compactness.

Three structural quality metrics are computed for the single OpenFold 3 prediction
per fragment:

  RMSD       — backbone RMSD (nm) against the reference structure.
  Q-value    — fraction of native contacts preserved in the prediction.
  Hydro Rg   — radius of gyration (nm) of hydrophobic sidechain heavy atoms
               (LIVFWMA). Lower = more compact hydrophobic core.

One prediction per fragment (no bootstrap CI, no ensemble averaging).
"""
import os
import glob
import warnings
import numpy as np
import h5py
import mdtraj as md

from config import (
    CONTACT_CUTOFF_NM, HYDROPHOBIC_AA, STRUCTURES_DIR,
    RAW_METRICS_DIR,
    EMERGENT_CONTACT_CONSISTENCY,
    PACKING_RADIUS_NM, MAX_SASA_PER_AA,
    LOCAL_PACKING_WEIGHT, HYDRO_BURIAL_WEIGHT,
    PLDDT_WEIGHT, PLDDT_GRADIENT_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_traj(pdb_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return md.load(pdb_path)
        except ValueError as e:
            if "MODELs" not in str(e) and "ATOMs" not in str(e):
                raise
        # NMR multi-model PDB with unequal atom counts — extract MODEL 1 into a temp file
        import tempfile, os
        with open(pdb_path) as f:
            lines = f.readlines()
        model1 = []
        in_m1 = False
        for line in lines:
            rec = line[:6].strip()
            if rec == "MODEL":
                in_m1 = True
                continue
            if rec == "ENDMDL":
                break
            model1.append(line)
        if not any(l.startswith("END") for l in model1[-3:]):
            model1.append("END\n")
        fd, tmp = tempfile.mkstemp(suffix=".pdb")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.writelines(model1)
            traj = md.load(tmp)
        finally:
            os.unlink(tmp)
        return traj


def _backbone_atoms(traj):
    return traj.topology.select("backbone")


def _ca_atoms(traj):
    return traj.topology.select("name CA")


# ---------------------------------------------------------------------------
# Native contacts
# ---------------------------------------------------------------------------

def compute_native_contacts(ref_traj, scheme="closest-heavy"):
    """
    Identify native contacts in the reference structure.
    Returns (native_pairs, total_native).
    """
    distances, residue_pairs = md.compute_contacts(ref_traj, scheme=scheme)
    d0 = distances[0]
    native_mask = d0 <= CONTACT_CUTOFF_NM
    native_pairs = residue_pairs[native_mask]
    return native_pairs, int(native_mask.sum())


def compute_relative_contact_order(ref_traj, scheme="closest-heavy"):
    """
    Relative contact order RCO = mean(|i-j|) / N over native contacts.

    Plaxco, Simons & Baker (1998) J Mol Biol 277:985.  RCO correlates with
    log(kf) (r = -0.81) and indicates how local or non-local the native contact
    topology is. It does not identify folding-nucleus residues by itself:
    low RCO → predominantly local contact topology;
    high RCO → greater long-range contact contribution.

    Returns float in [0, 1].
    """
    native_pairs, _ = compute_native_contacts(ref_traj, scheme=scheme)
    n_res = ref_traj.topology.n_residues
    if len(native_pairs) == 0 or n_res <= 1:
        return 0.0
    seq_sep = np.abs(native_pairs[:, 0] - native_pairs[:, 1]).astype(float)
    return float(seq_sep.mean()) / n_res


def compute_reference_ss(ref_traj):
    """
    Compute per-residue secondary structure (DSSP) from the reference structure.

    Returns a 1-D str array of length n_residues:
      'H' = α-helix, 'E' = β-strand, 'C' = coil/other.
    Used by C2/MANS to determine SS-aware pLDDT thresholds per fragment.
    """
    try:
        dssp = md.compute_dssp(ref_traj, simplified=True)[0]  # shape (n_residues,)
        return dssp  # 'H', 'E', 'C'
    except Exception:
        n = ref_traj.topology.n_residues
        return np.array(['C'] * n)


def compute_per_residue_q(model_traj, native_pairs, cutoff=None, scheme="closest-heavy"):
    """
    Per-residue Q: for each residue i, fraction of residue i's native contacts
    that are formed on average across all frames in model_traj.
    """
    if cutoff is None:
        cutoff = CONTACT_CUTOFF_NM
    n_res = model_traj.topology.n_residues
    valid_mask = (native_pairs[:, 0] < n_res) & (native_pairs[:, 1] < n_res)
    valid_pairs = native_pairs[valid_mask]
    if len(valid_pairs) == 0:
        return np.zeros(n_res, dtype=float)
    distances, _ = md.compute_contacts(model_traj, contacts=valid_pairs, scheme=scheme)
    contact_formed = (distances <= cutoff).mean(axis=0)
    q_per_res = np.zeros(n_res, dtype=float)
    count_per_res = np.zeros(n_res, dtype=float)
    np.add.at(q_per_res,    valid_pairs[:, 0], contact_formed)
    np.add.at(q_per_res,    valid_pairs[:, 1], contact_formed)
    np.add.at(count_per_res, valid_pairs[:, 0], 1.0)
    np.add.at(count_per_res, valid_pairs[:, 1], 1.0)
    with np.errstate(invalid="ignore"):
        q_per_res = np.where(count_per_res > 0, q_per_res / count_per_res, 0.0)
    return q_per_res


def compute_per_frame_per_residue_contacts(model_traj, native_pairs, cutoff=None):
    """
    Compute per-frame, per-residue contact fractions and global Q.

    Used for cooperativity correlation computation (coop_corr_i = r(q_i, Q_global)).

    Parameters
    ----------
    model_traj   : mdtraj trajectory
    native_pairs : (n_pairs, 2) int array of native contact residue pairs
    cutoff       : contact distance threshold (nm); defaults to CONTACT_CUTOFF_NM

    Returns
    -------
    q_matrix : (n_frames, n_res) float array
        Per-frame, per-residue contact fraction
    global_q : (n_frames,) float array
        Global Q (fraction of native contacts formed) per frame
    """
    if cutoff is None:
        cutoff = CONTACT_CUTOFF_NM

    n_res = model_traj.topology.n_residues
    n_frames = model_traj.n_frames

    # Compute distances for all native pairs
    distances, _ = md.compute_contacts(model_traj, contacts=native_pairs,
                                       scheme='closest-heavy')
    # (n_frames, n_pairs)

    # Global Q per frame
    formed = (distances <= cutoff)  # (n_frames, n_pairs) bool
    global_q = formed.mean(axis=1)  # (n_frames,)

    # Per-residue per-frame contact fractions
    q_matrix = np.zeros((n_frames, n_res), dtype=float)
    count_matrix = np.zeros((n_frames, n_res), dtype=float)

    for pair_idx, (ri, rj) in enumerate(native_pairs):
        if ri < n_res and rj < n_res:
            q_matrix[:, ri] += formed[:, pair_idx]
            q_matrix[:, rj] += formed[:, pair_idx]
            count_matrix[:, ri] += 1.0
            count_matrix[:, rj] += 1.0

    with np.errstate(invalid="ignore"):
        q_matrix = np.where(count_matrix > 0, q_matrix / count_matrix, 0.0)

    return q_matrix, global_q


def compute_emergent_contact_degree(model_traj, cutoff=None,
                                    consistency_threshold=None):
    """
    Per-residue count of emergent contacts discovered ab initio.
    """
    if cutoff is None:
        cutoff = CONTACT_CUTOFF_NM
    if consistency_threshold is None:
        consistency_threshold = EMERGENT_CONTACT_CONSISTENCY

    distances, pairs = md.compute_contacts(model_traj, contacts='all',
                                           scheme='closest-heavy')
    contact_prob = (distances <= cutoff).mean(axis=0)
    consistent = contact_prob >= consistency_threshold

    n_res = model_traj.topology.n_residues
    degree = np.zeros(n_res, dtype=int)
    consistent_pairs = pairs[consistent]
    if len(consistent_pairs) > 0:
        np.add.at(degree, consistent_pairs[:, 0], 1)
        np.add.at(degree, consistent_pairs[:, 1], 1)
    return degree


def compute_q_value(model_traj, native_pairs, total_native, scheme="closest-heavy"):
    """Q-value computation.

    Returns
    -------
    q_global : float  Fraction of all native contacts (total_native) that are formed.
    q_local  : float  Fraction of native contacts available in this fragment that are formed.
    """
    if total_native == 0:
        return 0.0, 0.0
    n_res = model_traj.topology.n_residues
    valid_mask = (native_pairs[:, 0] < n_res) & (native_pairs[:, 1] < n_res)
    valid_pairs = native_pairs[valid_mask]
    n_valid = len(valid_pairs)
    if n_valid == 0:
        return 0.0, 0.0
    distances, _ = md.compute_contacts(model_traj, contacts=valid_pairs, scheme=scheme)
    contacts_per_frame = (distances <= CONTACT_CUTOFF_NM).mean(axis=1)
    q_global = float((contacts_per_frame * n_valid / total_native).mean())
    q_local  = float(contacts_per_frame.mean())
    return q_global, q_local


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def compute_rmsd(ref_traj, model_traj):
    """Backbone RMSD (nm). Falls back to CA-only on atom count mismatch."""
    ref_bb = _backbone_atoms(ref_traj)
    mdl_bb = _backbone_atoms(model_traj)

    if len(ref_bb) == len(mdl_bb):
        atom_indices = mdl_bb
        ref_indices  = ref_bb
    else:
        ref_ca = _ca_atoms(ref_traj)
        mdl_ca = _ca_atoms(model_traj)
        n = min(len(ref_ca), len(mdl_ca))
        ref_indices  = ref_ca[:n]
        atom_indices = mdl_ca[:n]

    rmsd_values = md.rmsd(
        model_traj, ref_traj,
        frame=0,
        atom_indices=atom_indices,
        ref_atom_indices=ref_indices,
    )
    return float(rmsd_values.mean())


# ---------------------------------------------------------------------------
# Hydrophobic core
# ---------------------------------------------------------------------------

def _hydrophobic_sidechain_atoms(traj):
    BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}
    top = traj.topology
    indices = []
    for atom in top.atoms:
        if (atom.residue.code in HYDROPHOBIC_AA
                and atom.name not in BACKBONE_NAMES
                and atom.element.symbol != "H"):
            indices.append(atom.index)
    return np.array(indices, dtype=int)


def compute_hydrophobic_rg(traj):
    """Radius of gyration (nm) of hydrophobic sidechain heavy atoms."""
    hydro_idx = _hydrophobic_sidechain_atoms(traj)
    if len(hydro_idx) == 0:
        return float("nan")
    sub = traj.atom_slice(hydro_idx)
    return float(md.compute_rg(sub).mean())


# ---------------------------------------------------------------------------
# Phase 3 — Enhancement 1: Local packing + hydrophobic burial
# ---------------------------------------------------------------------------

def compute_local_packing(traj, radius=None):
    """
    Per-residue local packing density.

    For each residue, count the number of heavy atoms within `radius` nm
    that belong to OTHER residues (excludes hydrogen, excludes own residue).

    Returns
    -------
    packing : (n_res,) int array — neighbour heavy atom count
    """
    if radius is None:
        radius = PACKING_RADIUS_NM

    top = traj.topology
    n_res = top.n_residues

    # Heavy atom indices only
    heavy_idx = np.array([a.index for a in top.atoms
                          if a.element.symbol != "H"], dtype=int)
    if len(heavy_idx) == 0:
        return np.zeros(n_res, dtype=int)

    heavy_sub = traj.atom_slice(heavy_idx)
    # Map new atom index → residue index in original topology
    heavy_res = np.array([top.atom(i).residue.index for i in heavy_idx], dtype=int)

    # Pairwise distances between all heavy atoms (first frame only)
    xyz = heavy_sub.xyz[0]          # (n_heavy, 3) in nm
    packing = np.zeros(n_res, dtype=int)

    for ri in range(n_res):
        own_mask = heavy_res == ri
        other_mask = ~own_mask
        if not np.any(own_mask) or not np.any(other_mask):
            continue
        own_xyz   = xyz[own_mask]   # (n_own, 3)
        other_xyz = xyz[other_mask] # (n_other, 3)
        # Min distance from each other atom to any own atom
        diffs = other_xyz[:, None, :] - own_xyz[None, :, :]  # (n_other, n_own, 3)
        dists = np.sqrt((diffs ** 2).sum(axis=-1)).min(axis=1)  # (n_other,)
        packing[ri] = int((dists < radius).sum())

    return packing


def compute_hydrophobic_burial(traj):
    """
    Per-residue hydrophobic burial fraction.

    For hydrophobic residues: burial = 1 - (SASA / SASA_ref_max).
    For non-hydrophobic residues: 0.0.

    SASA computed with md.shrake_rupley() (probe radius 0.14 nm).

    Returns
    -------
    burial : (n_res,) float array in [0, 1]
    """
    top = traj.topology
    n_res = top.n_residues
    burial = np.zeros(n_res, dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # sasa shape: (n_frames, n_atoms) in nm²
            sasa = md.shrake_rupley(traj, mode="residue")   # (n_frames, n_res)
        sasa_per_res = sasa[0]  # first (and only) frame
    except Exception:
        return burial

    for ri in range(n_res):
        res = top.residue(ri)
        res_name = res.name.upper()
        if res_name not in HYDROPHOBIC_AA and res.code not in HYDROPHOBIC_AA:
            continue
        ref_max = MAX_SASA_PER_AA.get(res_name, None)
        if ref_max is None or ref_max <= 0:
            continue
        burial[ri] = max(0.0, 1.0 - sasa_per_res[ri] / ref_max)

    return burial


# ---------------------------------------------------------------------------
# Structure loading
# ---------------------------------------------------------------------------

def load_ensemble(length, structures_dir=None):
    """Load all PDBs for frag{length:02d}. Returns list of trajectories."""
    if structures_dir is None:
        structures_dir = STRUCTURES_DIR
    out_dir = os.path.join(structures_dir, f"frag{length:02d}")
    pdb_paths = sorted(glob.glob(os.path.join(out_dir, "*.pdb")))
    if not pdb_paths:
        return []
    trajs = []
    for p in pdb_paths:
        try:
            trajs.append(_load_traj(p))
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return trajs


def get_pdb_paths(length, structures_dir=None):
    """Return sorted list of decoy PDB paths for frag{length:02d}."""
    if structures_dir is None:
        structures_dir = STRUCTURES_DIR
    out_dir = os.path.join(structures_dir, f"frag{length:02d}")
    return sorted(glob.glob(os.path.join(out_dir, "*.pdb")))


# ---------------------------------------------------------------------------
# Raw metrics cache (used by advanced analysis modules)
# ---------------------------------------------------------------------------

def save_raw_metrics(length, rmsds, qs, hydros, out_dir=None):
    """Save per-fragment metric arrays to a compressed HDF5 file."""
    if out_dir is None:
        out_dir = RAW_METRICS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw_metrics.h5")
    key = f"frag{length:03d}"
    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        g.create_dataset("rmsds",  data=rmsds,  compression="gzip")
        g.create_dataset("qs",     data=qs,     compression="gzip")
        g.create_dataset("hydros", data=hydros, compression="gzip")


def load_raw_metrics(length, raw_dir=None):
    """Load per-fragment metric arrays from the HDF5 cache. Returns (rmsds, qs, hydros)."""
    if raw_dir is None:
        raw_dir = RAW_METRICS_DIR
    path = os.path.join(raw_dir, "raw_metrics.h5")
    key = f"frag{length:03d}"
    with h5py.File(path, "r") as f:
        g = f[key]
        return np.array(g["rmsds"]), np.array(g["qs"]), np.array(g["hydros"])


# ---------------------------------------------------------------------------
# Fragment analysis — single prediction per fragment
# ---------------------------------------------------------------------------

def _extract_plddt_safe(pdb_path):
    """Return (plddt_mean, plddt_per_residue) or (nan, None) on failure."""
    try:
        from run_esmfold import extract_plddt
        return extract_plddt(pdb_path)
    except Exception:
        return float("nan"), None


def _dominant_ss(ref_ss, start, end):
    """Return dominant SS type ('H', 'E', 'C') in ref_ss[start:end]."""
    if ref_ss is None or end <= start:
        return "C"
    slice_ss = ref_ss[start:end]
    if len(slice_ss) == 0:
        return "C"
    for ss_type in ("E", "H", "C"):
        if np.sum(slice_ss == ss_type) >= len(slice_ss) * 0.5:
            return ss_type
    return "C"


def analyze_fragment(length, ref_traj, native_pairs, total_native,
                     save_raw=False, raw_dir=None, direction="N",
                     alignment=None, ref_ss=None):
    """
    Analyse the single OpenFold 3 prediction for a given fragment length.

    Loads decoy_00000.pdb (the single prediction), computes RMSD, Q-value
    and hydrophobic Rg against the reference structure.

    Parameters
    ----------
    direction : 'N' (N-terminal, default) or 'C' (C-terminal).
    alignment : dict or None
        If provided (UniProt-based fragments), maps UniProt fragment positions
        to PDB reference positions.  Dict must have keys 'pdb_offset' and
        'pdb_n_res' (from core.alignment.compute_alignment).
        When None, the legacy behaviour (simple N-terminal PDB slice) is used.

    Returns a dict or None if no structure is found.
    """
    ref_res = ref_traj.topology.n_residues

    if direction == "C":
        # C-terminal fragments: always use PDB-only logic (no UniProt offset)
        if length > ref_res:
            length = ref_res
        start_resid = ref_res - length
        ref_slice = ref_traj.atom_slice(
            ref_traj.topology.select(f"resid >= {start_resid}")
        )
        ensemble_tag = f"fragC{length:02d}"
        ss_dominant = _dominant_ss(ref_ss, start_resid, ref_res)
    elif alignment is not None:
        # N-terminal fragment from UniProt domain sequence
        pdb_offset = alignment["pdb_offset"]
        pdb_n_res  = alignment["pdb_n_res"]
        K = max(0, min(length - pdb_offset, pdb_n_res))
        ensemble_tag = f"frag{length:02d}"

        out_dir = os.path.join(STRUCTURES_DIR, ensemble_tag)
        primary = os.path.join(out_dir, "decoy_00000.pdb")
        if os.path.exists(primary):
            pdb_path = primary
        else:
            candidates = sorted(glob.glob(os.path.join(out_dir, "*.pdb")))
            if not candidates:
                return None
            pdb_path = candidates[0]

        try:
            traj = _load_traj(pdb_path)
        except Exception as e:
            print(f"  Warning: could not load {pdb_path}: {e}")
            return None

        plddt_mean, plddt_per_res = _extract_plddt_safe(pdb_path)
        packing_mean = 0.0
        burial_mean  = 0.0

        if K <= 0:
            # Fragment is entirely in the N-terminal region missing from PDB
            result = {
                "length":           length,
                "rmsd_mean":        float("nan"),
                "q_mean":           0.0,
                "q_local_mean":     0.0,
                "hydro_rg_mean":    float("nan"),
                "rmsd_std":         0.0,
                "q_std":            0.0,
                "q_local_std":      0.0,
                "hydro_rg_std":     0.0,
                "n_decoys":         1,
                "plddt_mean":       plddt_mean,
                "packing_mean":     packing_mean,
                "hydro_burial_mean": burial_mean,
            }
            if plddt_per_res is not None:
                result["plddt_per_residue"] = plddt_per_res
            return result

        # Slice reference: first K PDB residues
        ref_slice = ref_traj.atom_slice(
            ref_traj.topology.select(f"resid < {K}")
        )
        # Slice model: residues pdb_offset..pdb_offset+K in the OpenFold prediction
        model_slice = traj.atom_slice(
            traj.topology.select(f"resid >= {pdb_offset} and resid < {pdb_offset + K}")
        )

        rmsd_val  = compute_rmsd(ref_slice, model_slice)
        q_global, q_local = compute_q_value(model_slice, native_pairs, total_native)
        hydro_val = compute_hydrophobic_rg(model_slice)

        if LOCAL_PACKING_WEIGHT > 0 or HYDRO_BURIAL_WEIGHT > 0:
            packing = compute_local_packing(model_slice)
            burial  = compute_hydrophobic_burial(model_slice)
            packing_mean = float(packing.mean()) if len(packing) > 0 else 0.0
            burial_mean  = float(burial.mean())  if len(burial)  > 0 else 0.0

        if save_raw:
            save_raw_metrics(
                length,
                np.array([rmsd_val],  dtype=float),
                np.array([q_global],  dtype=float),
                np.array([hydro_val], dtype=float),
                out_dir=raw_dir,
            )

        result = {
            "length":           length,
            "rmsd_mean":        rmsd_val,
            "q_mean":           q_global,
            "q_local_mean":     q_local,
            "hydro_rg_mean":    hydro_val,
            "rmsd_std":         0.0,
            "q_std":            0.0,
            "q_local_std":      0.0,
            "hydro_rg_std":     0.0,
            "n_decoys":         1,
            "plddt_mean":       plddt_mean,
            "packing_mean":     packing_mean,
            "hydro_burial_mean": burial_mean,
            "ss_dominant":      _dominant_ss(ref_ss, int(pdb_offset), int(pdb_offset + K)),
        }
        if plddt_per_res is not None:
            result["plddt_per_residue"] = plddt_per_res
        return result
    else:
        # Legacy: N-terminal PDB-only slice
        if length > ref_res:
            length = ref_res
        ref_slice = ref_traj.atom_slice(
            ref_traj.topology.select(f"resid < {length}")
        )
        ensemble_tag = f"frag{length:02d}"
        ss_dominant = _dominant_ss(ref_ss, 0, length)

    out_dir = os.path.join(STRUCTURES_DIR, ensemble_tag)

    # Prefer decoy_00000.pdb; fall back to any pdb in directory
    primary = os.path.join(out_dir, "decoy_00000.pdb")
    if os.path.exists(primary):
        pdb_path = primary
    else:
        candidates = sorted(glob.glob(os.path.join(out_dir, "*.pdb")))
        if not candidates:
            return None
        pdb_path = candidates[0]

    try:
        traj = _load_traj(pdb_path)
    except Exception as e:
        print(f"  Warning: could not load {pdb_path}: {e}")
        return None

    rmsd_val  = compute_rmsd(ref_slice, traj)
    q_global, q_local = compute_q_value(traj, native_pairs, total_native)
    hydro_val = compute_hydrophobic_rg(traj)

    # Phase 3 — Enhancement 2: pLDDT
    plddt_mean, plddt_per_res = _extract_plddt_safe(pdb_path)

    # Phase 3 — Enhancement 1: local packing + hydrophobic burial (only if enabled)
    if LOCAL_PACKING_WEIGHT > 0 or HYDRO_BURIAL_WEIGHT > 0:
        packing = compute_local_packing(traj)
        burial  = compute_hydrophobic_burial(traj)
        packing_mean = float(packing.mean()) if len(packing) > 0 else 0.0
        burial_mean  = float(burial.mean())  if len(burial)  > 0 else 0.0
    else:
        packing_mean = 0.0
        burial_mean  = 0.0

    if save_raw:
        save_raw_metrics(
            length,
            np.array([rmsd_val],  dtype=float),
            np.array([q_global],  dtype=float),
            np.array([hydro_val], dtype=float),
            out_dir=raw_dir,
        )

    result = {
        "length":        length,
        # Core metrics (single prediction)
        "rmsd_mean":     rmsd_val,
        "q_mean":        q_global,
        "q_local_mean":  q_local,
        "hydro_rg_mean": hydro_val,
        # Zero std (single structure)
        "rmsd_std":      0.0,
        "q_std":         0.0,
        "q_local_std":   0.0,
        "hydro_rg_std":  0.0,
        "n_decoys":      1,
        # Phase 3 additions
        "plddt_mean":       plddt_mean,
        "packing_mean":     packing_mean,
        "hydro_burial_mean": burial_mean,
        # C2/MANS: dominant secondary structure type for this fragment
        "ss_dominant":      ss_dominant,
    }
    # Store per-residue pLDDT array if available (used by gradient scoring)
    if plddt_per_res is not None:
        result["plddt_per_residue"] = plddt_per_res
    return result


def analyze_sliding_fragment(start, window_size, ref_traj, native_pairs, total_native,
                              structures_dir=None):
    """
    Analyse a single sliding/internal fragment prediction.

    The fragment covers residues [start : start+window_size] of the full sequence.
    The reference slice for RMSD is ref_traj residues [start : start+window_size].

    Parameters
    ----------
    start       : int — 0-based start residue index
    window_size : int — fragment length in residues
    ref_traj    : md.Trajectory — full-length reference structure
    native_pairs, total_native — from compute_native_contacts(ref_traj)
    structures_dir : str or None

    Returns dict or None if no structure found.
    """
    if structures_dir is None:
        structures_dir = STRUCTURES_DIR

    ref_res = ref_traj.topology.n_residues
    end = min(start + window_size, ref_res)
    ref_slice = ref_traj.atom_slice(
        ref_traj.topology.select(f"resid >= {start} and resid < {end}")
    )

    tag = f"fragS{start:03d}_W{window_size:02d}"
    out_dir = os.path.join(structures_dir, tag)
    primary = os.path.join(out_dir, "decoy_00000.pdb")
    if os.path.exists(primary):
        pdb_path = primary
    else:
        candidates = sorted(glob.glob(os.path.join(out_dir, "*.pdb")))
        if not candidates:
            return None
        pdb_path = candidates[0]

    try:
        traj = _load_traj(pdb_path)
    except Exception as e:
        print(f"  Warning: could not load {pdb_path}: {e}")
        return None

    rmsd_val          = compute_rmsd(ref_slice, traj)
    q_global, q_local = compute_q_value(traj, native_pairs, total_native)
    hydro_val         = compute_hydrophobic_rg(traj)
    plddt_mean, plddt_per_res = _extract_plddt_safe(pdb_path)

    result = {
        "start":       start,
        "window_size": window_size,
        "length":      window_size,
        "rmsd_mean":   rmsd_val,
        "q_mean":      q_global,
        "q_local_mean": q_local,
        "hydro_rg_mean": hydro_val,
        "rmsd_std":    0.0,
        "q_std":       0.0,
        "q_local_std": 0.0,
        "hydro_rg_std": 0.0,
        "n_decoys":    1,
        "plddt_mean":  plddt_mean,
    }
    if plddt_per_res is not None:
        result["plddt_per_residue"] = plddt_per_res
    return result
