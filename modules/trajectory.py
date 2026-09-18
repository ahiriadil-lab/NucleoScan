"""
trajectory.py — Fragment-length structure-series builder.

Selects the best decoy (lowest Cα RMSD to native) for each fragment length,
concatenates them into a multi-model PDB file and generates a PyMOL script for
animated inspection. Each state is an independent prediction at a different
prefix length; the state sequence is not a kinetic trajectory or folding time.

Main functions
--------------
select_best_decoy(fragment_dir, ref_traj)
    Find the decoy PDB with the lowest Cα RMSD vs the native structure.

build_progressive_trajectory(structures_dir, ref_pdb, out_pdb)
    One MODEL block per fragment length (best decoy).  Returns out_pdb path.

generate_pymol_script(trajectory_pdb, ref_pdb, out_pml)
    Write a .pml script that loads + animates the trajectory.
"""

import glob
import os
import warnings

import mdtraj as md
import numpy as np


# Helpers

def _load(path: str) -> md.Trajectory:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.load(path)


def _ca_rmsd(traj: md.Trajectory, ref_traj: md.Trajectory) -> float:
    """Cα RMSD (nm) between the first frames of traj and ref_traj."""
    ca_traj = traj.topology.select("name CA")
    ca_ref  = ref_traj.topology.select("name CA")
    n = min(len(ca_traj), len(ca_ref))
    if n < 2:
        return float("nan")
    sub     = traj.atom_slice(ca_traj[:n])
    ref_sub = ref_traj.atom_slice(ca_ref[:n])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(md.rmsd(sub, ref_sub, frame=0)[0])


# Best-decoy selection

def select_best_decoy(fragment_dir: str, ref_traj: md.Trajectory) -> str:
    """
    Return the path of the decoy PDB with the lowest Cα RMSD vs ref_traj.
    Returns None if no decoy PDBs are found in fragment_dir.
    """
    pdb_paths = sorted(glob.glob(os.path.join(fragment_dir, "decoy_*.pdb")))
    if not pdb_paths:
        return None

    best_path = None
    best_rmsd = float("inf")
    for path in pdb_paths:
        try:
            traj = _load(path)
            rmsd = _ca_rmsd(traj, ref_traj)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_path = path
        except Exception:
            continue

    return best_path


# Multi-model PDB fragment series

def build_progressive_trajectory(structures_dir: str,
                                  ref_pdb: str,
                                  out_pdb: str) -> str:
    """
    Build a multi-model PDB fragment series: one MODEL per prefix length, each
    being the best-RMSD decoy. Skips lengths with no decoys. MODEL order is
    sequence-context order, not elapsed folding time.

    Parameters
    ----------
    structures_dir : directory containing frag{length:03d}/ subdirectories
    ref_pdb        : native PDB used as RMSD reference
    out_pdb        : path for the output multi-model PDB

    Returns
    -------
    out_pdb if at least one model was written, else None.
    """
    ref_traj  = _load(ref_pdb)
    frag_dirs = sorted(glob.glob(os.path.join(structures_dir, "frag[0-9]*")))

    if not frag_dirs:
        print(f"  [trajectory] No fragment directories found in {structures_dir}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)

    models_written = 0
    with open(out_pdb, "w") as out_fh:
        for frag_dir in frag_dirs:
            frag_name  = os.path.basename(frag_dir)
            best_path  = select_best_decoy(frag_dir, ref_traj)
            if best_path is None:
                continue

            try:
                with open(best_path) as pdb_fh:
                    lines = pdb_fh.readlines()
            except Exception as e:
                print(f"  [trajectory] Cannot read {best_path}: {e}")
                continue

            out_fh.write(f"MODEL     {models_written + 1:4d}\n")
            out_fh.write(f"REMARK    Fragment {frag_name} — "
                         f"best decoy: {os.path.basename(best_path)}\n")
            for line in lines:
                if line.startswith(("MODEL", "ENDMDL", "END\n")):
                    continue
                out_fh.write(line)
            out_fh.write("ENDMDL\n")
            models_written += 1

    if models_written == 0:
        print("  [trajectory] No models written.")
        return None

    print(f"  [trajectory] {models_written} models written to {out_pdb}")
    return out_pdb


# PyMOL script for the fragment series

def generate_pymol_script(trajectory_pdb: str,
                           ref_pdb: str,
                           out_pml: str) -> str:
    """
    Write a PyMOL script that loads the multi-model fragment series and native
    reference, aligns them, and prepares an animation.

    Parameters
    ----------
    trajectory_pdb : multi-model PDB from build_progressive_trajectory
    ref_pdb        : native PDB
    out_pml        : output .pml script path

    Returns
    -------
    out_pml path.
    """
    traj_abs = os.path.abspath(trajectory_pdb)
    ref_abs  = os.path.abspath(ref_pdb)
    traj_name = os.path.splitext(os.path.basename(trajectory_pdb))[0]
    ref_name  = os.path.splitext(os.path.basename(ref_pdb))[0]

    script = f"""\
# PyMOL script: independently predicted prefix-length structure series
# State order is fragment length, not folding time.
# Generated by modules/trajectory.py
# Usage: pymol {os.path.basename(out_pml)}

load {traj_abs}, {traj_name}
load {ref_abs}, {ref_name}

# Representation
hide everything
show cartoon, {traj_name}
show cartoon, {ref_name}

# Colour native grey, fragment series by state (blue → red spectrum)
color grey70, {ref_name}
spectrum count, blue_white_red, {traj_name}

# Align all fragment-series states to the native structure
align {traj_name}, {ref_name}
intra_fit {traj_name}, 1

# Animation
set all_states, on
set movie_fps, 4
set ray_opaque_background, off

# Initial view
orient {ref_name}
zoom {ref_name}

print "Ready. States show prefix length, not folding time. Press Play or type 'mplay'."
"""

    os.makedirs(os.path.dirname(os.path.abspath(out_pml)), exist_ok=True)
    with open(out_pml, "w") as fh:
        fh.write(script)

    print(f"  [trajectory] PyMOL script written to {out_pml}")
    return out_pml
