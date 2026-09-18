"""
compare_to_native.py — Option B validation for Villin HP36 (1VII)

Computes for each fragment ensemble:
  - Per-decoy RMSD (global) vs native
  - Fraction of decoys with RMSD < NATIVE_RMSD_THRESHOLD (native-like)
  - Per-helix RMSD for the full protein (frag036), using HP36 helix definitions

HP36 secondary structure (experimental, PDB 1VII):
  α1  = residues  3- 8  (0-indexed 2-7)
  α2  = residues 10-17  (0-indexed 9-16)
  α3  = residues 23-32  (0-indexed 22-31)

Outputs:
  results/{protein}/decoy_rmsd_per_fragment.csv   — RMSD stats per fragment
  results/{protein}/decoy_rmsd_distribution.png   — violin/box plot per fragment
  results/{protein}/helix_rmsd.csv                — RMSD per helix for frag036
  results/{protein}/helix_rmsd.png                — bar chart of helix RMSDs

Usage:
  python compare_to_native.py [--protein Villin_HP36] [--results-dir results]
"""
import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

# Constants

NATIVE_RMSD_THRESHOLD = 0.3   # nm — standard "native-like" cutoff
HELIX_DEFINITIONS = {         # 1-indexed residue ranges (inclusive) for HP36
    "alpha1 (3-8)":   (3,  8),
    "alpha2 (10-17)": (10, 17),
    "alpha3 (23-32)": (23, 32),
}


# Helpers

def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.load(path)


def _ca_indices(traj, res_start_0idx, res_end_0idx_inclusive):
    """Return Cα atom indices for a residue range (0-indexed, inclusive end)."""
    sel = traj.topology.select(
        f"name CA and resid >= {res_start_0idx} and resid <= {res_end_0idx_inclusive}"
    )
    return sel


def rmsd_global(native_slice, decoy):
    """Backbone RMSD between decoy and the appropriate native slice, in nm."""
    ref_ca  = native_slice.topology.select("name CA")
    mdl_ca  = decoy.topology.select("name CA")
    n = min(len(ref_ca), len(mdl_ca))
    if n < 2:
        return float("nan")
    return float(md.rmsd(decoy, native_slice, frame=0,
                          atom_indices=mdl_ca[:n],
                          ref_atom_indices=ref_ca[:n]).mean())


def rmsd_helix(native, decoy, res_start_1idx, res_end_1idx):
    """
    Backbone Cα RMSD restricted to a single helix, in nm.

    Parameters
    ----------
    native : full native trajectory (36 residues)
    decoy  : full-protein decoy trajectory
    res_start_1idx, res_end_1idx : 1-indexed residue range (inclusive)
    """
    s = res_start_1idx - 1  # 0-indexed start
    e = res_end_1idx   - 1  # 0-indexed end (inclusive)

    ref_ca = _ca_indices(native, s, e)
    mdl_ca = _ca_indices(decoy,  s, e)
    n = min(len(ref_ca), len(mdl_ca))
    if n < 2:
        return float("nan")
    return float(md.rmsd(decoy, native, frame=0,
                          atom_indices=mdl_ca[:n],
                          ref_atom_indices=ref_ca[:n]).mean())


# Per-fragment analysis

def analyze_fragment(length, frag_dir, native):
    """
    Compute per-decoy RMSDs for all decoys in *frag_dir*.

    Returns a dict with aggregate statistics, or None if no structures found.
    """
    pattern = os.path.join(frag_dir, "*.pdb")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        return None

    # Native slice restricted to the first *length* residues
    n_native = native.topology.n_residues
    n = min(length, n_native)
    native_slice = native.atom_slice(
        native.topology.select(f"resid < {n}")
    )

    rmsds = []
    for p in paths:
        try:
            decoy = _load(p)
        except Exception:
            continue
        for fi in range(decoy.n_frames):
            frame = decoy.slice(fi)
            rmsds.append(rmsd_global(native_slice, frame))

    if not rmsds:
        return None

    rmsds  = np.array(rmsds, dtype=float)
    rmsds  = rmsds[~np.isnan(rmsds)]

    frac_native = float((rmsds < NATIVE_RMSD_THRESHOLD).mean())

    return {
        "fragment_length":   length,
        "n_decoys":          len(rmsds),
        "rmsd_mean":         float(np.mean(rmsds)),
        "rmsd_std":          float(np.std(rmsds)),
        "rmsd_min":          float(np.min(rmsds)),
        "rmsd_max":          float(np.max(rmsds)),
        "rmsd_median":       float(np.median(rmsds)),
        "frac_native_like":  frac_native,
        "n_native_like":     int((rmsds < NATIVE_RMSD_THRESHOLD).sum()),
        "rmsds":             rmsds,    # kept for plotting only
    }


# Per-helix analysis  (full protein only)

def analyze_helices(frag36_dir, native):
    """
    For each decoy in frag036, compute RMSD restricted to each helix.

    Returns a DataFrame with columns: decoy, helix, rmsd
    """
    paths  = sorted(glob.glob(os.path.join(frag36_dir, "*.pdb")))
    if not paths:
        return pd.DataFrame()

    n_native = native.topology.n_residues

    rows = []
    for idx, p in enumerate(paths):
        try:
            decoy = _load(p)
        except Exception:
            continue
        for fi in range(decoy.n_frames):
            frame = decoy.slice(fi)
            if frame.topology.n_residues < n_native:
                continue  # skip incomplete decoys
            for hname, (h_start, h_end) in HELIX_DEFINITIONS.items():
                r = rmsd_helix(native, frame, h_start, h_end)
                rows.append({
                    "decoy": f"{idx:04d}_{fi}",
                    "helix": hname,
                    "rmsd":  r,
                })

    return pd.DataFrame(rows)


# Plotting

def plot_rmsd_distribution(records, out_path):
    """
    Violin plot of RMSD distributions per fragment, colour-coded by
    fraction native-like.
    """
    lengths = [r["fragment_length"] for r in records]
    rmsds   = [r["rmsds"] for r in records]
    fracs   = [r["frac_native_like"] for r in records]

    fig, ax = plt.subplots(figsize=(max(10, len(lengths) * 0.45), 5))

    parts = ax.violinplot(rmsds, positions=lengths,
                          widths=0.8, showmedians=True, showextrema=True)

    cmap = plt.cm.RdYlGn
    for i, (body, f) in enumerate(zip(parts["bodies"], fracs)):
        body.set_facecolor(cmap(f))
        body.set_alpha(0.7)

    ax.axhline(NATIVE_RMSD_THRESHOLD, color="steelblue", linestyle="--",
               linewidth=1.2, label=f"Native threshold ({NATIVE_RMSD_THRESHOLD} nm)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Fraction native-like (RMSD < 0.3 nm)", fontsize=9)

    ax.set_xlabel("Fragment length (aa)")
    ax.set_ylabel("RMSD vs native (nm)")
    ax.set_title("Per-decoy RMSD distribution vs 1VII native")
    ax.set_xticks(lengths)
    ax.set_xticklabels(lengths, rotation=90, fontsize=7)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  RMSD distribution plot saved → {out_path}")


def plot_fraction_native(records, out_path):
    """Bar chart of fraction native-like per fragment length."""
    lengths = [r["fragment_length"] for r in records]
    fracs   = [r["frac_native_like"] for r in records]

    fig, ax = plt.subplots(figsize=(max(8, len(lengths) * 0.45), 4))
    colors  = ["#27ae60" if f > 0 else "#e74c3c" for f in fracs]
    ax.bar(lengths, fracs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.01, color="gray", linestyle=":", linewidth=1, label="1% threshold")
    ax.set_xlabel("Fragment length (aa)")
    ax.set_ylabel(f"Fraction decoys with RMSD < {NATIVE_RMSD_THRESHOLD} nm")
    ax.set_title("Native-like fraction per fragment (vs 1VII)")
    ax.set_xticks(lengths)
    ax.set_xticklabels(lengths, rotation=90, fontsize=7)
    ax.set_ylim(0, max(max(fracs) * 1.15, 0.05))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Fraction-native-like plot saved → {out_path}")


def plot_helix_rmsd(helix_df, out_path):
    """Grouped bar chart: mean RMSD per helix, with std error bars."""
    if helix_df.empty:
        return

    stats = (
        helix_df.groupby("helix")["rmsd"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    helices = stats["helix"].tolist()
    means   = stats["mean"].tolist()
    ses     = stats["se"].tolist()

    colors = ["#3498db", "#e67e22", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(helices, means, yerr=ses, capsize=5,
                  color=colors[:len(helices)], edgecolor="white")
    ax.axhline(NATIVE_RMSD_THRESHOLD, color="steelblue", linestyle="--",
               linewidth=1.2, label=f"Native threshold ({NATIVE_RMSD_THRESHOLD} nm)")
    ax.set_ylabel("Mean Cα RMSD vs native (nm)")
    ax.set_title("Per-helix RMSD — full protein (frag036 vs 1VII)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Helix RMSD plot saved → {out_path}")


# Main

def run(protein_name, results_dir, structures_base, pdb_clean_dir):
    out_dir = os.path.join(results_dir, protein_name)
    os.makedirs(out_dir, exist_ok=True)

    # Locate native PDB
    native_candidates = sorted(glob.glob(
        os.path.join(pdb_clean_dir, f"{protein_name}_*_clean.pdb")
    ))
    if not native_candidates:
        raise FileNotFoundError(
            f"No native PDB found for {protein_name} in {pdb_clean_dir}"
        )
    native_path = native_candidates[0]
    print(f"  Native structure : {native_path}")
    native = _load(native_path)
    print(f"  Native residues  : {native.topology.n_residues}")

    structs_dir = os.path.join(structures_base, protein_name)
    if not os.path.isdir(structs_dir):
        raise FileNotFoundError(f"No structures directory: {structs_dir}")

    # Discover all fragment directories
    frag_dirs = sorted(glob.glob(os.path.join(structs_dir, "frag*")))
    if not frag_dirs:
        raise FileNotFoundError(f"No frag* directories in {structs_dir}")

    # Per-fragment RMSD
    records = []
    for fdir in frag_dirs:
        base    = os.path.basename(fdir)           # e.g. "frag013"
        try:
            length = int(base.replace("frag", ""))
        except ValueError:
            continue
        result = analyze_fragment(length, fdir, native)
        if result is None:
            continue
        records.append(result)
        pct = result["frac_native_like"] * 100
        print(f"    frag{length:03d} : mean={result['rmsd_mean']:.3f} nm  "
              f"min={result['rmsd_min']:.3f} nm  "
              f"native-like={pct:.1f}%  (n={result['n_decoys']})")

    if not records:
        print("  No decoys found — aborting.")
        return

    # Save aggregate CSV
    summary_rows = [{k: v for k, v in r.items() if k != "rmsds"} for r in records]
    csv_path = os.path.join(out_dir, "decoy_rmsd_per_fragment.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  Summary CSV → {csv_path}")

    # Plots
    plot_rmsd_distribution(records,
        out_path=os.path.join(out_dir, "decoy_rmsd_distribution.png"))
    plot_fraction_native(records,
        out_path=os.path.join(out_dir, "fraction_native_like.png"))

    # Per-helix RMSD for the full-length model
    frag36_dir = os.path.join(structs_dir, "frag36")
    if not os.path.isdir(frag36_dir):
        # try zero-padded
        frag36_dir = os.path.join(structs_dir, "frag036")

    if os.path.isdir(frag36_dir):
        print("  Computing per-helix RMSD for full protein …")
        helix_df = analyze_helices(frag36_dir, native)
        if not helix_df.empty:
            helix_csv = os.path.join(out_dir, "helix_rmsd.csv")
            helix_df.to_csv(helix_csv, index=False)
            print(f"  Helix RMSD CSV → {helix_csv}")
            # Summary
            print("\n  Helix RMSD summary (frag036 vs 1VII):")
            for hname, grp in helix_df.groupby("helix"):
                print(f"    {hname:<22s}  mean={grp['rmsd'].mean():.3f} nm  "
                      f"std={grp['rmsd'].std():.3f}  n={len(grp)}")
            plot_helix_rmsd(helix_df,
                out_path=os.path.join(out_dir, "helix_rmsd.png"))
        else:
            print("  No full-protein decoys found in frag36/.")
    else:
        print("  frag36 directory not found — skipping per-helix analysis.")

    print("\n  compare_to_native done.")
    return pd.DataFrame(summary_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein",       default="Villin_HP36",
                    help="Protein name (must match results/ and structures/ subdirs)")
    ap.add_argument("--results-dir",   default="results",
                    help="Base results directory")
    ap.add_argument("--structures-dir",default="structures",
                    help="Base structures directory")
    ap.add_argument("--pdb-clean-dir", default="data/PDB_clean",
                    help="Directory containing *_clean.pdb files")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    run(
        protein_name   = args.protein,
        results_dir    = os.path.join(base, args.results_dir),
        structures_base= os.path.join(base, args.structures_dir),
        pdb_clean_dir  = os.path.join(base, args.pdb_clean_dir),
    )


if __name__ == "__main__":
    main()
