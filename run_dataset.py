"""
Batch pipeline runner for all proteins from data/start2fold_data.json.

For each protein:
  Step 1 — Copy PDB & extract FASTA
  Step 2 — Generate N-terminal fragment FASTAs
  Step 3 — Run ESMFold structure prediction (1 prediction per fragment)
  Step 4 — Analyze structures (RMSD, Q, Hydro Rg) + cache raw metrics
  Step 5 — Visualize & save CSVs
  Step 6 — Advanced analysis (contact maps, importance)

Results land in  results/{protein_name}/
Logs land in     logs/dataset_run.log

Usage:
  python run_dataset.py                        # all proteins, all modules
  python run_dataset.py --proteins TrpCage Ubiquitin
  python run_dataset.py --skip-advanced
  python run_dataset.py --modules contacts importance
  python run_dataset.py --dry-run              # print commands without executing
"""
import argparse
import importlib
import logging
import os
import sys
import time
import types
import warnings
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR     = os.path.join(DATASET_DIR, "pdb_cleaned")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
RESULTS_BASE = os.path.join(BASE_DIR, "results")

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "dataset_run.log"), mode="a"),
    ],
)
log = logging.getLogger("dataset")


# ---------------------------------------------------------------------------
# Config patching  (import config once, override per protein)
# ---------------------------------------------------------------------------

import config as _cfg

# PDB filenames that deviate from the standard {pdb_id}_{chain}.pdb convention
_PDB_FILENAME_OVERRIDES = {
    "AF-P14621": "AF-P14621-F1-model_v6.pdb",
    # 1mse is a DNA-protein complex; protein is chain C, not A
    "1mse": "1mse_C.pdb",
}


def _pdb_path(pdb_id: str, chain) -> str:
    """Resolve the cleaned PDB file path for a given protein."""
    override = _PDB_FILENAME_OVERRIDES.get(pdb_id)
    if override:
        return os.path.join(PDB_DIR, override)
    # chain may be None, float NaN, or literal "—" for NMR proteins — default to "A"
    try:
        import math
        chain_safe = "A" if (not chain or str(chain).strip() in ("—", "-", "–", "nan", "")
                             or (isinstance(chain, float) and math.isnan(chain))) else str(chain).strip()
    except Exception:
        chain_safe = chain or "A"
    return os.path.join(PDB_DIR, f"{pdb_id}_{chain_safe}.pdb")


def _patch_config(protein_name, pdb_id, chain, n_residues, results_dir,
                  structures_dir, fragments_dir, raw_dir, n_decoys,
                  alignment=None, pdb_n_res=None):
    """Monkey-patch config module attributes for the current protein."""
    from config import adaptive_score_threshold
    _cfg.PDB_ID            = pdb_id
    _cfg.REFERENCE_PDB     = _pdb_path(pdb_id, chain)
    _cfg.REFERENCE_FASTA   = os.path.join(results_dir, f"{pdb_id}.fasta")
    _cfg.RESULTS_DIR       = results_dir
    _cfg.STRUCTURES_DIR    = structures_dir
    _cfg.FRAGMENTS_DIR     = fragments_dir
    _cfg.RAW_METRICS_DIR   = raw_dir
    _cfg.MIN_FRAGMENT_LENGTH = 3
    _cfg.MAX_FRAGMENT_LENGTH = n_residues
    _cfg.NSTRUCT           = n_decoys
    _cfg.SCORE_THRESHOLD   = adaptive_score_threshold(n_residues)
    _cfg.ALIGNMENT_MAP     = alignment
    _cfg.PDB_N_RES         = pdb_n_res


def _reload_modules():
    """Force reload of all pipeline modules so they pick up patched config."""
    mods = [
        "core.analyze", "modules.contact_profile", "core.score",
        "viz.plots", "core.fragments", "run_folding",
        "modules.contacts", "modules.importance",
    ]
    for name in mods:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


# ---------------------------------------------------------------------------
# Step helpers (operate on already-patched config)
# ---------------------------------------------------------------------------

def step1_copy_pdb_and_fasta(pdb_id, chain, results_dir, uniprot_accession=None,
                            domain_start=None, domain_end=None):
    """
    Extract sequence and write FASTA.

    If `uniprot_accession` is provided with explicit `domain_start/end`, uses the exact
    domain range. Otherwise, fetches the UniProt sequence and uses the region covering
    the PDB structure (including any missing N-terminal residues).

    Falls back to the PDB-extracted sequence on any error.

    Returns (sequence, n_res, alignment_or_None, pdb_n_res).
    """
    src = _pdb_path(pdb_id, chain)

    # Auto-download + clean PDB if missing
    if not os.path.exists(src):
        log.info(f"  [1] PDB not found, attempting download…")
        try:
            from core.download import download_pdb, clean_pdb
            pdb_raw_dir = os.path.join(PDB_DIR, "..", "pdb_raw")
            os.makedirs(pdb_raw_dir, exist_ok=True)
            download_pdb(pdb_id, pdb_raw_dir)
            raw_pdb = os.path.join(pdb_raw_dir, f"{pdb_id}.pdb")
            os.makedirs(PDB_DIR, exist_ok=True)
            clean_pdb(raw_pdb, src, chain_id=chain)
            log.info(f"  [1] Downloaded and cleaned PDB → {src}")
        except Exception as e:
            log.warning(f"  [1] Auto-download failed ({e})")
            raise FileNotFoundError(f"PDB not found and download failed: {src}")

    from Bio import PDB
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, src)
    model = structure[0]
    pdb_chain = next(iter(model.get_chains()))
    residues = [r for r in pdb_chain.get_residues() if PDB.is_aa(r, standard=True)]
    pdb_seq = "".join(seq1(r.get_resname()) for r in residues)
    pdb_n_res = len(pdb_seq)

    sequence = pdb_seq
    alignment = None

    if uniprot_accession:
        try:
            from core.download import fetch_uniprot_sequence
            from core.alignment import compute_alignment
            uniprot_full = fetch_uniprot_sequence(uniprot_accession)

            # Use explicit domain range if provided
            if domain_start and domain_end:
                domain_seq = uniprot_full[domain_start - 1 : domain_end]
                sequence = domain_seq
                log.info(
                    f"  [1] UniProt {uniprot_accession}: "
                    f"explicit domain [{domain_start}–{domain_end}] = {len(sequence)} aa"
                )
            else:
                # Fallback to heuristic alignment
                alignment = compute_alignment(uniprot_full, pdb_seq)
                pdb_offset = alignment["pdb_offset"]
                domain_seq = uniprot_full[:pdb_offset + pdb_n_res]
                if len(domain_seq) >= pdb_n_res:
                    sequence = domain_seq
                    log.info(
                        f"  [1] UniProt {uniprot_accession}: "
                        f"pdb_offset={pdb_offset}, "
                        f"domain={len(sequence)} aa "
                        f"(+{len(sequence) - pdb_n_res} residus vs PDB), "
                        f"method={alignment['method']}"
                    )
                else:
                    log.warning(f"  [1] UniProt domain shorter than PDB — using PDB seq")
                    alignment = None
        except Exception as e:
            log.warning(f"  [1] UniProt fetch failed ({e}) — using PDB sequence")
            alignment = None

    fasta_path = _cfg.REFERENCE_FASTA
    os.makedirs(results_dir, exist_ok=True)
    with open(fasta_path, "w") as fh:
        fh.write(f">{pdb_id}\n{sequence}\n")

    log.info(f"  [1] Sequence {len(sequence)} aa → {fasta_path}")
    return sequence, len(sequence), alignment, pdb_n_res


def step2_generate_fragments(fragments_dir, direction="N"):
    """Write fragment FASTA files (N-terminal, C-terminal, or both)."""
    from core.fragments import main as gen_main
    os.makedirs(fragments_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_main(direction=direction)
    n = len([f for f in os.listdir(fragments_dir) if f.endswith(".fasta")])
    log.info(f"  [2] {n} fragment FASTAs written to {fragments_dir} (direction={direction})")


def step3_run_folding(structures_dir, dry_run=False, batched=False, nstruct=None):
    """Run OpenFold 3 structure prediction for all fragment lengths."""
    from run_folding import main as folding_main
    folding_main(dry_run=dry_run, batched=batched, nstruct=nstruct)
    log.info(f"  [3] OpenFold 3 folding complete in {structures_dir}")



def step4_analyze(results_dir, raw_dir, bidirectional=False):
    """
    Analyze all fragment ensembles, return (fragment_df, residue_df).

    If bidirectional=True, also analyzes C-terminal fragment ensembles and
    applies bidirectional scoring.  For combined scoring (SCORING_METHOD='combined'),
    tries to load an existing structural_importance.csv from a prior step-6 run.
    """
    import warnings as _w
    from core.analyze import (_load_traj, compute_native_contacts, analyze_fragment,
                               compute_relative_contact_order, compute_reference_ss)
    from core.score import compute_fragment_scores, score_residues

    os.makedirs(raw_dir, exist_ok=True)

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        ref_traj = _load_traj(_cfg.REFERENCE_PDB)

    native_pairs, total_native = compute_native_contacts(ref_traj)
    log.info(f"  [4] Native contacts: {total_native}")

    # C4/MANS: Relative Contact Order (Plaxco 1998)
    rco = compute_relative_contact_order(ref_traj)
    log.info(f"  [4] RCO={rco:.3f}")

    # C2/MANS: Reference SS for SS-aware pLDDT thresholds
    ref_ss = compute_reference_ss(ref_traj)

    # Read alignment (set by step1 if UniProt was used)
    alignment = getattr(_cfg, "ALIGNMENT_MAP", None)

    # Analyze N-terminal fragments
    metrics_n = []
    for length in range(_cfg.MIN_FRAGMENT_LENGTH, _cfg.MAX_FRAGMENT_LENGTH + 1):
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            result = analyze_fragment(length, ref_traj, native_pairs, total_native,
                                      save_raw=True, raw_dir=raw_dir, direction="N",
                                      alignment=alignment, ref_ss=ref_ss)
        if result is not None:
            metrics_n.append(result)

    if not metrics_n:
        log.warning("  [4] No N-terminal structures found.")
        return pd.DataFrame(), pd.DataFrame()

    fragment_df = compute_fragment_scores(metrics_n)

    # Analyze C-terminal fragments (bidirectional mode)
    c_fragment_df = pd.DataFrame()
    if bidirectional:
        metrics_c = []
        for length in range(_cfg.MIN_FRAGMENT_LENGTH, _cfg.MAX_FRAGMENT_LENGTH + 1):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # C-terminal fragments: alignment not applicable (use PDB-only logic)
                result = analyze_fragment(length, ref_traj, native_pairs, total_native,
                                          save_raw=False, raw_dir=raw_dir, direction="C",
                                          ref_ss=ref_ss)
            if result is not None:
                metrics_c.append(result)
        if metrics_c:
            c_fragment_df = compute_fragment_scores(metrics_c)
            log.info(f"  [4] {len(metrics_c)} C-terminal fragments analyzed.")

    # Load existing importance CSV for combined/unified scoring (if available from prior step-6)
    importance_df = None
    imp_path = os.path.join(results_dir, "structural_importance.csv")
    if os.path.exists(imp_path):
        try:
            importance_df = pd.read_csv(imp_path)
            log.info(f"  [4] Loaded structural importance from {imp_path}")
        except Exception as e:
            log.warning(f"  [4] Could not load importance CSV: {e}")

    scoring_method = _cfg.SCORING_METHOD
    if scoring_method == "bidirectional" or bidirectional:
        effective_method = "bidirectional"
    elif scoring_method == "unified":
        effective_method = "unified"
    elif scoring_method == "combined" and importance_df is not None:
        effective_method = "combined"
    else:
        effective_method = "marginal" if scoring_method == "combined" else scoring_method

    # Compute importance on-demand for unified/combined scoring if not already available
    if (effective_method in ("unified", "combined") and
        (importance_df is None or importance_df.empty)):
        try:
            from modules.importance import compute_importance_from_native
            importance_df = compute_importance_from_native(_cfg.REFERENCE_PDB)
            log.info(f"  [4] Computed structural importance from {_cfg.REFERENCE_PDB}")
        except Exception as e:
            log.warning(f"  [4] Could not compute importance: {e}")
            importance_df = None

    residue_df = score_residues(
        fragment_df,
        max_length=_cfg.MAX_FRAGMENT_LENGTH,
        method=effective_method,
        importance_df=importance_df,
        c_fragment_df=c_fragment_df if not c_fragment_df.empty else None,
        ref_pdb_path=_cfg.REFERENCE_PDB,
        relative_contact_order=rco,
    )

    nucleus = residue_df[residue_df["is_nucleus"]]["residue"].tolist()
    log.info(f"  [4] {len(metrics_n)} fragments analyzed — method={effective_method} "
             f"— high-score candidates: {nucleus}")
    return fragment_df, residue_df


def step5_visualize(fragment_df, residue_df, results_dir):
    """Save CSVs and PNG plots."""
    from viz.plots import (plot_nucleus_scores, plot_rmsd_convergence,
                           plot_q_convergence, save_results,
                           plot_hydro_rg_convergence,
                           plot_nucleus_score_distribution,
                           plot_metric_contribution_heatmap,
                           plot_composite_summary)
    save_results(fragment_df, residue_df, out_dir=results_dir)
    plot_nucleus_scores(residue_df,
                        out_path=os.path.join(results_dir, "folding_nucleus.png"))
    plot_rmsd_convergence(fragment_df,
                          out_path=os.path.join(results_dir, "rmsd_convergence.png"))
    plot_q_convergence(fragment_df,
                       out_path=os.path.join(results_dir, "q_convergence.png"))
    plot_hydro_rg_convergence(fragment_df,
                               out_path=os.path.join(results_dir, "hydro_rg_convergence.png"))
    plot_nucleus_score_distribution(residue_df,
                                    out_path=os.path.join(results_dir, "nucleus_score_distribution.png"))
    plot_metric_contribution_heatmap(fragment_df,
                                     out_path=os.path.join(results_dir, "metric_contribution_heatmap.png"))
    plot_composite_summary(residue_df, fragment_df,
                           out_path=os.path.join(results_dir, "composite_summary.png"))
    log.info(f"  [5] Plots & CSVs saved to {results_dir}")


def step6_advanced(ref_traj, residue_df, raw_dir, results_dir, modules):
    """Advanced analysis: contact maps, importance."""
    lengths = list(range(_cfg.MIN_FRAGMENT_LENGTH, _cfg.MAX_FRAGMENT_LENGTH + 1))

    cmaps = {}
    if "contacts" in modules or "importance" in modules:
        from modules.contacts import run_contact_map_analysis
        cmaps, _ = run_contact_map_analysis(ref_traj, lengths=lengths,
                                             out_dir=results_dir)

    if "importance" in modules:
        from modules.importance import run_importance_analysis
        run_importance_analysis(ref_traj, cmaps,
                                residue_df=residue_df if not residue_df.empty else None,
                                out_dir=results_dir, lengths=lengths,
                                pdb_path=_cfg.REFERENCE_PDB)

    # Contact formation heatmap (ab initio scoring diagnostics)
    try:
        import warnings as _w
        from core.analyze import compute_native_contacts
        from modules.contact_profile import build_contact_formation_profile
        from viz.plots import plot_contact_formation_heatmap

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            native_pairs_adv, _ = compute_native_contacts(ref_traj)
        profile, valid_lengths = build_contact_formation_profile(
            ref_traj, native_pairs_adv, lengths=lengths
        )
        if profile.size > 0:
            plot_contact_formation_heatmap(
                profile, valid_lengths,
                residue_df=residue_df if not residue_df.empty else None,
                out_path=os.path.join(results_dir, "contact_formation_heatmap.png"),
            )
    except Exception as e:
        log.warning(f"  [6] Contact formation heatmap skipped: {e}")

    log.info(f"  [6] Advanced analysis done.")


# ---------------------------------------------------------------------------
# Per-protein runner
# ---------------------------------------------------------------------------

def run_protein(row, n_decoys, modules, skip_advanced, only_advanced=False,
                run_sensitivity=False, dry_run=False,
                bidirectional=False, build_trajectory=False, dashboard=False,
                run_label=None, batched=False,
                run_annotation_comparison=False):
    name             = row["Name"]
    pdb_id           = row["PDB"]
    chain            = row.get("Chain", "A")
    n_res_ann        = int(row["Size_aa"])  # annotation size (may differ from PDB)
    uniprot_accession = row.get("UniProt") or None
    domain_start     = row.get("DomainStart") or None
    domain_end       = row.get("DomainEnd") or None

    if run_label:
        results_dir    = os.path.join(BASE_DIR, f"results_{run_label}", name)
        structures_dir = os.path.join(BASE_DIR, f"structures_{run_label}", name)
    else:
        results_dir    = os.path.join(RESULTS_BASE, name)
        structures_dir = os.path.join(BASE_DIR, "structures", name)
    fragments_dir  = os.path.join(BASE_DIR, "fragments", name)
    raw_dir        = os.path.join(results_dir, "raw")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)
    os.makedirs(fragments_dir, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  PROTEIN: {name}  ({pdb_id})  ~{n_res_ann} aa")
    log.info(f"{'='*60}")

    # Patch config BEFORE any step so all modules see the right paths
    # (alignment and pdb_n_res are updated after step1 when UniProt is used)
    _patch_config(name, pdb_id, chain, n_res_ann, results_dir,
                  structures_dir, fragments_dir, raw_dir, n_decoys)
    _reload_modules()
    s2f_metrics = {}

    if only_advanced:
        # Skip steps 1-5; derive n_res from the actual PDB file
        from Bio import PDB
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        src = _pdb_path(pdb_id, chain)
        structure = parser.get_structure(pdb_id, src)
        chain = next(iter(structure[0].get_chains()))
        n_res = sum(1 for r in chain.get_residues() if PDB.is_aa(r, standard=True))
        _cfg.MAX_FRAGMENT_LENGTH = n_res
        # Reload with correct MAX_FRAGMENT_LENGTH
        _reload_modules()
        # Load existing residue_df if available
        res_csv = os.path.join(results_dir, "residue_scores.csv")
        residue_df = pd.read_csv(res_csv) if os.path.exists(res_csv) else pd.DataFrame()
        fragment_df = pd.DataFrame()
    else:
        # Step 1 — extract sequence & FASTA (UniProt if available)
        try:
            sequence, n_res, alignment, pdb_n_res = step1_copy_pdb_and_fasta(
                pdb_id, chain, results_dir, uniprot_accession=uniprot_accession,
                domain_start=domain_start, domain_end=domain_end
            )
        except Exception as e:
            log.error(f"  [1] FAILED: {e}")
            return None

        # Update MAX_FRAGMENT_LENGTH to actual sequence length (UniProt domain or PDB)
        _cfg.MAX_FRAGMENT_LENGTH = n_res
        _cfg.ALIGNMENT_MAP = alignment
        _cfg.PDB_N_RES = pdb_n_res

        # Step 2 — fragments
        try:
            frag_dir = "both" if bidirectional else "N"
            step2_generate_fragments(fragments_dir, direction=frag_dir)
        except Exception as e:
            log.error(f"  [2] FAILED: {e}")
            return None

        # Step 3 — OpenFold 3 structure prediction
        try:
            step3_run_folding(structures_dir,
                              dry_run=dry_run, batched=batched, nstruct=n_decoys)
        except Exception as e:
            log.error(f"  [3] FAILED: {e}")
            return None

        # Step 4 — analyze
        try:
            fragment_df, residue_df = step4_analyze(results_dir, raw_dir,
                                                    bidirectional=bidirectional)
        except Exception as e:
            log.error(f"  [4] FAILED: {e}")
            traceback.print_exc()
            return None

        if fragment_df.empty:
            log.warning(f"  Skipping steps 5-6 (no data).")
            return None

        # Step 5 — visualize
        try:
            step5_visualize(fragment_df, residue_df, results_dir)
        except Exception as e:
            log.error(f"  [5] FAILED: {e}")

        # Step 5b — validation
        try:
            from validation.checks import run_validation
            run_validation(fragment_df, residue_df, results_dir)
        except Exception as e:
            log.error(f"  [5b] Validation FAILED: {e}")

        # Step 5f — comparison with Start2Fold HDX protection (all 6 levels)
        try:
            from validation.start2fold import load_start2fold, evaluate_protein_all_levels
            s2f_json = os.path.join(DATASET_DIR, "start2fold_data.json")
            s2f_data = load_start2fold(s2f_json)
            s2f_metrics = evaluate_protein_all_levels(
                pdb_id.lower(), Path(results_dir), s2f_data
            )
            early_f1 = s2f_metrics.get("s2f_early_f1", float("nan"))
            strong_f1 = s2f_metrics.get("s2f_strong_f1", float("nan"))
            log.info(
                f"  [5f] Start2Fold — EARLY F1={early_f1:.3f}  STRONG F1={strong_f1:.3f}"
                if not (early_f1 != early_f1) else  # nan check
                f"  [5f] Start2Fold — no EARLY data for {pdb_id}"
            )
        except Exception as e:
            log.debug(f"  [5f] Start2Fold comparison skipped: {e}")

        # Step 5c — independently predicted prefix-length structure series
        if build_trajectory:
            try:
                from modules.trajectory import (build_progressive_trajectory,
                                                generate_pymol_script)
                out_pdb = os.path.join(results_dir, "progressive_folding.pdb")
                out_pml = os.path.join(results_dir, "progressive_folding.pml")
                traj_pdb = build_progressive_trajectory(
                    structures_dir, _cfg.REFERENCE_PDB, out_pdb)
                if traj_pdb:
                    generate_pymol_script(traj_pdb, _cfg.REFERENCE_PDB, out_pml)
                    log.info(f"  [5c] Prefix-length structure series: {out_pdb}")
            except Exception as e:
                log.error(f"  [5c] Trajectory FAILED: {e}")

        # Step 5d — construction dashboard
        if dashboard and not fragment_df.empty:
            try:
                from viz.plots import plot_construction_dashboard
                plot_construction_dashboard(
                    structures_dir, _cfg.REFERENCE_PDB, fragment_df,
                    out_path=os.path.join(results_dir, "construction_dashboard.png"))
                log.info(f"  [5d] Construction dashboard saved.")
            except Exception as e:
                log.error(f"  [5d] Dashboard FAILED: {e}")

    # Step 5c — sensitivity analysis
    if run_sensitivity and not fragment_df.empty:
        try:
            from modules.sensitivity import run_sensitivity_analysis
            run_sensitivity_analysis(fragment_df, residue_df, out_dir=results_dir)
        except Exception as e:
            log.error(f"  [5c] Sensitivity FAILED: {e}")

    # Step 6 — advanced analysis
    if not skip_advanced and modules:
        try:
            import warnings as _w
            from core.analyze import _load_traj
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                ref_traj = _load_traj(_cfg.REFERENCE_PDB)
            step6_advanced(ref_traj, residue_df, raw_dir, results_dir, modules)
        except Exception as e:
            log.error(f"  [6] FAILED: {e}")

    # Summary row — including global metrics (§6.3)
    nucleus = residue_df[residue_df["is_nucleus"]]["residue"].tolist() if not residue_df.empty else []

    # Optional legacy comparison with heterogeneous residue annotations.
    nucleus_precision = nucleus_recall = nucleus_f1 = float("nan")
    if run_annotation_comparison:
        try:
            from validation.nucleus import compute_nucleus_metrics, get_resseq_mapping
            ann_row_val = row
            pdb_path_val = _pdb_path(pdb_id, chain)
            resseq_mapping = get_resseq_mapping(pdb_path_val) if os.path.exists(pdb_path_val) else None
            metrics_dict = compute_nucleus_metrics(
                nucleus, ann_row_val, resseq_mapping=resseq_mapping
            )
            if metrics_dict:
                nucleus_precision = metrics_dict.get("precision", float("nan"))
                nucleus_recall    = metrics_dict.get("recall",    float("nan"))
                nucleus_f1        = metrics_dict.get("f1",        float("nan"))
        except Exception:
            pass

    # Compute rmsd_min_full for the full-length fragment (compare_to_native)
    rmsd_min_full = float("nan")
    native_csv = os.path.join(results_dir, "compare_to_native.csv")
    if os.path.exists(native_csv):
        try:
            ctn_df = pd.read_csv(native_csv)
            full_rows = ctn_df[ctn_df["fragment"] == _cfg.MAX_FRAGMENT_LENGTH]
            if not full_rows.empty and "rmsd_min" in full_rows.columns:
                rmsd_min_full = float(full_rows["rmsd_min"].iloc[0])
        except Exception:
            pass

    summary = {
        "name":              name,
        "pdb_id":            pdb_id,
        "n_residues":        n_res,
        "n_fragments":       len(fragment_df),
        "nucleus_residues":  nucleus,
        "n_nucleus":         len(nucleus),
        "candidate_residues": nucleus,
        "n_candidates":       len(nucleus),
        "nucleus_precision": nucleus_precision,
        "nucleus_recall":    nucleus_recall,
        "nucleus_f1":        nucleus_f1,
        "rmsd_min_full":     rmsd_min_full,
        "scoring_method":    _cfg.SCORING_METHOD,
    }
    summary.update(s2f_metrics)   # adds s2f_early_f1, s2f_strong_f1, … for all 6 levels
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FoldNucleus — compute fragment-emergence scores for dataset proteins."
    )
    parser.add_argument("--proteins", nargs="+",
                        help="Restrict to these protein names (e.g. TrpCage Ubiquitin)")
    parser.add_argument("--n-decoys", type=int, default=1,
                        help="ESMFold predictions per fragment (default: 1). "
                             "ESMFold is deterministic — use 1 unless you enable "
                             "input perturbation for ensembling.")
    parser.add_argument("--skip-advanced", action="store_true",
                        help="Skip Step 6 advanced analysis")
    parser.add_argument("--only-advanced", action="store_true",
                        help="Skip steps 1-5, run only step 6 (requires existing results)")
    parser.add_argument("--modules", nargs="+",
                        default=["contacts", "importance"],
                        choices=["contacts", "importance"],
                        help="Advanced modules to run (default: all)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis on SCORE_THRESHOLD per protein")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print folding commands without executing")
    parser.add_argument("--batched", action="store_true",
                        help="Submit all fragments in one OpenFold 3 call")
    parser.add_argument("--no-bidirectional", dest="bidirectional",
                        action="store_false", default=False,
                        help="Disable bidirectional scoring (disabled by default)")
    parser.add_argument("--bidirectional", dest="bidirectional",
                        action="store_true",
                        help="Enable bidirectional scoring")
    parser.add_argument("--compare-native", action="store_true",
                        help="Run compare_to_native.py after the pipeline (per-decoy RMSD vs PDB)")
    parser.add_argument("--phi-compare", action="store_true",
                        help=("Run exploratory Φ-value comparison; bundled fallback "
                              "tables are smoke-test data only"))
    parser.add_argument("--validate-nucleus", action="store_true",
                        help=("Run legacy exploratory candidate/reference overlap "
                              "analysis; not folding-nucleus validation"))
    parser.add_argument("--low-mem", action="store_true",
                        help="Enable OpenFold 3 low-memory GPU preset")
    parser.add_argument("--trajectory", action="store_true",
                        help=("Build an animated prefix-length structure series; "
                              "states are not folding time"))
    parser.add_argument("--dashboard", action="store_true",
                        help="Generate construction dashboard (2D PCA + RMSD/Q panels)")
    parser.add_argument("--scoring-method",
                        choices=["combined", "ab_initio", "marginal", "zscore",
                                 "sliding_window", "bidirectional", "native", "unified"],
                        default=None,
                        help="Override SCORING_METHOD from config for this run")
    parser.add_argument("--parallel-proteins", type=int, default=1,
                        metavar="N",
                        help="Run N proteins in parallel (default: 1 = sequential).")
    parser.add_argument("--database", choices=["start2fold", "v3", "merged"], default="start2fold",
                        help="Dataset to use: 'start2fold' (start2fold_data.json, default), "
                             "'v3' (two_state_folding_v3.xlsx), 'merged' (v3 + Start2Fold deduped)")
    parser.add_argument("--run-label", default=None,
                        help="Label for this run (results_{label}/, structures_{label}/)")
    args = parser.parse_args()

    # Redirect all outputs to results_{label}/ when --run-label is set
    global RESULTS_BASE
    if args.run_label:
        RESULTS_BASE = os.path.join(BASE_DIR, f"results_{args.run_label}")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    if args.low_mem:
        import config as _c
        _c.OPENFOLD_LOW_MEM = True

    log.info("Loading dataset annotations …")
    if args.database == "start2fold":
        from data.start2fold_adapter import load_protein_list as s2f_load
        s2f_path = os.path.join(DATASET_DIR, "start2fold_data.json")
        df_ann = s2f_load(json_path=s2f_path, max_size=0)
        log.info(f"  Loaded {len(df_ann)} proteins from start2fold_data.json")
    elif args.database == "v3":
        from data.kprodb_adapter import load_v3_protein_list
        xlsx_path = os.path.join(DATASET_DIR, "two_state_folding_v3.xlsx")
        df_ann = load_v3_protein_list(xlsx_path)
        log.info(f"  Loaded {len(df_ann)} proteins from two_state_folding_v3.xlsx")
    elif args.database == "merged":
        from data.kprodb_adapter import load_v3_protein_list
        from data.start2fold_adapter import load_merged_protein_list
        xlsx_path = os.path.join(DATASET_DIR, "two_state_folding_v3.xlsx")
        s2f_path = os.path.join(DATASET_DIR, "start2fold_data.json")
        v3_df = load_v3_protein_list(xlsx_path)
        df_ann = load_merged_protein_list(v3_df, json_path=s2f_path, max_size=0)
        log.info(f"  Loaded {len(df_ann)} proteins (merged v3 + Start2Fold)")

    if args.proteins:
        from data.kprodb_adapter import _clean_name
        normalized = [_clean_name(p) for p in args.proteins]
        df_ann = df_ann[df_ann["Name"].isin(normalized)].reset_index(drop=True)
        if df_ann.empty:
            log.error(f"No proteins matched: {args.proteins}")
            sys.exit(1)

    # Apply scoring method override from CLI
    if args.scoring_method is not None:
        _cfg.SCORING_METHOD = args.scoring_method
        log.info(f"Scoring method overridden to: {args.scoring_method}")

    log.info(f"Dataset: {len(df_ann)} proteins | "
             f"nstruct={args.n_decoys} | mode=OpenFold3 | modules={args.modules} | "
             f"only_advanced={args.only_advanced} | scoring={_cfg.SCORING_METHOD}")

    summary_rows = []
    t0 = time.time()

    # Build kwargs dict shared by all protein runs
    _run_kwargs = dict(
        n_decoys=args.n_decoys,
        modules=args.modules,
        skip_advanced=args.skip_advanced,
        only_advanced=args.only_advanced,
        run_sensitivity=args.sensitivity,
        dry_run=args.dry_run,
        bidirectional=args.bidirectional,
        batched=args.batched,
        build_trajectory=args.trajectory,
        dashboard=args.dashboard,
        run_label=args.run_label,
        run_annotation_comparison=args.validate_nucleus,
    )

    rows_list = [row for _, row in df_ann.iterrows()]

    n_parallel = max(1, args.parallel_proteins)
    if n_parallel == 1:
        # Sequential (original behaviour)
        completed_rows = []
        for row in rows_list:
            result = run_protein(row, **_run_kwargs)
            if result:
                summary_rows.append(result)
            completed_rows.append(row)
    else:
        # Parallel across proteins — each protein is a separate process so
        # config patching (monkey-patch of _cfg) stays isolated per child.
        log.info(f"Parallel mode: {n_parallel} proteins simultaneously.")
        completed_rows = list(rows_list)  # for post-processing below

        # Use spawn context so children don't inherit stale module state
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_parallel, mp_context=ctx) as pool:
            future_to_row = {
                pool.submit(run_protein, row, **_run_kwargs): row
                for row in rows_list
            }
            for fut in as_completed(future_to_row):
                row = future_to_row[fut]
                name = row["Name"]
                try:
                    result = fut.result()
                    if result:
                        summary_rows.append(result)
                    log.info(f"  [{name}] completed.")
                except Exception as exc:
                    log.error(f"  [{name}] FAILED in parallel run: {exc}")

    # Optional post-processing (native comparison / Φ comparison / annotation overlap)
    for row in completed_rows:
        name = row["Name"]

        # Step 7a — compare_to_native
        if args.compare_native:
            try:
                from validation.native import run as ctn_run
                ctn_run(protein_name=name,
                        results_dir=RESULTS_BASE,
                        structures_base=os.path.join(BASE_DIR, "structures"),
                        pdb_clean_dir=PDB_DIR)
            except Exception as e:
                log.error(f"  [7a] compare_to_native FAILED for {name}: {e}")

        # Step 7b — phi_value_comparison
        if args.phi_compare:
            try:
                from validation.phi_values import run as phi_run
                phi_run(protein_name=name, results_dir=RESULTS_BASE,
                        pdb_id=row["PDB"])
            except Exception as e:
                log.error(f"  [7b] phi_value_comparison FAILED for {name}: {e}")

        # Step 7c — validate_nucleus
        if args.validate_nucleus:
            try:
                from validation.nucleus import run as vn_run
                ann_row_7c = df_ann[df_ann["Name"] == name].iloc[0] if not df_ann[df_ann["Name"] == name].empty else None
                pdb_id_7c = row["PDB"]
                chain_7c    = row.get("Chain", "A")
                pdb_path_7c = _pdb_path(pdb_id_7c, chain_7c)
                vn_run(protein_name=name, results_dir=RESULTS_BASE,
                       ann_row=ann_row_7c,
                       pdb_path=pdb_path_7c if os.path.exists(pdb_path_7c) else None)
            except Exception as e:
                log.error(f"  [7c] validate_nucleus FAILED for {name}: {e}")

    # Global summary
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"DATASET RUN COMPLETE  ({elapsed:.1f}s)")
    log.info(f"{'='*60}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_BASE, "dataset_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        log.info(f"Summary saved to {summary_path}")
        log.info("\n" + summary_df[["name", "pdb_id", "n_residues",
                                    "n_candidates"]].to_string(index=False))


if __name__ == "__main__":
    main()
