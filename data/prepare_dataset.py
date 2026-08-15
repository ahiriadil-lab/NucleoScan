"""
prepare_dataset.py — Standalone dataset preparation for two_state_folding_v3.xlsx

For each protein:
  1. Download raw PDB from RCSB
  2. Clean PDB (keep specified chain, standard AA only)
  3. If UniProt available: download + extract domain sequence
  4. If no UniProt: extract sequence from cleaned PDB

Usage:
  python data/prepare_dataset.py                        # all proteins
  python data/prepare_dataset.py --proteins 1CSP 1PGB   # specific proteins
  python data/prepare_dataset.py --force                # re-download existing
  python data/prepare_dataset.py --dry-run              # preview only
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bio import PDB
from Bio.SeqUtils import seq1
from core.download import download_pdb, fetch_uniprot_sequence, clean_pdb
from data.kprodb_adapter import load_v3_protein_list, parse_domain_range

# Setup logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "prepare_dataset.log"), mode="a"),
    ],
)
log = logging.getLogger("prepare")


def extract_pdb_sequence(pdb_path: str, chain_id: str = None) -> str:
    """Extract sequence from cleaned PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
    except Exception as e:
        log.warning(f"Could not parse {pdb_path}: {e}")
        return None

    model = structure[0]

    # Find chain
    target_chain = None
    if chain_id:
        for chain in model.get_chains():
            if chain.get_id() == chain_id:
                target_chain = chain
                break
    if target_chain is None:
        target_chain = next(iter(model.get_chains()), None)

    if target_chain is None:
        return None

    # Extract sequence
    residues = [r for r in target_chain.get_residues() if PDB.is_aa(r, standard=True)]
    sequence = "".join(seq1(r.get_resname()) for r in residues)
    return sequence


def prepare_protein(row, force=False, dry_run=False):
    """Prepare a single protein: download PDB, clean, extract domain."""
    name = row["Name"]
    pdb_id = row["PDB"]
    chain = row["Chain"]
    uniprot = row["UniProt"]
    domain_start = row["DomainStart"]
    domain_end = row["DomainEnd"]
    size_aa = row["Size_aa"]

    # Directories
    pdb_raw_dir = os.path.join(DATA_DIR, "pdb_raw")
    pdb_clean_dir = os.path.join(DATA_DIR, "pdb_cleaned")
    domain_fasta_dir = os.path.join(DATA_DIR, "domain_fasta")
    uniprot_cache_dir = os.path.join(DATA_DIR, "uniprot")

    os.makedirs(pdb_raw_dir, exist_ok=True)
    os.makedirs(pdb_clean_dir, exist_ok=True)
    os.makedirs(domain_fasta_dir, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  {name:30s} PDB={pdb_id:5s} UniProt={uniprot or '—':10s} chain={chain or '—':3s}")
    log.info(f"{'='*60}")

    try:
        # Step 1: Download raw PDB
        raw_pdb_path = os.path.join(pdb_raw_dir, f"{pdb_id}.pdb")
        if not os.path.exists(raw_pdb_path) or force:
            if not dry_run:
                log.info(f"  [1] Downloading {pdb_id} from RCSB…")
                download_pdb(pdb_id, pdb_raw_dir)
            log.info(f"  [1] Raw PDB → {raw_pdb_path}")
        else:
            log.info(f"  [1] Raw PDB exists, skipping download")

        # Step 2: Clean PDB
        # When chain is None (Excel shows "—"), peek at raw PDB to get first chain letter
        actual_chain = chain
        if actual_chain is None and os.path.exists(raw_pdb_path):
            with open(raw_pdb_path) as _f:
                for _line in _f:
                    if _line.startswith("ATOM") and len(_line) > 22:
                        _c = _line[21].strip()
                        if _c:
                            actual_chain = _c
                            break
        actual_chain = actual_chain or "A"

        clean_pdb_path = os.path.join(pdb_clean_dir, f"{pdb_id}_{actual_chain}.pdb")
        if not os.path.exists(clean_pdb_path) or force:
            if not dry_run and os.path.exists(raw_pdb_path):
                log.info(f"  [2] Cleaning PDB (chain={actual_chain})…")
                clean_pdb(raw_pdb_path, clean_pdb_path, chain_id=chain)
            log.info(f"  [2] Cleaned PDB → {clean_pdb_path}")
        else:
            log.info(f"  [2] Cleaned PDB exists, skipping")

        # Step 3: Domain sequence
        if uniprot:
            if not dry_run:
                log.info(f"  [3] Fetching UniProt {uniprot}…")
                uniprot_full = fetch_uniprot_sequence(uniprot, uniprot_cache_dir)

                if domain_start and domain_end:
                    # 1-based to 0-based
                    domain_seq = uniprot_full[domain_start - 1 : domain_end]
                    log.info(f"      Domain [{domain_start}–{domain_end}] = {len(domain_seq)} aa")
                else:
                    domain_seq = uniprot_full
                    log.info(f"      Full sequence = {len(domain_seq)} aa")

                # Validate
                expected_len = domain_end - domain_start + 1 if domain_start and domain_end else len(uniprot_full)
                if len(domain_seq) != expected_len:
                    log.warning(f"      Length mismatch: extracted {len(domain_seq)}, expected {expected_len}")

                # Save
                domain_fasta_path = os.path.join(domain_fasta_dir, f"{uniprot}_domain.fasta")
                with open(domain_fasta_path, "w") as fh:
                    fh.write(f">{pdb_id}|{uniprot}\n{domain_seq}\n")
            else:
                domain_fasta_path = os.path.join(domain_fasta_dir, f"{uniprot}_domain.fasta")
            log.info(f"  [3] Domain FASTA → {domain_fasta_path}")
        else:
            # No UniProt: extract from cleaned PDB
            if not dry_run and os.path.exists(clean_pdb_path):
                log.info(f"  [3] No UniProt, extracting sequence from PDB…")
                pdb_seq = extract_pdb_sequence(clean_pdb_path, chain)
                if pdb_seq:
                    log.info(f"      Sequence = {len(pdb_seq)} aa")
                    domain_fasta_path = os.path.join(domain_fasta_dir, f"{pdb_id}_domain.fasta")
                    with open(domain_fasta_path, "w") as fh:
                        fh.write(f">{pdb_id}\n{pdb_seq}\n")
                    log.info(f"  [3] Domain FASTA → {domain_fasta_path}")
                else:
                    log.warning(f"  [3] Could not extract sequence from PDB")
            else:
                log.info(f"  [3] No UniProt available (designed protein)")

        return True

    except Exception as e:
        log.error(f"  FAILED: {e}", exc_info=False)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: download PDB, clean, extract domains"
    )
    parser.add_argument("--proteins", nargs="+",
                        help="Restrict to PDB IDs (e.g., 1CSP 1PGB)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download/re-clean existing files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only (no downloads)")
    args = parser.parse_args()

    log.info("Loading v3 dataset…")
    xlsx_path = os.path.join(DATA_DIR, "two_state_folding_v3.xlsx")
    df = load_v3_protein_list(xlsx_path)
    log.info(f"Loaded {len(df)} proteins from {xlsx_path}")

    if args.proteins:
        pdb_list = [p.lower() for p in args.proteins]
        df = df[df["PDB"].isin(pdb_list)].reset_index(drop=True)
        if df.empty:
            log.error(f"No proteins matched: {args.proteins}")
            sys.exit(1)
        log.info(f"Filtered to {len(df)} proteins")

    success = 0
    failed = 0
    for _, row in df.iterrows():
        if prepare_protein(row, force=args.force, dry_run=args.dry_run):
            success += 1
        else:
            failed += 1

    log.info(f"\n{'='*60}")
    log.info(f"Summary: {success} succeeded, {failed} failed")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
