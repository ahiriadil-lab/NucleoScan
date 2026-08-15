"""
start2fold_adapter.py — Adapter for Start2Fold database (start2fold_data.json).

The Start2Fold database (Pancsa & Tompa 2016) reports per-residue HDX
protection levels along two independent axes:

  Folding at residue resolution (Type = "folding"):
    EARLY        — protected at the earliest reported refolding stage
    INTERMEDIATE — protected at intermediate reported stages
    LATE         — protected at later reported stages

  Stability at residue resolution (Type = "stability"):
    STRONG       — strongly protected under equilibrium conditions
    MEDIUM       — moderately stable residues
    WEAK         — marginally stable residues

EARLY is used as the primary kinetic-protection benchmark and STRONG as the
strongest equilibrium-protection class. Neither class is folding-nucleus or
transition-state ground truth.

Produces a DataFrame compatible with run_dataset.py.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data.kprodb_adapter import parse_domain_range

_MAX_SIZE_DEFAULT = 180  # aa — default upper size limit for ESMFold reliability

_NAME_CLEAN_RE = re.compile(r"[^\w]")


def _clean_name(name: str) -> str:
    return _NAME_CLEAN_RE.sub("_", name.strip()).strip("_")


def load_protein_list(
    json_path: str = None,
    max_size: int = _MAX_SIZE_DEFAULT,
    require_early: bool = False,
) -> pd.DataFrame:
    """
    Parse start2fold_data.json and return one row per unique PDB.

    Parameters
    ----------
    json_path    Path to start2fold_data.json (defaults to data/ next to this file).
    max_size     Discard proteins with n_residues > max_size (0 = no limit).
    require_early  If True, only include proteins with at least one EARLY
                 folding experiment.

    Returns DataFrame with columns compatible with run_dataset.py:
        Name, PDB, Chain, Size_aa, UniProt, DomainStart, DomainEnd,
        DomainRange, Noyau résidus (legacy alias of the EARLY union),
        start2fold_early (set serialised as "p1;p2;…"),
        start2fold_strong (idem for stability STRONG)
    """
    if json_path is None:
        json_path = _HERE / "start2fold_data.json"

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    # Aggregate per PDB
    proteins: dict[str, dict] = {}
    for rec in records:
        pdb = str(rec.get("PDB code", "")).strip().lower()
        if not pdb:
            continue

        ptype = str(rec.get("Type", "")).strip().lower()
        level = str(rec.get("Protection Level", "")).strip().upper()
        sequence = str(rec.get("Sequence", "")).strip()
        fragment = str(rec.get("Fragment (UniProt residues)", "")).strip()
        uniprot = str(rec.get("UniProt ID", "")).strip()
        name_raw = str(rec.get("Name in UniProt", rec.get("Entry Title", ""))).strip()

        try:
            n_res = int(rec.get("Number of residues", 0))
        except (ValueError, TypeError):
            n_res = len(sequence) if sequence else 0

        if pdb not in proteins:
            proteins[pdb] = {
                "pdb": pdb,
                "name_raw": name_raw,
                "uniprot": uniprot,
                "fragment": fragment,
                "sequence": sequence,
                "n_res": n_res,
                # Folding-protection axis
                "early_sets": [],        # EARLY protection
                "intermediate_sets": [], # INTERMEDIATE
                "late_sets": [],         # LATE protection
                # Equilibrium stability/protection axis
                "strong_sets": [],       # STRONG protection
                "medium_sets": [],       # MEDIUM
                "weak_sets": [],         # WEAK
            }

        res_raw = str(rec.get("Residues", "")).strip()
        pos_set = _parse_residues(res_raw, sequence, pdb)

        if ptype == "folding":
            if level == "EARLY":
                proteins[pdb]["early_sets"].append(pos_set)
            elif level == "INTERMEDIATE":
                proteins[pdb]["intermediate_sets"].append(pos_set)
            elif level == "LATE":
                proteins[pdb]["late_sets"].append(pos_set)
        elif ptype == "stability":
            if level == "STRONG":
                proteins[pdb]["strong_sets"].append(pos_set)
            elif level == "MEDIUM":
                proteins[pdb]["medium_sets"].append(pos_set)
            elif level == "WEAK":
                proteins[pdb]["weak_sets"].append(pos_set)

    # Build output rows
    rows = []
    for pdb, d in proteins.items():
        n_res = d["n_res"]
        if max_size > 0 and n_res > max_size:
            continue

        def _union(sets):
            return set().union(*sets) if sets else set()

        early_union        = _union(d["early_sets"])
        intermediate_union = _union(d["intermediate_sets"])
        late_union         = _union(d["late_sets"])
        strong_union       = _union(d["strong_sets"])
        medium_union       = _union(d["medium_sets"])
        weak_union         = _union(d["weak_sets"])

        if require_early and not early_union:
            continue

        # Parse fragment range
        domain_start, domain_end = None, None
        frag_str = d["fragment"]
        if frag_str:
            try:
                domain_start, domain_end = parse_domain_range(frag_str)
            except Exception:
                pass

        def _ser(s):
            return ";".join(str(p) for p in sorted(s))

        # Legacy compatibility field: EARLY protection union, not nucleus truth.
        noyau_str = ",".join(str(p) for p in sorted(early_union)) if early_union else None

        rows.append({
            "Name":           _clean_name(d["name_raw"]) or f"STF_{pdb}",
            "PDB":            pdb,
            "Chain":          "A",
            "Size_aa":        n_res,
            "UniProt":        d["uniprot"] or None,
            "DomainStart":    domain_start,
            "DomainEnd":      domain_end,
            "DomainRange":    frag_str or None,
            "Noyau résidus":  noyau_str,
            # Folding-protection axis
            "start2fold_early":        _ser(early_union),
            "start2fold_intermediate": _ser(intermediate_union),
            "start2fold_late":         _ser(late_union),
            # Equilibrium protection/stability axis
            "start2fold_strong":       _ser(strong_union),
            "start2fold_medium":       _ser(medium_union),
            "start2fold_weak":         _ser(weak_union),
            # Metadata
            "n_early_experiments": len(d["early_sets"]),
            "_sequence_s2f": d["sequence"],
        })

    return pd.DataFrame(rows)


def load_merged_protein_list(
    v3_df: pd.DataFrame,
    json_path: str = None,
    max_size: int = _MAX_SIZE_DEFAULT,
) -> pd.DataFrame:
    """
    Return v3_df + Start2Fold proteins not already in v3, deduplicated by PDB
    (lowercase comparison). Start2Fold entries larger than max_size are excluded.
    """
    s2f_df = load_protein_list(json_path=json_path, max_size=max_size)

    v3_pdbs = set(v3_df["PDB"].str.lower())
    new_rows = s2f_df[~s2f_df["PDB"].str.lower().isin(v3_pdbs)].copy()

    # Align columns — add missing columns from v3 that Start2Fold doesn't have
    for col in v3_df.columns:
        if col not in new_rows.columns:
            new_rows[col] = None

    merged = pd.concat([v3_df, new_rows], ignore_index=True)
    return merged


def verify_pdb_sequence_alignment(pdb_sequence: str, s2f_sequence: str, pdb_id: str) -> None:
    """
    Hard blocker: raise ValueError if pdb_sequence doesn't match s2f_sequence
    character by character (case-insensitive, ignoring whitespace).
    Call this from the run pipeline before processing a Start2Fold protein.
    """
    pdb_clean = pdb_sequence.strip().upper()
    s2f_clean = s2f_sequence.strip().upper()
    if pdb_clean != s2f_clean:
        # Build diff summary (first mismatch position)
        min_len = min(len(pdb_clean), len(s2f_clean))
        first_mismatch = next(
            (i for i in range(min_len) if pdb_clean[i] != s2f_clean[i]),
            min_len,
        )
        raise ValueError(
            f"[start2fold_adapter] PDB sequence mismatch for {pdb_id}: "
            f"len(PDB)={len(pdb_clean)}, len(S2F)={len(s2f_clean)}, "
            f"first mismatch at position {first_mismatch + 1} "
            f"(PDB={pdb_clean[first_mismatch:first_mismatch+5]!r} vs "
            f"S2F={s2f_clean[first_mismatch:first_mismatch+5]!r})"
        )


def _parse_residues(residues_raw: str, sequence: str, pdb: str) -> set:
    """Parse 'A8;A9;E10' → {8, 9, 10}."""
    import warnings
    positions = set()
    if not residues_raw or residues_raw.lower() in ("nan", "none", ""):
        return positions
    for token in residues_raw.split(";"):
        token = token.strip()
        if not token:
            continue
        aa_letter = token[0] if token and token[0].isalpha() and token[0].isupper() else None
        pos_str = token[1:] if aa_letter else token
        try:
            pos = int(pos_str)
        except ValueError:
            continue
        if pos <= 0:
            continue
        if aa_letter and sequence and 1 <= pos <= len(sequence):
            expected = sequence[pos - 1].upper()
            if expected != aa_letter.upper():
                warnings.warn(
                    f"[start2fold_adapter] {pdb}: {token} — expected {expected} at pos {pos}",
                    stacklevel=3,
                )
        positions.add(pos)
    return positions


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(_HERE / "start2fold_data.json"))
    ap.add_argument("--max-size", type=int, default=180)
    ap.add_argument("--require-early", action="store_true")
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    df = load_protein_list(args.json, max_size=args.max_size, require_early=args.require_early)
    print(f"Proteins loaded: {len(df)}")
    cols = ["Name", "PDB", "Size_aa", "n_early_experiments",
            "start2fold_early", "start2fold_intermediate", "start2fold_late",
            "start2fold_strong", "start2fold_medium", "start2fold_weak"]
    if args.show_all:
        print(df[cols].to_string(index=False))
    else:
        print(df[cols].head(20).to_string(index=False))
