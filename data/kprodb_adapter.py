"""Adapters for legacy and v3 protein-folding kinetics workbooks.

Residue annotations exposed by these adapters require source and numbering
audits before use; they are not folding-nucleus ground truth.
"""
import math
import re
import os
import pandas as pd

# Special-case: AF-P14621 has a non-standard PDB filename
_PDB_FILENAME_OVERRIDES = {
    "AF-P14621": "AF-P14621-F1-model_v6.pdb",
}

_NAME_CLEAN_RE = re.compile(r"[^\w]")


def _clean_name(name: str) -> str:
    """Convert protein name to filesystem-safe string (spaces → underscores)."""
    return _NAME_CLEAN_RE.sub("_", name.strip()).strip("_")


def load_protein_list(db_path: str) -> pd.DataFrame:
    """
    Parse EVO3_Strict_Database.xlsx (sheet 1) and return one row per protein.

    Returns a DataFrame with columns compatible with run_dataset.py:
        Name            -- cleaned protein name
        PDB             -- PDB identifier (e.g. '1fkb')
        Chain           -- chain letter (always 'A')
        Size_aa         -- protein length (int)
        UniProt         -- UniProt accession (or None)
        Noyau résidus   -- legacy residue-annotation field
    """
    df = pd.read_excel(db_path, sheet_name="EVO3 Strict Database",
                        header=3, engine="openpyxl")

    # Filter out category header rows (marked with "▶") and empty rows
    df = df[df["Protein Name"].notna()].copy()
    df = df[~df["Protein Name"].astype(str).str.startswith("▶")].copy()
    df = df[df["#"].notna()].copy()

    # Find columns dynamically (some have \n in their names)
    length_col = next((c for c in df.columns if "Length" in c), None)
    nucleus_col = next((c for c in df.columns if "Nucleus" in c and "résidus" in c), None)

    rows = []
    for _, src in df.iterrows():
        pdb_raw = str(src.get("PDB", "")).strip()
        uniprot_raw = str(src.get("UniProt", "")).strip()
        uniprot = uniprot_raw if uniprot_raw and uniprot_raw.lower() not in ("nan", "n/a", "") else None

        try:
            size = int(float(src[length_col])) if length_col else 0
        except (ValueError, TypeError):
            size = 0

        nucleus_raw = src.get(nucleus_col) if nucleus_col else None
        if pd.isna(nucleus_raw):
            nucleus_raw = None

        rows.append({
            "Name":          _clean_name(str(src["Protein Name"])),
            "PDB":           pdb_raw.lower(),
            "Chain":         "A",
            "Size_aa":       size,
            "UniProt":       uniprot,
            "Noyau résidus": nucleus_raw,
        })

    return pd.DataFrame(rows)


def parse_domain_range(raw: str) -> tuple:
    """
    Parse domain range from "Domain in UniProt" column.

    Input formats: "1–67", "228–282 (B1)", "21–83 (mature)", etc.
    Returns: (start, end) as 1-based inclusive integers.
    """
    if not raw or pd.isna(raw):
        return None

    raw = str(raw).strip()
    # Remove parenthetical annotations
    raw = re.sub(r'\s*\(.*\)', '', raw).strip()

    # Split on various dash characters (–, -, —)
    parts = re.split(r'[–\-\u2013\u2014]+', raw)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) != 2:
        raise ValueError(f"Cannot parse domain range: {raw}")

    try:
        start = int(parts[0])
        end = int(parts[1])
        return (start, end)
    except ValueError as e:
        raise ValueError(f"Cannot parse domain range {raw}: {e}")


def load_v3_protein_list(db_path: str) -> pd.DataFrame:
    """
    Parse two_state_folding_v3.xlsx and return protein list.

    Returns DataFrame with columns:
        Name, PDB, UniProt, Chain, Size_aa, DomainStart, DomainEnd, DomainRange
    """
    df = pd.read_excel(db_path, sheet_name="Two-State Folding + Coverage",
                        header=0, engine="openpyxl")

    # Filter out empty rows
    df = df[df["Protein"].notna()].copy()

    # Dynamic column lookup with substring matching
    pdb_col = "PDB"
    uniprot_col = "UniProt"
    chain_col = next((c for c in df.columns if "PDB chain" in c), None)
    size_col = next((c for c in df.columns if "Domain" in c and "length" in c), None)
    domain_col = next((c for c in df.columns if "Domain" in c and "UniProt" in c), None)
    high_phi_col = next((c for c in df.columns if "High" in c and "phi" in c.lower() or "φ" in c), None)
    kf_col = next((c for c in df.columns if "kf" in c and "ln" not in c), None)
    ln_kf_col = next((c for c in df.columns if "ln(kf)" in c), None)
    bt_col = next((c for c in df.columns if "βT" in c or "beta_T" in c or "BetaT" in c), None)
    mechanism_col = next((c for c in df.columns if "Mechanism" in c), None)

    rows = []
    for _, src in df.iterrows():
        pdb_raw = str(src.get(pdb_col, "")).strip()
        uniprot_raw = str(src.get(uniprot_col, "")).strip()
        chain_raw = str(src.get(chain_col, "A")).strip() if chain_col else "A"
        size_raw = src.get(size_col) if size_col else 0
        domain_raw = src.get(domain_col) if domain_col else None
        high_phi_raw = src.get(high_phi_col) if high_phi_col else None
        kf_raw = src.get(kf_col) if kf_col else None
        ln_kf_raw = src.get(ln_kf_col) if ln_kf_col else None
        bt_raw = src.get(bt_col) if bt_col else None
        mechanism_raw = src.get(mechanism_col) if mechanism_col else None

        # Convert em-dash to None
        uniprot = None if uniprot_raw in ("—", "nan", "N/A", "") else uniprot_raw
        chain = None if chain_raw in ("—", "nan") else chain_raw

        try:
            size = int(float(size_raw)) if size_raw else 0
        except (ValueError, TypeError):
            size = 0

        # Parse domain range
        domain_start = None
        domain_end = None
        if domain_raw and not pd.isna(domain_raw):
            try:
                domain_start, domain_end = parse_domain_range(domain_raw)
            except Exception:
                pass

        # Kinetic scalars
        def _safe_float(v):
            try:
                return float(v) if v is not None and not pd.isna(v) else None
            except (TypeError, ValueError):
                return None

        rows.append({
            "Name": _clean_name(str(src["Protein"])),
            "PDB": pdb_raw.lower(),
            "UniProt": uniprot,
            "Chain": chain,
            "Size_aa": size,
            "DomainStart": domain_start,
            "DomainEnd": domain_end,
            "DomainRange": str(domain_raw) if domain_raw else None,
            # Legacy high-Φ/reference annotation (used descriptively only).
            "Noyau résidus": str(high_phi_raw).strip() if high_phi_raw is not None and not (isinstance(high_phi_raw, float) and pd.isna(high_phi_raw)) else None,
            # Kinetic data
            "kf": _safe_float(kf_raw),
            "ln_kf": _safe_float(ln_kf_raw),
            "betaT": _safe_float(bt_raw),
            "Mechanism": str(mechanism_raw).strip() if mechanism_raw is not None and not (isinstance(mechanism_raw, float) and pd.isna(mechanism_raw)) else None,
        })

    df_out = pd.DataFrame(rows)
    return df_out


def extract_phi_tables(db_path: str) -> dict:
    """
    Parse EVO3_Strict_Database.xlsx (sheet 2: "Phi-values par résidu")
    and return per-protein phi-value tables.

    Returns:
        {pdb_id: {residue_pos: phi, ...}, ...}
    """
    # Build protein name -> PDB ID mapping from sheet 1
    df_sheet1 = pd.read_excel(db_path, sheet_name="EVO3 Strict Database",
                               header=3, engine="openpyxl")
    df_sheet1 = df_sheet1[df_sheet1["Protein Name"].notna()].copy()
    df_sheet1 = df_sheet1[~df_sheet1["Protein Name"].astype(str).str.startswith("▶")].copy()

    # Build mapping: full protein name -> PDB ID
    full_name_to_pdb = {}
    for _, r in df_sheet1.iterrows():
        full_name = str(r["Protein Name"]).strip()
        pdb_id = str(r["PDB"]).strip().lower()
        full_name_to_pdb[full_name] = pdb_id

    # Read sheet 2
    df = pd.read_excel(db_path, sheet_name="Phi-values par résidu",
                        header=2, engine="openpyxl")

    records: dict[str, dict[int, list[float]]] = {}

    for _, row in df.iterrows():
        prot_name = str(row.get("Protéine", "")).strip()
        if not prot_name or prot_name.lower() == "nan":
            continue

        # Resolve to PDB ID via exact match or substring match
        pdb_id = full_name_to_pdb.get(prot_name)

        # Fallback 1: substring matching (short name in full names, case-insensitive)
        if pdb_id is None:
            for full_name, pid in full_name_to_pdb.items():
                if prot_name.lower() in full_name.lower():
                    pdb_id = pid
                    break

        # Fallback 2: match by first word of short name (unique match required)
        if pdb_id is None:
            first_word = prot_name.lower().split()[0]
            candidates = [pid for fn, pid in full_name_to_pdb.items()
                          if first_word in fn.lower().split()]
            if len(candidates) == 1:
                pdb_id = candidates[0]

        if pdb_id is None:
            continue

        # Reliability filter
        reliable = str(row.get("Φ fiable?", "")).strip()
        if reliable.lower() == "no":
            continue

        # Parse phi value
        phi_raw = row.get("Φ exp.")
        try:
            phi = float(phi_raw)
        except (ValueError, TypeError):
            continue

        if math.isnan(phi) or phi < -0.5 or phi > 1.5:
            continue

        # Parse residue position
        try:
            pos = int(float(row.get("Rés. n°", 0)))
        except (ValueError, TypeError):
            continue
        if pos <= 0:
            continue

        records.setdefault(pdb_id, {}).setdefault(pos, []).append(phi)

    # Average phi-values per position (handles any duplicates)
    return {
        pdb_id: {pos: sum(vals) / len(vals) for pos, vals in pos_dict.items()}
        for pdb_id, pos_dict in records.items()
        if pos_dict
    }


if __name__ == "__main__":
    xlsx = os.path.join(os.path.dirname(__file__), "EVO3_Strict_Database.xlsx")
    proteins = load_protein_list(xlsx)
    print(f"Proteins loaded: {len(proteins)}")
    print(proteins[["Name", "PDB", "Chain", "Size_aa"]].to_string(index=False))

    phi_tables = extract_phi_tables(xlsx)
    print(f"\nProteins with phi-values: {len(phi_tables)}")
    for pid, table in list(phi_tables.items())[:3]:
        print(f"  {pid}: {len(table)} residues — {dict(list(table.items())[:5])}")
