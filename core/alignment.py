"""
alignment.py — Utilities for aligning UniProt sequences to PDB sequences.

The EVO3 pipeline generates N-terminal fragments from UniProt sequences
(which are complete) and compares folded fragments against the PDB reference
structure (which may have missing N-terminal residues).

This module computes the offset between the two sequences so that
analyze_fragment() can correctly slice both the reference PDB and the
ESMFold model when they have different lengths.

Key insight for N-terminal fragments:
  - UniProt positions 0..L-1 are in fragment of length L
  - PDB residues start at UniProt position pdb_offset
  - PDB residues covered by fragment = first K = max(0, L - pdb_offset) PDB residues
  - These are always the FIRST K consecutive PDB residues (indices 0..K-1)
  - So native_pairs filtering (pairs < K) works without any remapping
"""


def compute_alignment(uniprot_seq: str, pdb_seq: str) -> dict:
    """
    Find where the PDB sequence sits within the UniProt sequence.

    Attempts a simple substring match first (handles most cases where the PDB
    sequence is a contiguous region of UniProt).  Falls back to a global
    pairwise alignment (Bio.pairwise2) when the simple match fails — e.g.
    when the PDB structure contains an engineered mutation (Y17W, F166W, …).

    Parameters
    ----------
    uniprot_seq : str
        Full UniProt sequence (single-letter codes).
    pdb_seq : str
        Sequence extracted from the cleaned PDB file (standard AA only).

    Returns
    -------
    dict with keys:
        pdb_offset   : int   — 0-based index in uniprot_seq where pdb_seq starts
        pdb_n_res    : int   — len(pdb_seq)
        uniprot_n_res: int   — len(uniprot_seq)
        method       : str   — "substring" or "alignment"
    """
    # --- Fast path: exact substring match ---
    idx = uniprot_seq.find(pdb_seq)
    if idx >= 0:
        return {
            "pdb_offset":    idx,
            "pdb_n_res":     len(pdb_seq),
            "uniprot_n_res": len(uniprot_seq),
            "method":        "substring",
        }

    # --- Fallback: allow 1-2 mismatches via pairwise alignment ---
    # This handles engineered mutations in the PDB (e.g. Y17W, F166W).
    try:
        from Bio import pairwise2
        alignments = pairwise2.align.localms(
            uniprot_seq, pdb_seq,
            match=2, mismatch=-1, open=-5, extend=-0.5,
            one_alignment_only=True,
        )
        if alignments:
            aln = alignments[0]
            # aln.start is the start index in uniprot_seq (local alignment)
            offset = aln.start
            return {
                "pdb_offset":    offset,
                "pdb_n_res":     len(pdb_seq),
                "uniprot_n_res": len(uniprot_seq),
                "method":        "alignment",
            }
    except Exception:
        pass

    # --- Last resort: use pdb_offset = 0 (assume sequences start together) ---
    return {
        "pdb_offset":    0,
        "pdb_n_res":     len(pdb_seq),
        "uniprot_n_res": len(uniprot_seq),
        "method":        "fallback",
    }


def pdb_covered_length(alignment: dict, fragment_length: int) -> int:
    """
    Number of PDB residues covered by a UniProt-based N-terminal fragment.

    For a fragment of length `fragment_length` (UniProt positions 0..L-1):
      - PDB sequence starts at UniProt position `pdb_offset`
      - Covered PDB residues = positions pdb_offset..min(L-1, pdb_offset+pdb_n_res-1)
      - Count K = max(0, min(L - pdb_offset, pdb_n_res))

    Returns 0 if the fragment does not yet reach the PDB region.
    """
    pdb_offset = alignment["pdb_offset"]
    pdb_n_res  = alignment["pdb_n_res"]
    return max(0, min(fragment_length - pdb_offset, pdb_n_res))
