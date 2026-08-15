"""
Legacy candidate-set comparison with residue annotations in the optional
kinetics dataset.

The names of this module, its functions and several output fields are retained
for backward compatibility. Operationally, ``is_nucleus`` is a thresholded
fragment-emergence candidate flag. The reference annotations are heterogeneous
high-Φ or literature-derived residue lists, not folding-nucleus ground truth.
Consequently, the overlap statistics produced here are descriptive and must
not be reported as independent validation without auditing the source,
construct, residue mapping and selection criteria for every protein.

Outputs per protein:
  results/{protein}/nucleus_validation.csv  — legacy filename; candidate/reference labels
  results/{protein}/nucleus_validation.txt  — legacy filename; descriptive overlap report

Usage:
  python validate_nucleus.py [--protein Villin_HP36] [--results-dir results]
  python validate_nucleus.py --all                 # all proteins with annotations
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

# Structural element tokens that carry numbers but are NOT residue positions.
# e.g. "Loop1" → label for loop 1; "β2" → name of β-strand 2.
# Greek-letter tokens also absorb optional range suffixes (e.g. "β1-2", "β1–5")
# so that the trailing range number is not mistakenly extracted as a residue.
_STRUCTURAL_TOKEN_RE = re.compile(
    r'\b(?:Loop|Turn|Helix|Strand|alpha|Alpha|beta|Beta)\d*(?:[-–]\d+)?'  # English labels ± range
    r'|\b[Hh]\d+\b'                                                        # helix shorthand H1, H2
    r'|\b\d{3}-helix\b'                                                    # 310-helix
    r'|[αβ]\d*(?:[-–]\d+)?',                                               # Greek letters ± range
    re.IGNORECASE | re.UNICODE,
)
# Decimal φ-values like "0.40", "0.95" that can leak as "0" + "40"
_DECIMAL_RE = re.compile(r'\b\d+\.\d+\b')

# Valid residue tokens — two patterns tried in order:
# 1. Three-letter AA code (Ala, Gly, …) immediately followed by an integer: "Ala16", "Lys5"
# 2. Optional single-letter AA code followed by an integer (with word boundary): "F47", "47"
_3LETTER_AA = (
    "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val"
)
_RESIDUE_3LETTER_RE = re.compile(rf'\b(?:{_3LETTER_AA})(\d+)\b', re.IGNORECASE)
# φ-value parenthetical "(0.95)", "(≈1)", "(φ ≈ 1.0)" — absorb so the decimal doesn't leak
_PHI_PAREN_RE = re.compile(r'\([^)]*(?:φ|≈|[0-9]\.[0-9])[^)]*\)')
_RESIDUE_TOKEN_RE = re.compile(r'\b[A-Za-z]?(\d+)\b')


def parse_nucleus_annotation(nucleus_str) -> set:
    """
    Parse the 'Noyau résidus' annotation string into a set of integer residue
    positions (resSeq numbers as they appear in the PDB / annotation).

    Handles formats:
        "F47,F51,F58"        → {47, 51, 58}
        "47,51,58"           → {47, 51, 58}
        "A16-L49,I57"        → {16, 49, 57}
        "Loop1,W11,W34"      → {11, 34}   (Loop1 is a structural label, rejected)
        "Central β1-β5-β2"   → {}          (all tokens structural, return empty)
        "β1-2+β1-5 NC"       → {}          (same)
        NaN or empty         → {}
    """
    if nucleus_str is None or (isinstance(nucleus_str, float) and np.isnan(nucleus_str)):
        return set()
    s = str(nucleus_str).strip()
    if not s or s.lower() in ("nan", "nd", "n/a", "?", "-"):
        return set()

    # Remove φ-value parentheticals and standalone decimals to prevent leaking.
    s = _PHI_PAREN_RE.sub(" ", s)
    s = _DECIMAL_RE.sub(" ", s)

    # Extract three-letter AA codes (Ala16, Lys5, …) before structural filtering.
    three_letter_hits = {int(m.group(1)) for m in _RESIDUE_3LETTER_RE.finditer(s)}

    # Remove structural-element tokens so their embedded numbers don't leak.
    cleaned = _STRUCTURAL_TOKEN_RE.sub(" ", s)

    # Remove three-letter AA tokens to avoid their letters matching _RESIDUE_TOKEN_RE.
    cleaned = _RESIDUE_3LETTER_RE.sub(" ", cleaned)

    # Extract single-letter or bare-number residue tokens.
    single_hits = {int(m.group(1)) for m in _RESIDUE_TOKEN_RE.finditer(cleaned)}

    numbers = three_letter_hits | single_hits

    if not numbers and cleaned.strip() != s.strip():
        # All content was structural labels — annotation is not residue-specific.
        warnings.warn(
            f"parse_nucleus_annotation: annotation appears purely structural "
            f"(no residue numbers extracted): {nucleus_str!r}",
            UserWarning, stacklevel=2,
        )
    return numbers


# ---------------------------------------------------------------------------
# resSeq ↔ pipeline-index mapping
# ---------------------------------------------------------------------------

def get_resseq_mapping(pdb_path: str) -> dict:
    """
    Build a mapping {resSeq: pipeline_index} for a cleaned PDB file.

    MDTraj (and the folding pipeline) renumbers residues as consecutive
    1-based indices, while PDB files use the original resSeq numbers which
    may start at any value (e.g. Villin starts at 41).  This mapping allows
    experimental annotations (written in resSeq space) to be converted to
    the pipeline's index space before metric computation.

    Parameters
    ----------
    pdb_path : str — path to a *_clean.pdb file

    Returns
    -------
    dict  {resSeq_int: pipeline_1based_index}
          e.g. for Villin_HP36: {41: 1, 42: 2, ..., 76: 36}
    """
    from Bio import PDB as BioPDB
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        warnings.warn(f"get_resseq_mapping: could not parse {pdb_path}: {e}", UserWarning)
        return {}

    model = structure[0]
    chain = next(iter(model.get_chains()))
    residues = [r for r in chain.get_residues() if BioPDB.is_aa(r, standard=True)]

    mapping = {}
    for pipeline_idx, residue in enumerate(residues, start=1):
        resseq = residue.get_id()[1]   # integer resSeq field
        mapping[resseq] = pipeline_idx
    return mapping


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_nucleus_metrics(predicted_nucleus: list, ann_row,
                            resseq_mapping: dict = None) -> dict:
    """
    Compare a thresholded candidate set with a residue-level reference list.

    Parameters
    ----------
    predicted_nucleus : list of int — legacy ``is_nucleus=True`` candidates,
                        in pipeline 1-based consecutive indexing.
    ann_row           : pandas Series or dict containing 'Noyau résidus' key.
                        Residue numbers in the annotation are PDB resSeq values.
    resseq_mapping    : dict {resSeq: pipeline_index} from get_resseq_mapping().
                        If provided, reference resSeq numbers are converted to
                        pipeline indices before comparison.  If None, residue numbers
                        are used as-is (only valid when resSeq == pipeline index).

    Returns
    -------
    dict with keys: precision, recall, f1, jaccard, n_predicted, n_experimental,
                    n_overlap, positional_distance
    Legacy return keys containing ``experimental`` are preserved. Returns None
    if no residue-specific reference annotation is available.
    """
    nucleus_str = None
    for col in ["Noyau résidus", "Noyau_residus", "nucleus_residues", "Nucleus",
                "Nucleus résidus (Φ>0.5 exp.)"]:
        if ann_row is not None and col in ann_row and ann_row[col] is not None:
            nucleus_str = ann_row[col]
            break

    experimental_resseq = parse_nucleus_annotation(nucleus_str)
    if not experimental_resseq:
        return None

    # Convert resSeq → pipeline indices when a mapping is available.
    if resseq_mapping:
        experimental = set()
        for rs in experimental_resseq:
            if rs in resseq_mapping:
                experimental.add(resseq_mapping[rs])
            else:
                warnings.warn(
                    f"compute_nucleus_metrics: resSeq {rs} not found in mapping "
                    f"(annotation may reference a non-standard residue)", UserWarning
                )
        if not experimental:
            # All annotation residues were unmappable — skip.
            return None
    else:
        experimental = experimental_resseq

    predicted = set(predicted_nucleus)

    overlap      = predicted & experimental
    n_pred       = len(predicted)
    n_exp        = len(experimental)
    n_overlap    = len(overlap)

    precision = n_overlap / n_pred        if n_pred  > 0 else 0.0
    recall    = n_overlap / n_exp         if n_exp   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    union     = predicted | experimental
    jaccard   = n_overlap / len(union)    if union   else 0.0

    # Positional distance: mean distance from each experimental residue to
    # its nearest predicted residue (and vice versa), averaged
    pos_dist = float("nan")
    if predicted and experimental:
        pred_arr = np.array(sorted(predicted))
        exp_arr  = np.array(sorted(experimental))
        d_e2p = np.mean([min(abs(e - p) for p in pred_arr) for e in exp_arr])
        d_p2e = np.mean([min(abs(p - e) for e in exp_arr) for p in pred_arr])
        pos_dist = float((d_e2p + d_p2e) / 2)

    return {
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "jaccard":             jaccard,
        "n_predicted":         n_pred,
        "n_experimental":      n_exp,
        "n_overlap":           n_overlap,
        "positional_distance": pos_dist,
        "experimental_set":    sorted(experimental),
        "predicted_set":       sorted(predicted),
        "overlap_set":         sorted(overlap),
    }


# ---------------------------------------------------------------------------
# Per-protein runner
# ---------------------------------------------------------------------------

def run(protein_name: str, results_dir: str, ann_row=None, pdb_path: str = None):
    """
    Run the legacy descriptive candidate/reference comparison for one protein.

    Reads residue_scores.csv from results_dir/protein_name/, parses is_nucleus,
    compares to annotation in ann_row.

    Parameters
    ----------
    protein_name : str
    results_dir  : str — base results directory
    ann_row      : pandas Series containing the annotation row (optional; loaded
                   from data/EVO3_Strict_Database.xlsx if not provided)
    pdb_path     : str — path to the cleaned PDB file.  When provided, a
                   resSeq→pipeline_index mapping is built so that experimental
                   annotations (written in PDB resSeq numbering) are correctly
                   compared to the pipeline's consecutive 1-based indices.
    """
    print(f"\n=== Exploratory candidate/reference comparison for {protein_name} ===")
    prot_dir = os.path.join(results_dir, protein_name)

    ns_path = os.path.join(prot_dir, "residue_scores.csv")
    if not os.path.exists(ns_path):
        print(f"  residue_scores.csv not found: {ns_path}")
        return None

    df = pd.read_csv(ns_path)
    if "is_nucleus" not in df.columns:
        print("  No 'is_nucleus' column found.")
        return None

    predicted_nucleus = df[df["is_nucleus"] == True]["residue"].astype(int).tolist()
    print(f"  Thresholded FES candidates: {predicted_nucleus}")

    if ann_row is None:
        # Try to load annotations from the included v3 workbook.
        base = os.path.dirname(os.path.dirname(os.path.abspath(prot_dir)))
        xlsx_path = os.path.join(base, "data", "two_state_folding_v3.xlsx")
        if os.path.exists(xlsx_path):
            try:
                from data.kprodb_adapter import load_v3_protein_list
                df_ann = load_v3_protein_list(xlsx_path)
                row_matches = df_ann[df_ann["Name"] == protein_name]
                if not row_matches.empty:
                    ann_row = row_matches.iloc[0]
            except Exception:
                pass

    # Build resSeq→pipeline mapping from the PDB file when available.
    resseq_mapping = None
    if pdb_path and os.path.exists(pdb_path):
        resseq_mapping = get_resseq_mapping(pdb_path)
        if resseq_mapping:
            min_rs, max_rs = min(resseq_mapping), max(resseq_mapping)
            print(f"  resSeq mapping: {len(resseq_mapping)} residues "
                  f"(resSeq {min_rs}–{max_rs} → pipeline 1–{len(resseq_mapping)})")
    elif pdb_path:
        print(f"  Warning: pdb_path not found: {pdb_path}")

    metrics = compute_nucleus_metrics(predicted_nucleus, ann_row,
                                      resseq_mapping=resseq_mapping)

    if metrics is None:
        print("  No residue-specific reference annotation — comparison skipped.")
        return None

    print(f"  Reference residue list: {metrics['experimental_set']}")
    print(f"  Overlap: {metrics['overlap_set']}")
    print(f"  Precision: {metrics['precision']:.3f}  "
          f"Recall: {metrics['recall']:.3f}  "
          f"F1: {metrics['f1']:.3f}  "
          f"Jaccard: {metrics['jaccard']:.3f}")
    print(f"  Positional distance: {metrics['positional_distance']:.1f} residues")

    # Save per-residue CSV
    df["is_reference_high_phi"] = df["residue"].astype(int).isin(
        metrics["experimental_set"]
    )
    # Legacy alias retained for consumers of earlier result archives.
    df["is_experimental_nucleus"] = df["is_reference_high_phi"]
    df["is_overlap"] = df["residue"].astype(int).isin(metrics["overlap_set"])
    csv_path = os.path.join(prot_dir, "nucleus_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Validation CSV → {csv_path}")

    # Save text report
    txt_path = os.path.join(prot_dir, "nucleus_validation.txt")
    lines = [
        f"Exploratory Candidate/Reference Report — {protein_name}",
        "=" * 55,
        f"Thresholded candidates: {metrics['predicted_set']}",
        f"Reference residues    : {metrics['experimental_set']}",
        f"Overlap            : {metrics['overlap_set']}",
        "",
        f"Precision : {metrics['precision']:.3f}  ({metrics['n_overlap']}/{metrics['n_predicted']} candidates overlap)",
        f"Recall    : {metrics['recall']:.3f}  ({metrics['n_overlap']}/{metrics['n_experimental']} references overlap)",
        f"F1-score  : {metrics['f1']:.3f}",
        f"Jaccard   : {metrics['jaccard']:.3f}",
        f"Positional distance: {metrics['positional_distance']:.1f} residues (mean nearest-neighbour)",
        "",
        "Interpretation:",
        "  Descriptive overlap only; no universal F1 threshold is prespecified.",
        "  This report does not establish folding-nucleus membership.",
        "  Audit reference provenance and residue mapping before scientific use.",
    ]
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Validation report → {txt_path}")

    return metrics


# ---------------------------------------------------------------------------
# Dataset-wide runner
# ---------------------------------------------------------------------------

def run_all(results_dir: str, annot_path: str = None, pdb_dir: str = None):
    """Run the descriptive comparison for proteins with annotations.

    Parameters
    ----------
    results_dir  : str — base results directory
    annot_path   : str — path to EVO3_Strict_Database.xlsx (auto-detected if None)
    pdb_dir      : str — directory containing cleaned ``{pdb}_{chain}.pdb`` files.
                   When provided, resSeq→pipeline mapping is built per protein
                   so experimental annotations are correctly compared.
    """
    if annot_path is None:
        base = os.path.dirname(os.path.abspath(results_dir))
        annot_path = os.path.join(base, "data", "two_state_folding_v3.xlsx")

    if not os.path.exists(annot_path):
        print(f"Annotation file not found: {annot_path}")
        return

    from data.kprodb_adapter import load_v3_protein_list
    df_ann = load_v3_protein_list(annot_path)
    rows = []
    for _, ann_row in df_ann.iterrows():
        name = ann_row["Name"]
        prot_dir = os.path.join(results_dir, name)
        if not os.path.isdir(prot_dir):
            continue

        # Resolve PDB path for resSeq mapping.
        pdb_path = None
        if pdb_dir and "PDB" in ann_row:
            chain_raw = ann_row.get("Chain", "A")
            chain = "A" if pd.isna(chain_raw) or str(chain_raw).strip() in ("", "—", "-") else str(chain_raw).strip()
            candidate = os.path.join(pdb_dir, f"{ann_row['PDB']}_{chain}.pdb")
            if os.path.exists(candidate):
                pdb_path = candidate

        metrics = run(name, results_dir, ann_row=ann_row, pdb_path=pdb_path)
        if metrics:
            rows.append({"protein": name, **{k: v for k, v in metrics.items()
                                              if not isinstance(v, list)}})

    if rows:
        summary = pd.DataFrame(rows)
        out = os.path.join(results_dir, "nucleus_validation_summary.csv")
        summary.to_csv(out, index=False)
        print(f"\nExploratory comparison summary saved to {out}")
        print(summary[["protein", "precision", "recall", "f1",
                        "jaccard", "positional_distance"]].to_string(index=False))
        avg_f1 = summary["f1"].mean()
        print(f"\nMean descriptive F1-score: {avg_f1:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein",      default=None,
                    help="Single protein name (e.g. Villin_HP36)")
    ap.add_argument("--all",          action="store_true",
                    help="Compare all proteins with residue-specific annotations")
    ap.add_argument("--results-dir",  default="results")
    ap.add_argument("--pdb-dir",      default=None,
                    help="Directory containing *_clean.pdb files (enables resSeq mapping)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rdir = (args.results_dir if os.path.isabs(args.results_dir)
            else os.path.join(base, args.results_dir))

    # Default cleaned structures directory used by run_dataset.py.
    pdb_dir = args.pdb_dir or os.path.join(base, "data", "pdb_cleaned")
    if not os.path.isdir(pdb_dir):
        pdb_dir = None

    if args.all:
        run_all(rdir, pdb_dir=pdb_dir)
    elif args.protein:
        # Resolve pdb_path for single-protein run.
        pdb_path = None
        if pdb_dir:
            # Try to find the PDB by scanning the directory for matching prefix.
            import glob as _glob
            candidates = _glob.glob(os.path.join(pdb_dir, f"{args.protein}_*_clean.pdb"))
            if candidates:
                pdb_path = candidates[0]
        run(args.protein, rdir, pdb_path=pdb_path)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
