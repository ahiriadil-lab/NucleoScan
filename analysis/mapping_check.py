"""Check the three residue numberings per protein: predicted FASTA, native PDB chain, Start2Fold labels."""
import json
import sys
from pathlib import Path

import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parent
ALIGNER = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                open_gap_score=-4, extend_gap_score=-0.5,
                                end_gap_score=0)


def chain_seqs(pdb_path):
    model = PDBParser(QUIET=True).get_structure("x", str(pdb_path))[0]
    return {c.id: "".join(seq1(r.get_resname()) for r in c if is_aa(r, standard=True)) for c in model}


def align_stats(a, b):
    aln = ALIGNER.align(a, b)[0]
    pairs = [(i, j) for (qs, qe), (ts, te) in zip(*aln.aligned) for i, j in zip(range(qs, qe), range(ts, te))]
    ident = sum(a[i] == b[j] for i, j in pairs)
    return ident / max(1, min(len(a), len(b))), pairs


def main():
    records = json.load(open(ROOT / "data" / "start2fold_data.json"))
    s2f = {}
    for r in records:
        pdb = r["PDB code"].strip().lower()
        s2f.setdefault(pdb, {"seq": r["Sequence"].strip(), "tokens": []})
        s2f[pdb]["tokens"] += [t.strip() for t in str(r.get("Residues", "")).split(";") if t.strip()]
    rows = []
    for d in sorted((ROOT / "fragments").iterdir()):
        pdb = d.name.split("_")[-1]
        fasta = (ROOT / "results_rerun" / d.name / f"{pdb}.fasta").read_text().split("\n")[1].strip()
        s = s2f[pdb]
        ok = [fasta[int(t[1:]) - 1] == t[0] for t in s["tokens"]
              if t[0].isalpha() and t[1:].isdigit() and 0 < int(t[1:]) <= len(fasta)]
        out = sum(1 for t in s["tokens"] if t[1:].isdigit() and int(t[1:]) > len(fasta))
        id_fs, _ = align_stats(fasta, s["seq"])
        chains = chain_seqs(ROOT / "data" / "pdb_raw" / f"{pdb}.pdb")
        best = max(chains, key=lambda c: align_stats(fasta, chains[c])[0])
        used = chain_seqs(ROOT / "data" / "pdb_cleaned" / f"{pdb}_A.pdb")
        used_seq = next(iter(used.values()))
        id_used, _ = align_stats(fasta, used_seq)
        rows.append({"name": d.name, "pdb": pdb, "len_fasta": len(fasta), "len_s2f_seq": len(s["seq"]),
                     "fasta_eq_s2f": fasta == s["seq"], "identity_fasta_s2f": round(id_fs, 3),
                     "label_tokens": len(s["tokens"]), "label_checksum_ok": round(sum(ok) / max(1, len(ok)), 3),
                     "labels_beyond_fasta": out, "chain_used_identity": round(id_used, 3),
                     "best_chain": best, "best_chain_identity": round(align_stats(fasta, chains[best])[0], 3),
                     "positional_native_identity": round(sum(a == b for a, b in zip(fasta, used_seq)) / min(len(fasta), len(used_seq)), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "logs" / "numbering_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())


if __name__ == "__main__":
    main()
