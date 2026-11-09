"""Sequence-aligned residue mapping for the revision (corrects v1.1.0 positional mapping).

All quantities are expressed on the predicted-sequence positions (the prefix-scan frame):

- native reference: chain chosen by sequence identity, residues mapped to sequence
  positions by global alignment; Q, RMSD, reference SS, topology axis and native
  contact degree use mapped residue pairs only;
- Start2Fold labels: each experiment's residue numbers refer to that experiment's own
  Start2Fold sequence (constructs differ between experiments of one protein, e.g. 1rg8);
  they are mapped to sequence positions through a global alignment of that sequence,
  keeping only labels whose amino-acid letter matches; classes are then united across
  experiments as in v1.1.0;
- evaluable positions: sequence positions aligned to any Start2Fold construct.

Deviation from v1.1.0 semantics: RMSD is computed on the backbone atoms (N, CA, C, O)
of mapped residue pairs instead of all backbone atoms of the first k residues.
"""
from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from Bio import Align

from core.analyze import (CONTACT_CUTOFF_NM, _load_traj, compute_hydrophobic_rg,
                          compute_q_value)
from modules.importance import (compute_betweenness, compute_contact_degree, compute_importance,
                                compute_ss_persistence)
from run_esmfold import extract_plddt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_rerun"
STRUCTURES = ROOT / os.environ.get("NS_STRUCTURES", "structures_rerun")
MAPPED_PDB = ROOT / "data" / "pdb_mapped"
ALIGNER = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                open_gap_score=-4, extend_gap_score=-0.5, end_gap_score=0)
BACKBONE = ("N", "CA", "C", "O")


def aligned_pairs(a: str, b: str):
    aln = ALIGNER.align(a, b)[0]
    return [(i, j) for (qs, qe), (ts, te) in zip(*aln.aligned) for i, j in zip(range(qs, qe), range(ts, te))]


def fasta_seq(name):
    pdb = name.split("_")[-1]
    return (RESULTS / name / f"{pdb}.fasta").read_text().split("\n")[1].strip()


@lru_cache(maxsize=None)
def s2f_records():
    out = {}
    for r in json.load(open(ROOT / "data" / "start2fold_data.json", encoding="utf-8")):
        pdb = r["PDB code"].strip().lower()
        out.setdefault(pdb, {"seq": r["Sequence"].strip(), "tokens": []})
        out[pdb]["tokens"] += [t.strip() for t in str(r.get("Residues", "")).split(";")
                               if t.strip()[:1].isalpha() and t.strip()[1:].isdigit()]
    return out


@lru_cache(maxsize=None)
def s2f_experiments(pdb):
    """Start2Fold records of one PDB entry: (class, own sequence, [(aa, position)])."""
    out = []
    for r in json.load(open(ROOT / "data" / "start2fold_data.json", encoding="utf-8")):
        if r["PDB code"].strip().lower() != pdb:
            continue
        toks = [(t.strip()[0], int(t.strip()[1:])) for t in str(r.get("Residues", "")).split(";")
                if t.strip()[:1].isalpha() and t.strip()[1:].isdigit()]
        out.append((str(r.get("Protection Level", "")).strip().lower(), r["Sequence"].strip(), toks))
    return out


def native_traj(name):
    """Native chain with the highest sequence identity to the predicted sequence."""
    pdb = name.split("_")[-1]
    seq = fasta_seq(name)
    MAPPED_PDB.mkdir(parents=True, exist_ok=True)
    out = MAPPED_PDB / f"{pdb}.pdb"
    if not out.exists():
        from core.download import clean_pdb
        from Bio.PDB import PDBParser, is_aa
        from Bio.SeqUtils import seq1
        model = PDBParser(QUIET=True).get_structure(pdb, str(ROOT / "data" / "pdb_raw" / f"{pdb}.pdb"))[0]

        def identity(chain):
            s = "".join(seq1(r.get_resname()) for r in chain if is_aa(r, standard=True))
            return sum(seq[i] == s[j] for i, j in aligned_pairs(seq, s)) / max(1, min(len(seq), len(s)))
        best = max(model.get_chains(), key=identity).id
        clean_pdb(str(ROOT / "data" / "pdb_raw" / f"{pdb}.pdb"), str(out), chain_id=best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = _load_traj(str(out))
    return traj.atom_slice(traj.topology.select("protein"))


@lru_cache(maxsize=None)
def mapping(name):
    """native residue index -> sequence index (0-based), plus evaluable positions and labels."""
    seq = fasta_seq(name)
    ref = native_traj(name)
    nat_seq = "".join(r.code or "X" for r in ref.topology.residues)
    nat2seq = {j: i for i, j in aligned_pairs(seq, nat_seq)}

    pdb = name.split("_")[-1]
    labels, construct, offsets = {}, set(), []
    n_tok = n_ok = 0
    for level, rseq, toks in s2f_experiments(pdb):
        r2s = {j: i for i, j in aligned_pairs(seq, rseq)}
        construct |= set(r2s.values())
        for aa, p in toks:
            n_tok += 1
            i = r2s.get(p - 1)
            if i is not None and seq[i] == aa:
                n_ok += 1
                labels.setdefault(level, set()).add(i + 1)
                offsets.append(i - (p - 1))
    return {"seq": seq, "nat2seq": nat2seq, "construct": construct, "labels": labels,
            "label_offset": int(max(offsets, key=abs)) if offsets else 0,
            "label_checksum": n_ok / n_tok if n_tok else np.nan, "labels_dropped": n_tok - n_ok,
            "native_identity": sum(seq[i] == nat_seq[j] for j, i in nat2seq.items()) / max(1, min(len(seq), len(nat_seq))),
            "native_mapped": len(nat2seq), "native_len": len(nat_seq)}


def mapped_labels(name, level) -> set:
    """1-based sequence positions protected at one Start2Fold class (union over experiments)."""
    return mapping(name)["labels"].get(level, set())


def evaluable_mask(name):
    m = mapping(name)
    mask = np.zeros(len(m["seq"]), dtype=bool)
    mask[sorted(m["construct"])] = True
    return mask


def native_contacts_seq(name):
    """Native heavy-atom contacts (v1.1.0 definition) re-indexed to sequence positions."""
    ref = native_traj(name)
    nat2seq = mapping(name)["nat2seq"]
    dist, pairs = md.compute_contacts(ref, scheme="closest-heavy")
    pairs = pairs[dist[0] <= CONTACT_CUTOFF_NM]
    mapped = np.array([(nat2seq[a], nat2seq[b]) for a, b in pairs if a in nat2seq and b in nat2seq], dtype=int)
    return mapped.reshape(-1, 2)


def mapped_fragment_metrics(name):
    """Per-fragment metrics as in core.analyze.analyze_fragment, on mapped residues."""
    m = mapping(name)
    ref = native_traj(name)
    n = len(m["seq"])
    pairs = native_contacts_seq(name)
    total = len(pairs)
    seq2nat = {i: j for j, i in m["nat2seq"].items()}
    ss_nat = md.compute_dssp(ref, simplified=True)[0]
    ss_seq = np.array([ss_nat[seq2nat[i]] if i in seq2nat else "-" for i in range(n)])

    def bb(top, res_idx):
        res = top.residue(res_idx)
        return {a.name: a.index for a in res.atoms if a.name in BACKBONE}

    rows = []
    for k in range(3, n + 1):
        pdb_path = STRUCTURES / name / f"frag{k:02d}" / "decoy_00000.pdb"
        if not pdb_path.exists():
            continue  # rejected at prediction (mean pLDDT < 40), as in v1.1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _load_traj(str(pdb_path))
        plddt_mean, _ = extract_plddt(str(pdb_path))
        idx_m, idx_r = [], []
        for i in range(min(k, model.topology.n_residues)):
            if i not in seq2nat:
                continue
            am, ar = bb(model.topology, i), bb(ref.topology, seq2nat[i])
            for a in BACKBONE:
                if a in am and a in ar:
                    idx_m.append(am[a])
                    idx_r.append(ar[a])
        rmsd = float(md.rmsd(model, ref, atom_indices=idx_m, ref_atom_indices=idx_r)[0]) if len(idx_m) >= 3 else np.nan
        frag_pairs = pairs[(pairs[:, 0] < k) & (pairs[:, 1] < k)] if total else pairs
        q_global, q_local = compute_q_value(model, frag_pairs, total) if len(frag_pairs) else (0.0, 0.0)
        mapped_ss = ss_seq[:k][ss_seq[:k] != "-"]
        dom = "C"
        for s in ("E", "H", "C"):
            if len(mapped_ss) and np.sum(mapped_ss == s) >= len(mapped_ss) * 0.5:
                dom = s
                break
        rows.append({"length": k, "rmsd_mean": rmsd, "q_mean": q_global, "q_local_mean": q_local,
                     "hydro_rg_mean": compute_hydrophobic_rg(model), "plddt_mean": plddt_mean,
                     "ss_dominant": dom, "packing_mean": 0.0, "hydro_burial_mean": 0.0})
    return pd.DataFrame(rows)


def mapped_importance(name):
    """Topology axis (v1.1.0 compute_importance_from_native) on sequence positions; unmapped = 0."""
    ref = native_traj(name)
    nat2seq = mapping(name)["nat2seq"]
    n_nat = ref.topology.n_residues
    dist, pairs = md.compute_contacts(ref, scheme="closest-heavy")
    cmap = np.zeros((n_nat, n_nat))
    for a, b in pairs[dist[0] <= CONTACT_CUTOFF_NM]:
        cmap[a, b] = cmap[b, a] = 1.0
    imp_nat = compute_importance(compute_contact_degree(cmap, 0.5).astype(float),
                                 compute_ss_persistence(ref),
                                 compute_betweenness(cmap, 0.5).astype(float))
    n = len(mapping(name)["seq"])
    imp = np.zeros(n)
    for j, i in nat2seq.items():
        imp[i] = imp_nat[j]
    return pd.DataFrame({"residue": np.arange(1, n + 1), "importance": imp})


def mapped_contact_degree(name):
    """B4 baseline: C-alpha degree at 8 A, |i-j| >= 3 in native order; unmapped = NaN."""
    ref = native_traj(name)
    nat2seq = mapping(name)["nat2seq"]
    ca = [a.index for a in ref.topology.atoms if a.name == "CA"]
    res_of = [ref.topology.atom(a).residue.index for a in ca]
    xyz = ref.xyz[0, ca] * 10.0
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    sep = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None])
    deg_nat = ((d <= 8.0) & (sep >= 3)).sum(axis=1).astype(float)
    n = len(mapping(name)["seq"])
    deg = np.full(n, np.nan)
    for k, r in enumerate(res_of):
        if r in nat2seq:
            deg[nat2seq[r]] = deg_nat[k]
    return deg
