# Data provenance

NucleoScan includes compact input tables needed to exercise and reproduce the
benchmark workflow. Generated structures and analysis outputs are not stored in
Git because of their size.

## Bundled files

| File | Purpose | Upstream source |
|---|---|---|
| `data/start2fold_data.json` | Default HDX benchmark consumed by the pipeline | Start2Fold |
| `data/start2fold_data.csv` | Tabular representation of the bundled Start2Fold records | Start2Fold |
| `data/two_state_folding_v3.xlsx` | Project-specific two-state folding table used by `--database v3` | Curated project input |
| `data/1L2Y.pdb` | Trp-cage reference structure used by the standalone example | RCSB PDB entry 1L2Y |
| `data/1L2Y.fasta` | Sequence extracted for the Trp-cage example | RCSB PDB entry 1L2Y |

The Start2Fold snapshot contains 209 records associated with 57 PDB identifiers.
The pipeline may download additional structures from the RCSB Protein Data Bank
and sequences from UniProt when they are not already cached.

## Integrity

`DATA_MANIFEST.tsv` records the byte count and SHA-256 digest of every bundled
input. Verify the snapshot with:

```bash
sha256sum data/1L2Y.fasta data/1L2Y.pdb \
  data/start2fold_data.csv data/start2fold_data.json \
  data/two_state_folding_v3.xlsx
```

The CPU-only repository tests also verify these digests.

## Citation and terms

When using the HDX annotations, cite:

> Pancsa R, Varadi M, Tompa P, Vranken WF. Start2Fold: a database of
> hydrogen/deuterium exchange data on protein folding and stability. *Nucleic
> Acids Research*. 2016;44(D1):D429–D434. https://doi.org/10.1093/nar/gkv1185

When using a PDB structure, cite the corresponding PDB entry and its primary
structure publication. UniProt records should likewise be cited according to
UniProt guidance.

The repository's MIT license applies to NucleoScan source code. It does not
replace the attribution requirements or usage terms of Start2Fold, RCSB PDB,
UniProt or other third-party sources. Users are responsible for confirming the
terms that apply to redistributed data in their own archives.
