"""Flank control: as predict_docker.py but without glycine flanks.

Proteins listed in FLANK0_PROTEINS (comma-separated); output in structures_flank0/,
timing in logs/flank0_timing.tsv. Rejection still uses mean pLDDT < 40, now computed
on the unflanked prediction.
"""
import os
import sys
import time
import types
from pathlib import Path

sys.modules.setdefault("mdtraj", types.ModuleType("mdtraj"))  # only used when n_seeds > 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_esmfold  # noqa: E402

root = Path(__file__).resolve().parent
timing = root / "logs" / "flank0_timing.tsv"
done = set()
if timing.exists():
    done = {tuple(l.split("\t")[:2]) for l in timing.read_text().splitlines()[1:]}
else:
    timing.write_text("protein\tlength\tseconds\tstatus\n")

model = run_esmfold.get_model()
proteins = [root / "fragments" / n for n in os.environ["FLANK0_PROTEINS"].split(",")]
for prot in proteins:
    for fasta in sorted(prot.glob("fragment_*.fasta"), key=lambda p: int(p.stem.split("_")[1])):
        length = fasta.stem.split("_")[1]
        out_dir = root / "structures_flank0" / prot.name / f"frag{length}"
        if (prot.name, length) in done or (out_dir / "decoy_00000.pdb").exists():
            continue
        t0 = time.time()
        try:
            out = run_esmfold.fold_fragment(str(fasta), str(out_dir), n_seeds=1, base_seed=42,
                                            flank_glycines=0, score_output=str(out_dir / "scores.tsv"))
            status = "ok" if out else "rejected"
        except RuntimeError as e:  # CUDA OOM: retry with chunked attention
            import torch
            torch.cuda.empty_cache()
            model.set_chunk_size(64)
            out = run_esmfold.fold_fragment(str(fasta), str(out_dir), n_seeds=1, base_seed=42,
                                            flank_glycines=0, score_output=str(out_dir / "scores.tsv"))
            model.set_chunk_size(None)
            status = ("ok" if out else "rejected") + "_chunk64"
        with timing.open("a") as fh:
            fh.write(f"{prot.name}\t{length}\t{time.time() - t0:.3f}\t{status}\n")
