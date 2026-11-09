"""Same predictions as predict_docker.py for prefixes whose full fp16 model does not fit in 8 GB.

ESM-2 representations are computed in batches with only the language model on the GPU,
then the language model is moved to the CPU and the folding trunk runs on the GPU with the
cached representations. Every operation runs on the same device and precision as in
predict_docker.py, so outputs are unchanged; only weight placement differs.
Logged seconds = language-model time + trunk time for that prefix (weight swaps excluded).
"""
import os
import sys
import time
import types
from pathlib import Path

sys.modules.setdefault("mdtraj", types.ModuleType("mdtraj"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import esm  # noqa: E402
from esm.esmfold.v1.misc import batch_encode_sequences  # noqa: E402
import run_esmfold  # noqa: E402

BATCH = 25
root = Path(__file__).resolve().parent
timing = root / "logs" / "esmfold_timing.tsv"
done = {tuple(l.split("\t")[:2]) for l in timing.read_text().splitlines()[1:]}

model = esm.pretrained.esmfold_v1().eval().half()
run_esmfold._MODEL = model
lm_forward = model._compute_language_model_representations
cache = {}


def esm_tokens(seq):
    aatype, mask, *_ = batch_encode_sequences([seq], 512, "G" * 25)
    return model._af2_idx_to_esm_idx(aatype, mask)


def cached_lm(esmaa):
    s = cache[tuple(esmaa[0].tolist())]
    return s.cuda(), None


todo = []
for prot in sorted((root / "fragments").iterdir(), key=lambda d: len(list(d.glob("*.fasta")))):
    for fasta in sorted(prot.glob("fragment_*.fasta"), key=lambda p: int(p.stem.split("_")[1])):
        length = fasta.stem.split("_")[1]
        out_dir = root / "structures_rerun" / prot.name / f"frag{length}"
        if (prot.name, length) not in done and not (out_dir / "decoy_00000.pdb").exists():
            todo.append((prot.name, length, fasta, out_dir))

if os.environ.get("OFFLOAD_TEST"):  # re-predict existing prefixes into logs/offload_test to check identity
    todo = [(n, l, root / "fragments" / n / f"fragment_{l}.fasta", root / "logs" / "offload_test" / f"{n}_frag{l}")
            for n, l in (x.split(":") for x in os.environ["OFFLOAD_TEST"].split(","))]
    timing = root / "logs" / "offload_test" / "timing.tsv"
    timing.parent.mkdir(exist_ok=True)

for start in range(0, len(todo), BATCH):
    batch = todo[start:start + BATCH]
    # phase 1: language model only
    model.cpu()
    torch.cuda.empty_cache()
    model.esm.cuda()
    lm_time = {}
    model._compute_language_model_representations = lm_forward
    with torch.no_grad():
        for name, length, fasta, _ in batch:
            seq = run_esmfold._add_flanking_glycines(run_esmfold._read_fasta(str(fasta))[1], 15)
            toks = esm_tokens(seq)
            t0 = time.time()
            esm_s, esm_z = lm_forward(toks.cuda())
            assert esm_z is None
            torch.cuda.synchronize()
            lm_time[(name, length)] = time.time() - t0
            cache[tuple(toks[0].tolist())] = esm_s.cpu()
    # phase 2: trunk and heads with cached representations
    model.esm.cpu()
    torch.cuda.empty_cache()
    for child_name, child in model.named_children():
        if child_name != "esm":
            child.cuda()
    model.esm_s_combine.data = model.esm_s_combine.data.cuda()
    for buf_name, buf in list(model.named_buffers(recurse=False)):
        setattr(model, buf_name, buf.cuda())
    model._compute_language_model_representations = cached_lm
    for name, length, fasta, out_dir in batch:
        t0 = time.time()
        out = run_esmfold.fold_fragment(str(fasta), str(out_dir), n_seeds=1, base_seed=42,
                                        flank_glycines=15, score_output=str(out_dir / "scores.tsv"))
        with timing.open("a") as fh:
            fh.write(f"{name}\t{length}\t{time.time() - t0 + lm_time[(name, length)]:.3f}\t"
                     f"{'ok' if out else 'rejected'}_offload\n")
    cache.clear()
