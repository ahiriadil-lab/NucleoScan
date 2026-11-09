"""Run the v1.1.0 benchmark with one directory per PDB entry.

v1.1.0 keys result/structure directories by UniProt name; Myoglobin, Lysozyme_C
and Alpha_lactalbumin each occur for several PDB entries, so their directories
collide. Appending the PDB code keeps every entry separate. Nothing else in the scoring changes.
"""
import sys

import data.start2fold_adapter as adapter

_load = adapter.load_protein_list


def load_unique(*args, **kwargs):
    df = _load(*args, **kwargs)
    df["Name"] = df["Name"] + "_" + df["PDB"]
    return df


adapter.load_protein_list = load_unique

import run_dataset  # noqa: E402

# Predictions are made beforehand by predict_docker.py (ESMFold container);
# rejected prefixes (pLDDT < 40) have no structure and must not be re-folded here.
run_dataset.step3_run_folding = lambda *a, **k: run_dataset.log.info("  [3] using existing ESMFold predictions")

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--database", "start2fold", "--run-label", "rerun",
                "--skip-advanced", "--batched", *sys.argv[1:]]
    run_dataset.main()
