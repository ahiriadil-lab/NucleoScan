# Benchmark analysis scripts

These scripts perform the benchmark: residue mapping, scoring, statistics and figures.
They run on NucleoScan v1.1.0 (tag `v1.1.0`, commit `a93fa2b`), copied into the same folder.

Order: `rerun.py` (fragments), `run_predictions.sh` / `run_predictions_offload.sh` (ESMFold), `rerun.py` (fragment
metrics), `revision_analysis.py mapped|flank0`, `final_numbers.py`, `make_figures.py`.

Per-residue outputs, metrics and logs are in Online Resource 2 of the article.

## Evaluate your own residue-level score

`evaluate_table.py` needs only pandas, NumPy, SciPy and scikit-learn. Give it a CSV with `name,residue,score`:

```bash
python evaluate_table.py my_scores.csv --residues residue_scores_all_methods.csv.gz --out my_scores
```

It applies the same labels, metrics, pairing and Holm adjustment as the article. To verify the tool against the
published values: `python evaluate_table.py --check --residues residue_scores_all_methods.csv.gz --metrics metrics_long.csv`.
