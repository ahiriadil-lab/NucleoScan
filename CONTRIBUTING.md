# Contributing to NucleoScan

Contributions that improve reproducibility, numerical robustness, documentation
or scientific validation are welcome.

## Before opening an issue

- Search existing issues and confirm the problem on the latest release.
- For a runtime problem, include the operating system, Python version, GPU,
  CUDA version, command, full traceback and a minimal input example.
- For a scientific concern, distinguish a software defect from a limitation of
  the experimental reference or interpretation.
- Report security-sensitive problems according to `SECURITY.md` rather than in
  a public issue.

## Development setup

```bash
git clone https://github.com/ahiriadil-lab/NucleoScan.git
cd NucleoScan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
```

The CI suite is CPU-only and does not install ESMFold:

```bash
ruff check --select E9,F63,F7,F82 .
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Pull requests

1. Create a focused branch from `main`.
2. Keep changes small and avoid committing generated structures or results.
3. Add or update tests when behavior changes.
4. Update the README, reproducibility guide and changelog when relevant.
5. Preserve the interpretation of `nucleus_score` as FES and document any
   proposed change to public output columns.
6. Confirm that all CI checks pass.

By contributing, you agree that your contribution is distributed under the
repository's MIT License and that you will follow the Code of Conduct.
