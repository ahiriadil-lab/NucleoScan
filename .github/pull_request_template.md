## Summary

Describe the problem and the proposed change.

## Validation

- [ ] `ruff check --select E9,F63,F7,F82 .`
- [ ] `python -m compileall -q .`
- [ ] `python -m unittest discover -s tests -v`
- [ ] Documentation and changelog updated when required

## Scientific and compatibility impact

State whether the change affects scoring, thresholds, residue numbering,
datasets, output columns or biological interpretation. Include before/after
evidence for numerical changes.
