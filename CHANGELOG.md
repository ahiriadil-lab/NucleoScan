# Changelog

All notable changes to NucleoScan are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Updated GitHub Actions to their current major versions.
- Configured conservative Python dependency updates to preserve broad
  compatibility.

## [1.1.0] - 2026-09-18

### Added

- ESMFold-based fragment prediction and pLDDT-aware scoring.
- Unified fragment-emergence score with marginal, intermediate-contact and
  native-topology axes.
- Start2Fold evaluation, baseline comparisons and automated sanity checks.
- Reproducibility guide, data checksums, community health files and CI.

### Changed

- Renamed the project from FoldNucleus to NucleoScan.
- Clarified that FES is exploratory and does not establish folding-nucleus or
  transition-state membership.
- Simplified code comments and removed obsolete OpenFold terminology.

[Unreleased]: https://github.com/ahiriadil-lab/NucleoScan/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ahiriadil-lab/NucleoScan/releases/tag/v1.1.0
