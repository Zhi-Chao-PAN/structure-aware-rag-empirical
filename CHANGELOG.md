# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-28

### Added
- **Core Pivot**: Implemented "Structure-Aware RAG" pipeline using LlamaParse.
- **Comparison Engine**: Added `run_comparison.py` with async support and checkpointing.
- **Evaluation**: Added `auto_score.py` and `visualize.py` for automated benchmarking.
- **Data**: Included NVIDIA 2024 10-K dataset and golden question benchmark.
- **Documentation**: Comprehensive Technical Report in `report/`.
- **Hardware Optimization**: `hardware_config.py` for RTX 4060 + Core Ultra 9 tuning.

### Changed
- Refactored project structure to separate `src`, `data`, and `experiments`.
- Updated `README.md` with final experiment results (+37.5% accuracy lift).
- Optimized caching strategy for HuggingFace embeddings.

### Fixed
- Resolved Unicode encoding issues in CSV outputs.
- Fixed GPU memory fragmentation issues during long inference runs.

## [0.1.0] - 2026-01-27

### Initialized
- Initial project scaffolding.
- Basic PyPDF parsing capability.
