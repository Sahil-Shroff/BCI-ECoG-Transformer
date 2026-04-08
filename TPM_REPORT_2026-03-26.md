# TPM Status Report: ECoG Deep Learning Baseline

Date: 2026-03-26
Project: BCI Competition III Dataset I ECoG decoding
Owner roles:
- Sahil: Client PM
- ChatGPT: TPM
- Codex: TL/dev

## Executive Summary

Phase 1 baseline infrastructure is in place and runnable. The repository now supports config-driven experiments, deterministic splitting, modular preprocessing, a raw-signal 1D CNN baseline, artifact logging, checkpointing, and automatic output generation for plots and metrics.

The first complete raw baseline run succeeded locally after dataset download issues were resolved. Current raw baseline performance on the deterministic in-session split is:

- Validation balanced accuracy: 0.881
- Test balanced accuracy: 0.833
- Best validation epoch: 10

This is a solid Phase 1 starting point. The codebase is in a usable state for controlled baseline comparisons, but the spectrogram path still needs to be run and compared under the same evaluation setup before we treat the modeling story as complete.

## Completed

- Created clean project structure under `src/`, `configs/`, `scripts/`, `outputs/`, and `notebooks/`
- Added config-driven training entrypoint: `python -m src.main --config configs/raw_1dcnn.yaml`
- Added dataset download helper and improved missing-data error handling
- Implemented deterministic seeding
- Implemented YAML config loading
- Implemented training logger and checkpoint saver
- Implemented preprocessing skeleton for:
  - raw ECoG loading
  - normalization
  - optional windowing
  - spectrogram generation
- Implemented dataset/dataloader skeletons for raw and spectrogram modes
- Implemented modest 1D CNN baseline
- Implemented generic training/evaluation loop
- Implemented automatic outputs:
  - `history.csv`
  - `training_curves.png`
  - `metrics.csv`
  - `metrics.json`
  - `summary.json`
  - confusion matrices
  - checkpoints

## Current Run Status

Latest complete run:
- Output directory: `outputs/raw_1dcnn/20260326_203359`
- Device used: CPU
- Input representation: raw
- Input shape: `[64, 3000]`

Split summary:
- Train: 194 trials, perfectly balanced
- Validation: 42 trials, perfectly balanced
- Test: 42 trials, perfectly balanced

Metrics from latest run:
- Validation accuracy: 0.881
- Validation balanced accuracy: 0.881
- Validation macro-F1: 0.880
- Test accuracy: 0.833
- Test balanced accuracy: 0.833
- Test macro-F1: 0.831

## Risks / Gaps

- The completed run used CPU, not the target single-GPU path. Functionally this is fine for validation, but we still need one confirmed GPU run on the intended environment.
- The current quantitative evaluation is based on a deterministic split of the labeled session. This is appropriate for baseline development, but it is not yet a session-transfer comparison.
- Spectrogram baseline has been scaffolded but not yet executed and benchmarked against the raw baseline.
- No report-ready interpretation exists yet for what channels, time regions, or preprocessing choices matter most.
- Output directories are now tracked in git except for checkpoint files. That is useful for report artifacts, but it will increase repo noise unless commits are curated.

## Recommended Next Steps

1. Run the spectrogram baseline with matched split settings and produce the same artifact bundle.
2. Confirm one training run on the intended GCP L4 setup and record runtime, memory use, and any batch-size adjustments.
3. Add a concise experiment tracking table that compares raw vs spectrogram on validation and test splits.
4. Freeze Phase 1 preprocessing choices once the first comparison is complete.
5. Defer any attention or more complex modeling until the baseline comparison is stable and documented.

## Suggested TPM Message

The project has cleared initial infrastructure setup and completed its first end-to-end raw baseline run. The repository is now reproducible, config-driven, and generating report-friendly outputs automatically. The immediate next milestone is a matched spectrogram baseline run so we can make a scientifically clean raw-vs-time-frequency comparison before considering any more advanced architectures.
