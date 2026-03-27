# BCI Competition III Dataset I Baselines

This repository provides a clean, reproducible deep learning baseline for ECoG decoding on BCI Competition III Dataset I. The project keeps the architecture modest and centers the engineering around trial-level data hygiene, deterministic splits, and matched evaluation between raw time-series and spectrogram inputs.

The official dataset description states that Dataset I contains 278 labeled training trials from one session and 100 unlabeled competition trials from a second session recorded about one week later. Each trial is a 3 second ECoG segment sampled at 1000 Hz from an 8x8 grid, stored as `X[trials, channels, samples]`, with training labels `Y` in `{-1, 1}`. Source: [BCI Competition III Dataset I description](https://www.bbci.de/competition/iii/desc_I.html) and [download page](https://www.bbci.de/competition/iii/download/).

## What is included

- Package-style source tree under `src/`
- Config-driven experiments in `configs/`
- Deterministic train/val/test splitting at the trial level
- Train-only fitted preprocessing for raw and spectrogram representations
- Raw multichannel 1D CNN baseline
- Spectrogram 2D CNN baseline
- Automatic saving of history CSVs, summary JSONs, confusion matrices, predictions, and training curves

## Environment

Install the package into your environment:

```bash
pip install -e .
```

The code is designed for a single GPU machine and will fall back to CPU if CUDA is unavailable.

## Data setup

Option 1: place the official files manually under:

```text
data/raw/bci_competition_iii_dataset_i/
  Competition_train.mat.gz
  Competition_test.mat.gz
```

Option 2: try the helper downloader:

```bash
python -m bci_ecog.download_data --data-dir data/raw/bci_competition_iii_dataset_i
```

If the host changes the direct download URLs, download from the official competition page and place the two files in the directory above.

## Exact run commands

Train the raw baseline:

```bash
python -m src.main --config configs/raw_1dcnn.yaml
```

Train the spectrogram baseline:

```bash
python -m bci_ecog.train --config configs/spectrogram_cnn.yaml
```

Re-run evaluation for a saved checkpoint:

```bash
python -m bci_ecog.evaluate --checkpoint outputs/raw_cnn_baseline/<timestamp>/best_model.pt --split test
python -m bci_ecog.evaluate --checkpoint outputs/spectrogram_cnn_baseline/<timestamp>/best_model.pt --split test
python -m bci_ecog.evaluate --checkpoint outputs/raw_cnn_baseline/<timestamp>/best_model.pt --split competition
```

## Reproducibility and data hygiene

- Splits are created before any feature extraction and are deterministic from the experiment seed.
- Normalization is fit on the training subset only, then applied unchanged to validation, test, and competition-session data.
- The raw baseline uses channel-wise z-scoring over the training trials.
- The spectrogram baseline applies raw normalization first, then computes per-channel log-power spectrograms, and optionally normalizes those spectrogram features using training-only statistics.
- Saved artifacts include split indices and split summaries so report figures can be traced back to the exact partition used.

## Outputs

Each training run creates a timestamped directory under `outputs/<experiment_name>/...` with:

- `config.yaml`
- `history.csv`
- `training_curves.png`
- `metrics.csv`
- `summary.json`
- `split_indices.json`
- `split_summary.json`
- `best_model.pt`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png`
- `competition_test_predictions.csv`

## Notes on evaluation

The official second-session competition set is unlabeled on the public download page, so this repo uses a deterministic in-session train/val/test split over the labeled session for quantitative model comparison and exports predictions for the official competition session separately. That keeps the benchmarking honest while still supporting session-transfer inference artifacts.
