from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bci_ecog.data.competition_iii import load_dataset
from bci_ecog.data.transforms import (
    ChannelwiseZScore,
    FeaturewiseZScore,
    SpectrogramTransform,
    TransformPipeline,
)


@dataclass
class WindowedBatch:
    signals: np.ndarray
    labels: np.ndarray | None = None


def load_raw_ecog_data(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    return load_dataset(
        dataset_dir=data_config["dataset_dir"],
        train_file=data_config["train_file"],
        competition_test_file=data_config["competition_test_file"],
    )


def build_preprocessing_pipeline(config: dict[str, Any]) -> TransformPipeline:
    preprocessing_config = config.get("preprocessing", {})
    normalization_config = preprocessing_config.get("normalization", {})
    spectrogram_config = preprocessing_config.get("spectrogram", {})
    representation_name = config["representation"]["name"]

    transforms = []
    if normalization_config.get("enabled", True):
        transforms.append(ChannelwiseZScore())

    if representation_name == "spectrogram" or spectrogram_config.get("enabled", False):
        transforms.append(
            SpectrogramTransform(
                sample_rate_hz=config["data"]["sample_rate_hz"],
                nperseg=spectrogram_config.get("nperseg", 128),
                noverlap=spectrogram_config.get("noverlap", 96),
                nfft=spectrogram_config.get("nfft", 128),
                log_power=spectrogram_config.get("log_power", True),
            )
        )
        if normalization_config.get("enabled", True):
            transforms.append(FeaturewiseZScore())

    return TransformPipeline(transforms=transforms)


def apply_normalization(
    signals: np.ndarray,
    pipeline: TransformPipeline,
    fit: bool,
) -> np.ndarray:
    if fit:
        return pipeline.fit_transform(signals)
    return pipeline.transform(signals)


def apply_windowing(
    signals: np.ndarray,
    labels: np.ndarray | None,
    window_config: dict[str, Any],
) -> WindowedBatch:
    if not window_config.get("enabled", False):
        return WindowedBatch(signals=signals, labels=labels)

    window_size = int(window_config["size"])
    stride = int(window_config["stride"])
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive.")

    num_trials, num_channels, num_samples = signals.shape
    if window_size > num_samples:
        raise ValueError("Window size cannot exceed the number of time samples.")

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    for trial_index in range(num_trials):
        for start in range(0, num_samples - window_size + 1, stride):
            stop = start + window_size
            windows.append(signals[trial_index, :, start:stop])
            if labels is not None:
                window_labels.append(int(labels[trial_index]))

    stacked_windows = np.stack(windows).astype(np.float32)
    stacked_labels = np.asarray(window_labels, dtype=np.int64) if labels is not None else None
    return WindowedBatch(signals=stacked_windows, labels=stacked_labels)


def generate_spectrograms(
    signals: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    pipeline = build_preprocessing_pipeline(
        {
            "data": config["data"],
            "representation": {"name": "spectrogram"},
            "preprocessing": config.get("preprocessing", {}),
        }
    )
    return pipeline.fit_transform(signals)

