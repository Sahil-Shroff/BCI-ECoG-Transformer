from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

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


def _interpolate_envelope(points: np.ndarray, values: np.ndarray, num_samples: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(num_samples, dtype=np.float64)

    x = points.astype(np.float64)
    y = values[points].astype(np.float64)

    if x[0] != 0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([values[0]], y))
    if x[-1] != num_samples - 1:
        x = np.concatenate((x, [float(num_samples - 1)]))
        y = np.concatenate((y, [values[-1]]))

    xi = np.arange(num_samples, dtype=np.float64)
    if x.size >= 4:
        spline = CubicSpline(x, y, bc_type="natural")
        return spline(xi)
    return np.interp(xi, x, y)


def _extract_imfs_1d(
    signal: np.ndarray,
    max_imfs: int,
    max_siftings: int,
    stopping_tolerance: float,
) -> list[np.ndarray]:
    residue = signal.astype(np.float64, copy=True)
    imfs: list[np.ndarray] = []

    for _ in range(max_imfs):
        maxima, _ = find_peaks(residue)
        minima, _ = find_peaks(-residue)
        if maxima.size + minima.size < 3:
            break

        candidate = residue.copy()
        for _ in range(max_siftings):
            maxima, _ = find_peaks(candidate)
            minima, _ = find_peaks(-candidate)
            if maxima.size < 2 or minima.size < 2:
                break

            upper = _interpolate_envelope(maxima, candidate, candidate.size)
            lower = _interpolate_envelope(minima, candidate, candidate.size)
            mean_envelope = 0.5 * (upper + lower)

            previous = candidate
            candidate = candidate - mean_envelope

            denominator = np.linalg.norm(previous) + 1e-12
            sift_delta = np.linalg.norm(previous - candidate) / denominator
            if sift_delta < stopping_tolerance:
                break

        if np.allclose(candidate, 0.0):
            break

        imfs.append(candidate.astype(np.float32))
        residue = residue - candidate

    imfs.append(residue.astype(np.float32))
    return imfs


def _augment_trial_with_emd(
    trial_signals: np.ndarray,
    rng: np.random.Generator,
    max_imfs: int,
    max_siftings: int,
    stopping_tolerance: float,
    imf_jitter_std: float,
    keep_residue: bool,
) -> np.ndarray:
    augmented = np.empty_like(trial_signals, dtype=np.float32)

    for channel_index in range(trial_signals.shape[0]):
        imfs = _extract_imfs_1d(
            signal=trial_signals[channel_index],
            max_imfs=max_imfs,
            max_siftings=max_siftings,
            stopping_tolerance=stopping_tolerance,
        )

        if len(imfs) == 1:
            scale = float(rng.normal(1.0, imf_jitter_std))
            augmented[channel_index] = (imfs[0] * scale).astype(np.float32)
            continue

        scales = rng.normal(loc=1.0, scale=imf_jitter_std, size=len(imfs)).astype(np.float32)
        if keep_residue:
            scales[-1] = 1.0

        reconstructed = np.zeros_like(imfs[0], dtype=np.float32)
        for scale, component in zip(scales, imfs):
            reconstructed += scale * component
        augmented[channel_index] = reconstructed

    return augmented


def apply_emd_augmentation(
    signals: np.ndarray,
    labels: np.ndarray,
    augmentation_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not augmentation_config.get("enabled", False):
        return signals, labels

    ratio = float(augmentation_config.get("ratio", 1.0))
    if ratio < 0:
        raise ValueError("EMD augmentation ratio must be non-negative.")

    num_trials = signals.shape[0]
    num_augmented = int(np.floor(num_trials * ratio))
    if num_augmented == 0:
        return signals, labels

    max_imfs = int(augmentation_config.get("max_imfs", 6))
    max_siftings = int(augmentation_config.get("max_siftings", 10))
    stopping_tolerance = float(augmentation_config.get("stopping_tolerance", 0.05))
    imf_jitter_std = float(augmentation_config.get("imf_jitter_std", 0.10))
    keep_residue = bool(augmentation_config.get("keep_residue", True))
    if max_imfs <= 0 or max_siftings <= 0:
        raise ValueError("EMD augmentation max_imfs and max_siftings must be positive.")
    if imf_jitter_std < 0:
        raise ValueError("EMD augmentation imf_jitter_std must be non-negative.")
    if stopping_tolerance <= 0:
        raise ValueError("EMD augmentation stopping_tolerance must be positive.")

    rng = np.random.default_rng(seed)
    source_indices = rng.choice(num_trials, size=num_augmented, replace=True)

    synthetic = np.empty((num_augmented, signals.shape[1], signals.shape[2]), dtype=np.float32)
    synthetic_labels = labels[source_indices].astype(np.int64, copy=True)

    for augmented_index, source_index in enumerate(source_indices):
        synthetic[augmented_index] = _augment_trial_with_emd(
            trial_signals=signals[source_index],
            rng=rng,
            max_imfs=max_imfs,
            max_siftings=max_siftings,
            stopping_tolerance=stopping_tolerance,
            imf_jitter_std=imf_jitter_std,
            keep_residue=keep_residue,
        )

    augmented_signals = np.concatenate((signals.astype(np.float32), synthetic), axis=0)
    augmented_labels = np.concatenate((labels.astype(np.int64), synthetic_labels), axis=0)
    return augmented_signals, augmented_labels


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

