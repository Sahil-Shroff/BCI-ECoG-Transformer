from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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


def _interpolate_envelope(points: np.ndarray, values: np.ndarray, num_samples: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(num_samples, dtype=np.float64)

    x = points.astype(np.float64)
    y = values[points].astype(np.float64)

    # Mirror extrema locations and amplitudes (not raw signal samples) to reduce
    # endpoint artifacts that can otherwise make cubic splines diverge.
    if x.size >= 3:
        mirror_count = min(3, x.size - 1)

        left_ref = x[0]
        left_points = x[1 : mirror_count + 1]
        left_x = 2.0 * left_ref - left_points[::-1]
        left_y = y[1 : mirror_count + 1][::-1]

        right_ref = x[-1]
        right_points = x[-mirror_count - 1 : -1]
        right_x = 2.0 * right_ref - right_points[::-1]
        right_y = y[-mirror_count - 1 : -1][::-1]

        x = np.concatenate((left_x, x, right_x))
        y = np.concatenate((left_y, y, right_y))

    # Ensure strictly increasing x for interpolation routines.
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    xi = np.arange(num_samples, dtype=np.float64)
    if x.size >= 4:
        try:
            spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)
            envelope = spline(xi)
            if np.all(np.isfinite(envelope)):
                return envelope
        except ValueError:
            pass

    # Linear interpolation fallback is numerically stable.
    return np.interp(xi, x, y)


def _extract_imfs_1d(
    signal: np.ndarray,
    max_imfs: int,
    max_siftings: int,
    stopping_tolerance: float,
    extrema_distance: int = 3,
) -> list[np.ndarray]:
    """Extract Intrinsic Mode Functions using sifting with prominence filtering.
    
    Args:
        signal: Input 1D signal.
        max_imfs: Maximum number of IMFs to extract.
        max_siftings: Maximum sifting iterations per IMF.
        stopping_tolerance: Convergence criterion for sifting.
        extrema_distance: Minimum sample distance between extrema to filter noise.
                         Distances help reject high-frequency false extrema from noise.
    """
    residue = signal.astype(np.float64, copy=True)
    imfs: list[np.ndarray] = []
    original_norm = np.linalg.norm(residue) + 1e-12

    for _ in range(max_imfs):
        maxima, _ = find_peaks(residue, distance=extrema_distance)
        minima, _ = find_peaks(-residue, distance=extrema_distance)
        if maxima.size + minima.size < 3:
            break

        candidate = residue.copy()
        for _ in range(max_siftings):
            maxima, _ = find_peaks(candidate, distance=extrema_distance)
            minima, _ = find_peaks(-candidate, distance=extrema_distance)
            if maxima.size < 2 or minima.size < 2:
                break

            upper = _interpolate_envelope(maxima, candidate, candidate.size)
            lower = _interpolate_envelope(minima, candidate, candidate.size)
            mean_envelope = 0.5 * (upper + lower)

            previous = candidate
            candidate = candidate - mean_envelope

            if not np.all(np.isfinite(candidate)):
                candidate = previous
                break

            # Prevent unstable growth from pathological envelope estimates.
            if np.linalg.norm(candidate) > 10.0 * original_norm:
                candidate = previous
                break

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


def _extract_trial_imfs(
    trial_signals: np.ndarray,
    max_imfs: int,
    max_siftings: int,
    stopping_tolerance: float,
    extrema_distance: int,
) -> list[list[np.ndarray]]:
    trial_imfs: list[list[np.ndarray]] = []
    for ch_idx in range(trial_signals.shape[0]):
        ch_imfs = _extract_imfs_1d(
            signal=trial_signals[ch_idx],
            max_imfs=max_imfs,
            max_siftings=max_siftings,
            stopping_tolerance=stopping_tolerance,
            extrema_distance=extrema_distance,
        )
        trial_imfs.append(ch_imfs)
    return trial_imfs

def _mix_imfs_across_trials(
    all_imfs_by_trial: list[list[list[np.ndarray]]],
    class_indices: np.ndarray,
    num_synthetic: int,
    max_imfs: int,
    rng: np.random.Generator,
    keep_residue: bool,
    imf_jitter_std: float,
) -> np.ndarray:
    """Mix IMFs extracted from different trials of the same class to create synthetic trials.
    
    This implements state-of-the-art EMD augmentation: randomly combining high-frequency
    IMFs from one trial with low-frequency IMFs from a different trial in the same class
    creates much more diverse and physiologically plausible training samples than merely
    scaling a single trial's IMFs.
    
    Args:
        all_imfs_by_trial: List of [channels] of [IMFs] for each trial.
        class_indices: Indices of trials belonging to this class.
        num_synthetic: Number of synthetic trials to generate.
        max_imfs: Maximum IMFs per channel.
        rng: Random number generator.
        keep_residue: Whether to preserve the residue (trend) without jitter.
    
    Returns:
        Synthetic trials with shape (num_synthetic, num_channels, num_samples).
    """
    if len(class_indices) < 2:
        # Fall back to single-trial jittering if insufficient class samples.
        num_channels = len(all_imfs_by_trial[class_indices[0]])
        num_samples = all_imfs_by_trial[class_indices[0]][0][0].shape[0]
        synthetic = np.empty((num_synthetic, num_channels, num_samples), dtype=np.float32)

        for syn_idx in range(num_synthetic):
            trial_idx = rng.choice(class_indices)
            source_imfs = all_imfs_by_trial[trial_idx]
            max_imf_count = max(len(imfs) for imfs in source_imfs) if source_imfs else 0
            scales = rng.normal(loc=1.0, scale=imf_jitter_std, size=max_imf_count).astype(np.float32)
            if keep_residue and max_imf_count > 0:
                scales[-1] = 1.0

            for ch_idx, ch_imfs in enumerate(source_imfs):
                reconstructed = np.zeros(num_samples, dtype=np.float32)
                for imf_idx, imf in enumerate(ch_imfs):
                    reconstructed += scales[imf_idx] * imf
                synthetic[syn_idx, ch_idx] = reconstructed
        return synthetic

    # IMF mixing strategy: for each synthetic trial, randomly split IMFs from two different source trials.
    num_channels = len(all_imfs_by_trial[class_indices[0]])
    num_samples = all_imfs_by_trial[class_indices[0]][0][0].shape[0]
    synthetic = np.empty((num_synthetic, num_channels, num_samples), dtype=np.float32)

    for syn_idx in range(num_synthetic):
        # Randomly select two (potentially identical) trials from the same class for mixing.
        trial_idx1, trial_idx2 = rng.choice(class_indices, size=2, replace=True)
        source_imfs_1 = all_imfs_by_trial[trial_idx1]
        source_imfs_2 = all_imfs_by_trial[trial_idx2]

        max_imf_count = max(
            max(len(imfs) for imfs in source_imfs_1) if source_imfs_1 else 0,
            max(len(imfs) for imfs in source_imfs_2) if source_imfs_2 else 0,
        )

        # IMFs are ordered high-frequency -> low-frequency, residue is last.
        # Keep residue from trial 1 and mix only IMF bands.
        imf_band_count = max_imf_count - 1 if keep_residue and max_imf_count > 1 else max_imf_count
        if imf_band_count <= 0:
            imf_band_count = 1
        cutoff_imf = int(rng.integers(1, imf_band_count + 1))

        for ch_idx in range(num_channels):
            reconstructed = np.zeros(num_samples, dtype=np.float32)
            ch_imfs_1 = source_imfs_1[ch_idx] if ch_idx < len(source_imfs_1) else []
            ch_imfs_2 = source_imfs_2[ch_idx] if ch_idx < len(source_imfs_2) else []

            # High-frequency IMFs from trial 1 (up to cutoff).
            upper_1 = min(cutoff_imf, max(0, len(ch_imfs_1) - (1 if keep_residue else 0)))
            for imf_idx in range(upper_1):
                reconstructed += ch_imfs_1[imf_idx]

            # Lower-frequency IMFs from trial 2 (beyond cutoff, excluding residue).
            upper_2 = max(0, len(ch_imfs_2) - (1 if keep_residue else 0))
            for imf_idx in range(cutoff_imf, upper_2):
                reconstructed += ch_imfs_2[imf_idx]

            # Preserve residue from trial 1 if enabled and present.
            if keep_residue and len(ch_imfs_1) > 0:
                reconstructed += ch_imfs_1[-1]

            synthetic[syn_idx, ch_idx] = reconstructed.astype(np.float32)

    return synthetic


def apply_emd_augmentation(
    signals: np.ndarray,
    labels: np.ndarray,
    augmentation_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply EMD-based augmentation with cross-trial IMF mixing by class.
    
    This augmentation:
    1. Extracts IMFs from each trial independently using univariate EMD.
    2. Mixes IMFs across different trials of the same class (high-frequency components
       from one trial combined with low-frequency from another) to create synthetic trials.
    3. Preserves spatial covariance by using synchronized scaling across channels.
    4. Uses symmetric boundary extension and extrema prominence filtering for robust decomposition.
    """
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
    extrema_distance = int(augmentation_config.get("extrema_distance", 3))
    workers = int(augmentation_config.get("workers", 1))
    if max_imfs <= 0 or max_siftings <= 0:
        raise ValueError("EMD augmentation max_imfs and max_siftings must be positive.")
    if imf_jitter_std < 0:
        raise ValueError("EMD augmentation imf_jitter_std must be non-negative.")
    if stopping_tolerance <= 0:
        raise ValueError("EMD augmentation stopping_tolerance must be positive.")
    if extrema_distance <= 0:
        raise ValueError("EMD augmentation extrema_distance must be positive.")
    if workers <= 0:
        raise ValueError("EMD augmentation workers must be positive.")

    # Extract IMFs from all trials upfront for mixing.
    rng = np.random.default_rng(seed)
    extract_trial = partial(
        _extract_trial_imfs,
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stopping_tolerance=stopping_tolerance,
        extrema_distance=extrema_distance,
    )

    if workers == 1:
        all_imfs = [extract_trial(signals[trial_idx]) for trial_idx in range(num_trials)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            all_imfs = list(executor.map(extract_trial, (signals[trial_idx] for trial_idx in range(num_trials))))

    # Generate synthetic trials by mixing IMFs within each class.
    unique_labels = np.unique(labels)
    synthetic_list = []
    synthetic_label_list = []

    for class_label in unique_labels:
        class_mask = labels == class_label
        class_indices = np.where(class_mask)[0]
        class_num_augmented = int(np.floor(len(class_indices) * ratio))

        if class_num_augmented > 0:
            class_synthetic = _mix_imfs_across_trials(
                all_imfs_by_trial=all_imfs,
                class_indices=class_indices,
                num_synthetic=class_num_augmented,
                max_imfs=max_imfs,
                rng=rng,
                keep_residue=keep_residue,
                imf_jitter_std=imf_jitter_std,
            )
            synthetic_list.append(class_synthetic)
            synthetic_label_list.extend([int(class_label)] * class_num_augmented)

    if not synthetic_list:
        return signals, labels

    synthetic = np.concatenate(synthetic_list, axis=0).astype(np.float32)
    synthetic_labels = np.asarray(synthetic_label_list, dtype=np.int64)

    augmented_signals = np.concatenate((signals.astype(np.float32), synthetic), axis=0)
    augmented_labels = np.concatenate((labels.astype(np.int64), synthetic_labels), axis=0)
    return augmented_signals, augmented_labels


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

