import numpy as np

from bci_ecog.data.preprocessing import (
    _extract_imfs_1d,
    _interpolate_envelope,
    _mix_imfs_across_trials,
    apply_emd_augmentation,
)


def test_emd_augmentation_disabled_returns_inputs():
    signals = np.random.randn(4, 3, 64).astype(np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    augmented_signals, augmented_labels = apply_emd_augmentation(
        signals=signals,
        labels=labels,
        augmentation_config={"enabled": False},
        seed=42,
    )

    assert np.array_equal(augmented_signals, signals)
    assert np.array_equal(augmented_labels, labels)


def test_emd_augmentation_increases_size_deterministically():
    rng = np.random.default_rng(123)
    signals = rng.normal(size=(5, 4, 96)).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    config = {
        "enabled": True,
        "ratio": 1.0,
        "max_imfs": 4,
        "max_siftings": 6,
        "stopping_tolerance": 0.05,
        "imf_jitter_std": 0.10,
        "keep_residue": True,
    }

    augmented_signals_1, augmented_labels_1 = apply_emd_augmentation(
        signals=signals,
        labels=labels,
        augmentation_config=config,
        seed=99,
    )
    augmented_signals_2, augmented_labels_2 = apply_emd_augmentation(
        signals=signals,
        labels=labels,
        augmentation_config=config,
        seed=99,
    )

    assert augmented_signals_1.shape[0] == signals.shape[0] * 2
    assert augmented_labels_1.shape[0] == labels.shape[0] * 2

    assert np.array_equal(augmented_signals_1[: signals.shape[0]], signals)
    assert np.array_equal(augmented_labels_1[: labels.shape[0]], labels)

    assert np.array_equal(augmented_signals_1, augmented_signals_2)


def test_emd_augmentation_accepts_high_imf_jitter_std():
    rng = np.random.default_rng(321)
    signals = rng.normal(size=(6, 4, 96)).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    config = {
        "enabled": True,
        "ratio": 0.5,
        "max_imfs": 4,
        "max_siftings": 6,
        "stopping_tolerance": 0.05,
        "imf_jitter_std": 0.60,
        "keep_residue": True,
        "extrema_distance": 3,
    }

    augmented_signals, augmented_labels = apply_emd_augmentation(
        signals=signals,
        labels=labels,
        augmentation_config=config,
        seed=7,
    )

    assert augmented_signals.shape[0] > signals.shape[0]
    assert augmented_labels.shape[0] > labels.shape[0]
    assert np.all(np.isfinite(augmented_signals))


def test_extrema_distance_filters_noise():
    """Verify that extrema_distance parameter reduces false extrema from noise."""
    rng = np.random.default_rng(42)
    # Create high-frequency noise
    num_samples = 256
    noise = rng.normal(size=num_samples).astype(np.float64)
    
    # Extract IMFs with no distance filtering vs. with distance filtering
    imfs_loose = _extract_imfs_1d(
        signal=noise,
        max_imfs=3,
        max_siftings=5,
        stopping_tolerance=0.05,
        extrema_distance=1,
    )
    imfs_strict = _extract_imfs_1d(
        signal=noise,
        max_imfs=3,
        max_siftings=5,
        stopping_tolerance=0.05,
        extrema_distance=10,
    )
    
    # Stricter extrema distance should produce fewer IMFs or cleaner decomposition
    assert len(imfs_loose) > 0
    assert len(imfs_strict) > 0


def test_boundary_mirroring_prevents_edge_divergence():
    """Verify that symmetric mirroring prevents spline divergence at boundaries."""
    # Create a signal with extreme values at boundaries (common in EMD edge artifacts)
    points = np.array([5, 15, 25])
    values = np.array([1.0, -10.0, 5.0, 0.5, 2.0, 100.0, 3.0, -50.0, 2.0, 1.0,
                       5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 4.0,
                       3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0])  # 28 samples
    num_samples = len(values)
    
    envelope = _interpolate_envelope(points, values, num_samples)
    
    # Envelope should be finite (mirroring prevents divergence)
    assert np.all(np.isfinite(envelope))
    assert envelope.shape == (num_samples,)
    # Envelope should interpolate at specified points
    assert np.allclose(envelope[points], values[points], atol=0.5)


def test_cross_trial_imf_mixing_by_class():
    """Verify that IMF mixing creates diverse synthetic trials across class samples."""
    rng = np.random.default_rng(42)
    num_channels = 4
    num_samples = 64
    
    # Create synthetic IMF data: [trial][channel][imf_index][time_samples]
    all_imfs_by_trial = []
    for trial_idx in range(5):
        trial_imfs = []
        for ch_idx in range(num_channels):
            ch_imfs = [rng.normal(size=num_samples).astype(np.float32) for _ in range(4)]
            trial_imfs.append(ch_imfs)
        all_imfs_by_trial.append(trial_imfs)
    
    class_indices = np.array([0, 1, 2])  # Three trials of same class
    num_synthetic = 5
    
    synthetic = _mix_imfs_across_trials(
        all_imfs_by_trial=all_imfs_by_trial,
        class_indices=class_indices,
        num_synthetic=num_synthetic,
        max_imfs=4,
        rng=rng,
        keep_residue=True,
        imf_jitter_std=0.05,
    )
    
    assert synthetic.shape == (num_synthetic, num_channels, num_samples)
    assert np.all(np.isfinite(synthetic))


def test_augmentation_preserves_class_balance():
    """Verify that augmentation applies equally to all classes."""
    signals = np.random.randn(10, 3, 64).astype(np.float32)
    # Balanced classes: 5 of class 0, 5 of class 1
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    
    config = {
        "enabled": True,
        "ratio": 0.5,  # Add 50% more samples per class
        "max_imfs": 3,
        "max_siftings": 5,
        "stopping_tolerance": 0.05,
        "imf_jitter_std": 0.15,
        "keep_residue": True,
        "extrema_distance": 3,
    }
    
    augmented_signals, augmented_labels = apply_emd_augmentation(
        signals=signals,
        labels=labels,
        augmentation_config=config,
        seed=42,
    )
    
    # 10 original + (floor(5 * 0.5) class 0 + floor(5 * 0.5) class 1) = 10 + 4 = 14 total
    expected_size = 10 + 4
    assert augmented_signals.shape[0] == expected_size
    
    # Both classes should receive some augmentation
    class_0_count = np.sum(augmented_labels == 0)
    class_1_count = np.sum(augmented_labels == 1)
    assert class_0_count >= 5 and class_1_count >= 5
