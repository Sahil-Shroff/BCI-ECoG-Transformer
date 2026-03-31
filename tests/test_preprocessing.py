import numpy as np

from bci_ecog.data.preprocessing import apply_emd_augmentation


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
    assert np.array_equal(augmented_labels_1, augmented_labels_2)
