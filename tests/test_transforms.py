import numpy as np

from bci_ecog.data.transforms import build_transform_pipeline


def test_raw_pipeline_keeps_trial_channel_time_shape():
    config = {
        "data": {"sample_rate_hz": 1000},
        "representation": {"name": "raw", "normalize_raw": True},
    }
    inputs = np.random.randn(10, 64, 3000).astype(np.float32)
    pipeline = build_transform_pipeline(config)
    outputs = pipeline.fit_transform(inputs)
    assert outputs.shape == inputs.shape


def test_spectrogram_pipeline_returns_four_dimensional_tensor():
    config = {
        "data": {"sample_rate_hz": 1000},
        "representation": {
            "name": "spectrogram",
            "normalize_raw": True,
            "normalize_spectrogram": True,
            "spectrogram": {
                "nperseg": 128,
                "noverlap": 96,
                "nfft": 128,
                "log_power": True,
            },
        },
    }
    inputs = np.random.randn(10, 64, 3000).astype(np.float32)
    pipeline = build_transform_pipeline(config)
    outputs = pipeline.fit_transform(inputs)
    assert outputs.ndim == 4
    assert outputs.shape[:2] == (10, 64)

