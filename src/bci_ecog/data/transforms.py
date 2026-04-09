from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import spectrogram


class Transform:
    def fit(self, inputs: np.ndarray) -> "Transform":
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state


class ChannelwiseZScore(Transform):
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, inputs: np.ndarray) -> "ChannelwiseZScore":
        self.mean = inputs.mean(axis=(0, 2), keepdims=True)
        self.std = inputs.std(axis=(0, 2), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ChannelwiseZScore must be fit before transform.")
        return ((inputs - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)


class FeaturewiseZScore(Transform):
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, inputs: np.ndarray) -> "FeaturewiseZScore":
        # For spectrogram inputs shaped [trials, channels, freq, time], compute
        # statistics across the training batch so they broadcast to any split size.
        self.mean = inputs.mean(axis=0, keepdims=True)
        self.std = inputs.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeaturewiseZScore must be fit before transform.")
        return ((inputs - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)


class SpectrogramTransform(Transform):
    def __init__(
        self,
        sample_rate_hz: int,
        nperseg: int,
        noverlap: int,
        nfft: int | None,
        log_power: bool,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.log_power = log_power

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        num_trials, num_channels, num_samples = inputs.shape
        flattened = inputs.reshape(num_trials * num_channels, num_samples)
        _, _, spec = spectrogram(
            flattened,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            detrend=False,
            scaling="density",
            mode="psd",
            axis=-1,
        )
        if self.log_power:
            spec = np.log1p(spec)
        return spec.reshape(num_trials, num_channels, spec.shape[-2], spec.shape[-1]).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "nfft": self.nfft,
            "log_power": self.log_power,
        }


@dataclass
class TransformPipeline:
    transforms: list[Transform]

    def fit(self, inputs: np.ndarray) -> "TransformPipeline":
        current = inputs
        for transform in self.transforms:
            transform.fit(current)
            current = transform.transform(current)
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        current = inputs
        for transform in self.transforms:
            current = transform.transform(current)
        return current.astype(np.float32)

    def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
        self.fit(inputs)
        return self.transform(inputs)

    def state_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "name": transform.__class__.__name__,
                "state": transform.state_dict(),
            }
            for transform in self.transforms
        ]

    def load_state_dict(self, state: list[dict[str, Any]]) -> None:
        if len(state) != len(self.transforms):
            raise ValueError("Transform state is incompatible with configured pipeline.")
        for transform, transform_state in zip(self.transforms, state):
            transform.load_state_dict(transform_state["state"])


def build_transform_pipeline(config: dict[str, Any]) -> TransformPipeline:
    representation_config = config["representation"]
    transforms: list[Transform] = []

    if representation_config.get("normalize_raw", True):
        transforms.append(ChannelwiseZScore())

    if representation_config["name"] == "spectrogram":
        spectrogram_config = representation_config["spectrogram"]
        transforms.append(
            SpectrogramTransform(
                sample_rate_hz=config["data"]["sample_rate_hz"],
                nperseg=spectrogram_config["nperseg"],
                noverlap=spectrogram_config["noverlap"],
                nfft=spectrogram_config.get("nfft"),
                log_power=spectrogram_config.get("log_power", True),
            )
        )
        if representation_config.get("normalize_spectrogram", True):
            transforms.append(FeaturewiseZScore())

    return TransformPipeline(transforms=transforms)
