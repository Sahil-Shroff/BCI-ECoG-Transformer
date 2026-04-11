from __future__ import annotations

from typing import Any

import torch.nn as nn

from bci_ecog.models.raw_cnn import RawCNN1D
from bci_ecog.models.raw_cnn_transformer import RawCNNTransformer1D
from bci_ecog.models.spectrogram_cnn import SpectrogramCNN2D
from bci_ecog.models.spectrogram_cnn_transformer import SpectrogramCNNTransformer2D

def build_model(model_config: dict[str, Any], input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    model_name = model_config["name"]
    if model_name == "raw_cnn_1d":
        return RawCNN1D(
            input_channels=input_shape[0],
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            classifier_hidden_dim=model_config["classifier_hidden_dim"],
            num_classes=num_classes,
        )
    if model_name == "raw_cnn_transformer_1d":
        return RawCNNTransformer1D(
            input_channels=input_shape[0],
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            classifier_hidden_dim=model_config["classifier_hidden_dim"],
            num_classes=num_classes,
            transformer_layers=model_config.get("transformer_layers", 2),
            transformer_num_heads=model_config.get("transformer_num_heads", 4),
            transformer_ff_multiplier=model_config.get("transformer_ff_multiplier", 4),
        )
    if model_name == "spectrogram_cnn_2d":
        return SpectrogramCNN2D(
            input_channels=input_shape[0],
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            classifier_hidden_dim=model_config["classifier_hidden_dim"],
            num_classes=num_classes,
        )
    if model_name == "spectrogram_cnn_transformer_2d":
        return SpectrogramCNNTransformer2D(
            input_channels=input_shape[0],
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            classifier_hidden_dim=model_config["classifier_hidden_dim"],
            num_classes=num_classes,
            transformer_layers=model_config.get("transformer_layers", 2),
            transformer_num_heads=model_config.get("transformer_num_heads", 4),
            transformer_ff_multiplier=model_config.get("transformer_ff_multiplier", 4),
        )
    raise ValueError(f"Unsupported model name: {model_config['name']}")

