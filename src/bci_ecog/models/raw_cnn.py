from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class RawCNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
        classifier_hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = input_channels
        for output_channels in channels:
            blocks.append(
                ConvBlock1D(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            current_channels = output_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current_channels, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        return self.classifier(features)

