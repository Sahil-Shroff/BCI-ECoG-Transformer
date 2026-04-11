from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)



def _select_num_heads(embed_dim: int, preferred_heads: int) -> int:
    """Dynamically selects the number of attention heads to ensure divisibility."""
    for num_heads in range(min(preferred_heads, embed_dim), 0, -1):
        if embed_dim % num_heads == 0:
            return num_heads
    return 1



class SpectrogramCNNTransformer2D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
        classifier_hidden_dim: int,
        num_classes: int,
        transformer_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ff_multiplier: int = 4,
    ) -> None:
        super().__init__()

        # 1. Local Spatio-Temporal Feature Extractor (CNN Stem)
        blocks: list[nn.Module] = []
        current_channels = input_channels
        for output_channels in channels:
            blocks.append(
                ConvBlock2D(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            current_channels = output_channels

        self.feature_extractor = nn.Sequential(*blocks)

        # 2. Transformer Attention Mechanism
        embed_dim = current_channels
        num_heads = _select_num_heads(embed_dim, transformer_num_heads)
        feedforward_dim = embed_dim * transformer_ff_multiplier

        # Depthwise 2D Convolution for relative positional encoding
        self.positional_mixer = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim, 
            bias=False,
        )
        
        self.sequence_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True, # Enforcing norm_first for better gradient stability in deep networks
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )

        # 3. Global Pooling and Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Step 1: Extract local features using the CNN blocks
        # Output Shape: (Batch, Channels, Height, Width)
        features = self.feature_extractor(inputs)
        
        # Step 2: Inject relative positional information
        features = features + self.positional_mixer(features)
        
        # Step 3: Flatten spatial dimensions into tokens for batch_first=True Transformer input
        # Output Shape: (Batch, Height * Width, Channels)
        sequence = self.sequence_dropout(features).flatten(2).transpose(1, 2)
        
        # Step 4: Apply the Transformer Encoder for global temporal dependencies
        encoded_sequence = self.transformer(sequence)
        
        # Step 5: Global Mean Pooling across the temporal sequence
        # Output Shape: (Batch, Channels)
        pooled_state = encoded_sequence.mean(dim=1)
        
        # Step 6: Final classification mapping
        return self.classifier(pooled_state)