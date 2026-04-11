import torch

from bci_ecog.models.raw_cnn import RawCNN1D
from bci_ecog.models.raw_cnn_transformer import RawCNNTransformer1D
from bci_ecog.models.spectrogram_cnn import SpectrogramCNN2D
from bci_ecog.models.spectrogram_cnn_transformer import SpectrogramCNNTransformer2D


def test_raw_cnn_forward_shape():
    model = RawCNN1D(
        input_channels=64,
        channels=[32, 64],
        kernel_size=7,
        dropout=0.2,
        classifier_hidden_dim=32,
        num_classes=2,
    )
    inputs = torch.randn(4, 64, 3000)
    outputs = model(inputs)
    assert outputs.shape == (4, 2)


def test_spectrogram_cnn_forward_shape():
    model = SpectrogramCNN2D(
        input_channels=64,
        channels=[32, 64],
        kernel_size=3,
        dropout=0.2,
        classifier_hidden_dim=32,
        num_classes=2,
    )
    inputs = torch.randn(4, 64, 65, 90)
    outputs = model(inputs)
    assert outputs.shape == (4, 2)


def test_raw_cnn_transformer_forward_shape():
    model = RawCNNTransformer1D(
        input_channels=64,
        channels=[32, 64],
        kernel_size=7,
        dropout=0.2,
        classifier_hidden_dim=32,
        num_classes=2,
        transformer_layers=2,
        transformer_num_heads=4,
        transformer_ff_multiplier=4,
    )
    inputs = torch.randn(4, 64, 3000)
    outputs = model(inputs)
    assert outputs.shape == (4, 2)


def test_spectrogram_cnn_transformer_forward_shape():
    model = SpectrogramCNNTransformer2D(
        input_channels=64,
        channels=[32, 64],
        kernel_size=3,
        dropout=0.2,
        classifier_hidden_dim=32,
        num_classes=2,
        transformer_layers=2,
        transformer_num_heads=4,
        transformer_ff_multiplier=4,
    )
    inputs = torch.randn(4, 64, 65, 90)
    outputs = model(inputs)
    assert outputs.shape == (4, 2)

