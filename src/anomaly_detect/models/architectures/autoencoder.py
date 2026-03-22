"""PyTorch Autoencoder architecture for anomaly detection.

Symmetric encoder-decoder with configurable depth, bottleneck, dropout,
batch norm, and activation.  Higher reconstruction error → more anomalous.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh":       nn.Tanh,
    "selu":       nn.SELU,
    "gelu":       nn.GELU,
}


def _make_block(
    in_dim: int,
    out_dim: int,
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
) -> list:
    layers: list = [nn.Linear(in_dim, out_dim)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(ACTIVATIONS[activation]())
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    return layers


class Autoencoder(nn.Module):
    """Symmetric autoencoder.

    Encoder: input → encoder_dims[0] → ... → bottleneck_dim
    Decoder: bottleneck_dim → ... → encoder_dims[0] → input (linear output)
    """

    is_neural: bool = True

    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = (64, 32, 16),
        bottleneck_dim: int = 8,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
        **kwargs,  # absorb optimizer / evaluation_set from Hydra config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.train_on_negatives_only: bool = False  # controlled via DataModule train_on_negatives

        # ── Encoder ───────────────────────────────────────────────────────
        enc_layers: list = []
        prev = input_dim
        for dim in encoder_dims:
            enc_layers.extend(_make_block(prev, dim, activation, dropout_rate, use_batch_norm))
            prev = dim
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror of encoder, linear output) ────────────────────
        dec_dims = list(reversed(encoder_dims))
        dec_layers: list = [nn.Linear(bottleneck_dim, dec_dims[0])]
        act_cls = ACTIVATIONS[activation]
        dec_layers.append(act_cls())
        prev = dec_dims[0]
        for dim in dec_dims[1:]:
            dec_layers.extend(_make_block(prev, dim, activation, dropout_rate, use_batch_norm))
            prev = dim
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
