"""Compact longitudinal Transformer interface for CoRT experiments."""

from __future__ import annotations

import torch
from torch import nn


class LongitudinalTransformer(nn.Module):
    """Encode padded event-token sequences and return one binary logit."""

    def __init__(
        self,
        *,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        padding_index: int = 0,
    ) -> None:
        super().__init__()
        self.padding_index = int(padding_index)
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=padding_index)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = input_ids.eq(self.padding_index)
        encoded = self.encoder(self.embedding(input_ids), src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_probability(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(input_ids))

