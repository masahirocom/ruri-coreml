"""The pooling head that turns ModernBERT's per-token output into one
sentence vector, mirroring ruri-v3's own mean-pooling + L2-normalize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import L2_NORMALIZATION_DIMENSION, POOLING_DIVISION_EPSILON


class MeanPoolingSentenceEmbedder(nn.Module):
    """Wraps a ModernBERT encoder with attention-mask-weighted mean pooling
    followed by L2 normalization, so the traced/exported graph outputs a
    single ready-to-compare embedding vector directly — no pooling logic
    needs to be reimplemented by whatever runtime loads the exported model.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        real_token_counts = mask.sum(dim=1).clamp(min=POOLING_DIVISION_EPSILON)
        mean_pooled = summed / real_token_counts

        return F.normalize(mean_pooled, p=2, dim=L2_NORMALIZATION_DIMENSION)
