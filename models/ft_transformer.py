"""
ft_transformer.py
=================
Feature Tokenizer + Transformer (FT-Transformer) for encoding tabular user
profile features into a dense embedding vector.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular
Data" (NeurIPS 2021).

The FT-Transformer treats each input feature as a "token":
    1. Each feature gets its own learned embedding via a linear projection.
    2. A [CLS] token is prepended.
    3. A standard Transformer encoder processes the sequence.
    4. The [CLS] output is the final user embedding.

This handles the heterogeneous nature of Chase's user features natively:
binary product indicators, counts, and derived ratios are all projected into
the same embedding space before attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Project each scalar feature into a d_model-dimensional token embedding.

    For feature i with value x_i:
        token_i = x_i * W_i + b_i

    where W_i ∈ R^{d_model} and b_i ∈ R^{d_model} are learnable per-feature.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-feature linear projection: (num_features, d_model)
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(0)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) - raw feature values.

        Returns:
            (batch_size, num_features, d_model) - tokenized features.
        """
        # x: (B, F) → (B, F, 1) * (F, D) → (B, F, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm architecture."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout1(x) + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer: Feature Tokenizer + Transformer for tabular data.

    Architecture:
        1. FeatureTokenizer: Each feature → d_model embedding
        2. Prepend learnable [CLS] token
        3. Add positional embeddings (optional, learned)
        4. N Transformer encoder blocks
        5. [CLS] output → Linear → output_dim user embedding

    Args:
        num_features: Number of input features (55 for our user profile).
        d_model: Internal Transformer dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ffn_factor: FFN hidden dim = d_model * factor.
        dropout: General dropout rate.
        attention_dropout: Attention-specific dropout.
        ffn_dropout: FFN-specific dropout.
        output_dim: Final user embedding dimension.
    """

    def __init__(
        self,
        num_features: int = 55,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn_factor: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.output_dim = output_dim

        # Feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(num_features, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings (num_features + 1 for CLS)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_features + 1, d_model) * 0.02
        )

        # Transformer blocks
        d_ffn = int(d_model * d_ffn_factor)
        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm + projection
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) - raw feature vector.

        Returns:
            (batch_size, output_dim) - user embedding.
        """
        B = x.size(0)

        # Tokenize features: (B, F, D)
        tokens = self.feature_tokenizer(x)

        # Prepend CLS: (B, F+1, D)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Extract CLS representation
        cls_output = self.final_norm(tokens[:, 0, :])

        # Project to output dimension
        user_embedding = self.head(cls_output)

        return user_embedding

    def get_attention_weights(self, x: torch.Tensor):
        """
        Extract attention weights for interpretability.
        Shows which user features the model attends to most.

        Returns:
            List of (batch, heads, seq, seq) attention tensors per layer.
        """
        B = x.size(0)
        tokens = self.feature_tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        attention_weights = []
        for block in self.blocks:
            normed = block.norm1(tokens)
            _, weights = block.attn(normed, normed, normed, need_weights=True)
            attention_weights.append(weights)
            tokens = block(tokens)

        return attention_weights
