"""
e2p_projector.py
================
Embedding-to-Prefix (E2P) Projection Module.

Projects a compact user embedding (from FT-Transformer) into one or more
soft prefix tokens in the LLM's hidden space. The LLM processes these
prefix tokens as if they were regular input tokens, allowing user context
to condition generation without modifying LLM weights.

Reference: Validated in production at Spotify-scale (+12.9% engagement),
adding <2ms latency with ~100K parameters.

Architecture:
    user_embedding (128-dim)
    → MLP (128 → 512 → 1152)
    → Reshape to (n_prefix_tokens, llm_hidden_dim)
    → Concatenated before input embeddings in the LLM forward pass
"""

import torch
import torch.nn as nn


class E2PProjector(nn.Module):
    """
    Project user embedding into soft prefix tokens for LLM conditioning.

    The projected tokens are prepended to the LLM's input embedding sequence.
    During training, only this projector (and optionally LoRA weights) are
    updated — the base LLM stays frozen.

    Args:
        user_embed_dim: Dimension of user embedding (FT-Transformer output).
        llm_hidden_dim: Hidden dimension of the target LLM (1152 for Gemma3-1B).
        n_prefix_tokens: Number of soft prefix tokens to generate.
        projection_layers: Number of MLP layers (2 recommended).
        dropout: Dropout rate in the projection MLP.
    """

    def __init__(
        self,
        user_embed_dim: int = 128,
        llm_hidden_dim: int = 1152,
        n_prefix_tokens: int = 1,
        projection_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_embed_dim = user_embed_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.n_prefix_tokens = n_prefix_tokens

        output_dim = llm_hidden_dim * n_prefix_tokens

        # Build MLP projection
        layers = []
        in_dim = user_embed_dim
        hidden_dim = (user_embed_dim + output_dim) // 2  # Interpolated hidden

        for i in range(projection_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection to LLM space
        layers.append(nn.Linear(in_dim, output_dim))

        self.projection = nn.Sequential(*layers)

        # Learnable scaling factor (PEPNet-inspired)
        # δ = 2 * sigmoid(BN(projected)) — gates the injection strength
        self.gate_norm = nn.LayerNorm(output_dim)
        self.use_gating = True

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights to avoid disrupting LLM at start."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project user embedding into soft prefix tokens.

        Args:
            user_embedding: (batch_size, user_embed_dim)

        Returns:
            prefix_tokens: (batch_size, n_prefix_tokens, llm_hidden_dim)
                Ready to be prepended to LLM input embeddings.
        """
        B = user_embedding.size(0)

        # Project: (B, user_embed_dim) → (B, n_prefix * llm_hidden)
        projected = self.projection(user_embedding)

        # Apply PEPNet-style gating: δ = 2 * sigmoid(LN(x))
        if self.use_gating:
            gate = 2.0 * torch.sigmoid(self.gate_norm(projected))
            projected = projected * gate

        # Reshape to prefix token sequence
        prefix_tokens = projected.view(B, self.n_prefix_tokens, self.llm_hidden_dim)

        return prefix_tokens

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
