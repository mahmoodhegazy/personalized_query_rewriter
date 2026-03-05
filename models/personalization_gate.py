"""
personalization_gate.py
=======================
"When to Write" personalization gate following WeWrite (2025).

Not all queries benefit from personalization. Functional queries like
"check balance" or "pay credit card" have clear intent regardless of user
context. Only ambiguous queries like "rewards," "fees," or "travel" benefit
from user-conditioned rewriting.

This gate decides whether to route a query through the personalized rewriter
or pass it through unchanged, preserving latency for the majority of queries.

Architecture:
    Input:  user_embedding (128-dim) ⊕ query_embedding (128-dim)
    → MLP → sigmoid → binary decision (personalize or passthrough)

The gate is trained jointly with the rewriter via:
    1. Supervised labels from ambiguity detection (data_loader.identify_ambiguous_queries)
    2. End-to-end gradient from GRPO rewards (queries that benefit from
       rewriting should have gate_score > threshold)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalizationGate(nn.Module):
    """
    Binary gate that decides whether a query needs personalized rewriting.

    The gate takes:
        - User embedding from FT-Transformer
        - Query embedding (from the LLM's input embedding layer)

    And outputs a scalar probability indicating whether personalization
    would improve search results for this (user, query) pair.

    Args:
        user_embed_dim: Dimension of user embedding.
        query_embed_dim: Dimension of query embedding (can differ from user).
        hidden_dim: Hidden layer dimension.
        threshold: Decision threshold for personalization.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        user_embed_dim: int = 128,
        query_embed_dim: int = 128,
        hidden_dim: int = 64,
        threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.threshold = threshold

        input_dim = user_embed_dim + query_embed_dim

        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Query encoder: lightweight projection if query embeddings are
        # high-dimensional (e.g., from LLM hidden states)
        self.query_proj = None  # Set dynamically if needed

        self._init_weights()

    def _init_weights(self):
        for m in self.gate_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_query_projector(self, llm_hidden_dim: int, query_embed_dim: int):
        """Add a projector if query embeddings come from the LLM hidden space."""
        self.query_proj = nn.Linear(llm_hidden_dim, query_embed_dim)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(
        self,
        user_embedding: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute personalization gate score.

        Args:
            user_embedding:  (batch_size, user_embed_dim)
            query_embedding: (batch_size, query_embed_dim) or (B, llm_hidden_dim)

        Returns:
            gate_score: (batch_size, 1) — probability in [0, 1].
        """
        # Project query if needed
        if self.query_proj is not None:
            query_embedding = self.query_proj(query_embedding)

        # Concatenate user + query
        combined = torch.cat([user_embedding, query_embedding], dim=-1)

        # Compute gate score
        logit = self.gate_network(combined)
        score = torch.sigmoid(logit)

        return score

    def decide(
        self,
        user_embedding: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.BoolTensor:
        """
        Make binary personalization decision.

        Returns:
            should_personalize: (batch_size,) boolean tensor.
        """
        score = self.forward(user_embedding, query_embedding)
        return (score.squeeze(-1) > self.threshold)

    def compute_loss(
        self,
        user_embedding: torch.Tensor,
        query_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss for gate training.

        Args:
            labels: (batch_size,) — 1.0 if query is ambiguous, 0.0 otherwise.

        Returns:
            Scalar loss.
        """
        score = self.forward(user_embedding, query_embedding).squeeze(-1)
        return F.binary_cross_entropy(score, labels.float())
