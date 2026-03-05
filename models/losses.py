"""
losses.py
=========
Loss functions for the personalized query rewriting system:

    1. InfoNCELoss       - Contrastive learning for user encoder
    2. RewriteRewardModel - Reward computation for GRPO alignment

References:
    - InfoNCE: CoPPS (KDD 2023), CLE-QR (CIKM 2022)
    - Reward: WeWrite (2025), GRAPE, CardRewriter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for training the FT-Transformer user encoder.

    Learns user embeddings such that (user, clicked_navlink) pairs are close
    while (user, other_navlinks) are distant. Uses in-batch negatives.

    L = -log( exp(sim(u, n+) / τ) / Σ_j exp(sim(u, n_j) / τ) )

    where u is the user embedding, n+ is the positive navlink embedding,
    and n_j iterates over all navlinks in the batch (in-batch negatives).

    Args:
        temperature: Softmax temperature τ (default 0.07).
        navlink_embed_dim: Dimension of navlink embeddings.
        num_navlinks: Size of navlink vocabulary.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        navlink_embed_dim: int = 128,
        num_navlinks: int = 100,
    ):
        super().__init__()
        self.temperature = temperature
        self.navlink_embeddings = nn.Embedding(num_navlinks, navlink_embed_dim)
        nn.init.normal_(self.navlink_embeddings.weight, std=0.02)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        navlink_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with in-batch negatives.

        Args:
            user_embeddings: (B, D) from FT-Transformer.
            navlink_ids: (B,) integer navlink IDs.

        Returns:
            Scalar loss.
        """
        # Get navlink embeddings
        navlink_embeds = self.navlink_embeddings(navlink_ids)  # (B, D)

        # L2 normalize
        user_norm = F.normalize(user_embeddings, p=2, dim=-1)
        navlink_norm = F.normalize(navlink_embeds, p=2, dim=-1)

        # Compute similarity matrix: (B, B)
        similarity = torch.matmul(user_norm, navlink_norm.T) / self.temperature

        # Labels: diagonal entries are positives
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(similarity, labels)

        return loss


class RewriteRewardModel(nn.Module):
    """
    Reward model for GRPO alignment of the query rewriter.

    Computes a composite reward for each candidate rewrite based on:

    1. Index Hit Rate: Does the rewritten query retrieve relevant results?
       - Computed by checking if gold navlink appears in top-K retrieval
         results for the rewritten query.

    2. Semantic Fidelity: Is the rewrite semantically close to the original?
       - Cosine similarity between original and rewritten query embeddings.
       - Prevents intent drift (e.g., "rewards" → completely unrelated topic).

    3. Length Penalty: Is the rewrite appropriately concise?
       - Penalizes overly long or overly short rewrites.

    R = w1 * hit_rate + w2 * semantic_sim + w3 * length_score

    Args:
        embed_model: A model that produces query embeddings (e.g., sentence-transformers).
        index_fn: Callable that takes a query and returns top-K navlinks.
        weights: Dict with keys 'index_hit', 'semantic_fidelity', 'length_penalty'.
    """

    def __init__(
        self,
        weights: dict = None,
    ):
        super().__init__()
        self.weights = weights or {
            "index_hit": 0.6,
            "semantic_fidelity": 0.3,
            "length_penalty": 0.1,
        }

    def compute_rewards(
        self,
        original_queries: list,
        rewritten_queries: list,
        gold_navlinks: list,
        original_embeddings: torch.Tensor,
        rewritten_embeddings: torch.Tensor,
        retrieved_navlinks: list = None,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of rewrites.

        Args:
            original_queries: List of original query strings.
            rewritten_queries: List of rewritten query strings.
            gold_navlinks: List of gold navlink strings.
            original_embeddings: (B, D) embeddings of original queries.
            rewritten_embeddings: (B, D) embeddings of rewritten queries.
            retrieved_navlinks: List of Lists — top-K retrieved navlinks per rewrite.
                               If None, index_hit is set to 0.

        Returns:
            rewards: (B,) reward scores.
        """
        B = len(original_queries)
        device = original_embeddings.device

        rewards = torch.zeros(B, device=device)

        # --- Reward 1: Index Hit Rate ---
        if retrieved_navlinks is not None:
            hit_rewards = torch.zeros(B, device=device)
            for i in range(B):
                if gold_navlinks[i] and gold_navlinks[i] in retrieved_navlinks[i]:
                    rank = retrieved_navlinks[i].index(gold_navlinks[i])
                    # Reciprocal rank reward
                    hit_rewards[i] = 1.0 / (rank + 1)
            rewards += self.weights["index_hit"] * hit_rewards

        # --- Reward 2: Semantic Fidelity ---
        orig_norm = F.normalize(original_embeddings, p=2, dim=-1)
        rewr_norm = F.normalize(rewritten_embeddings, p=2, dim=-1)
        semantic_sim = (orig_norm * rewr_norm).sum(dim=-1)  # (B,)
        rewards += self.weights["semantic_fidelity"] * semantic_sim

        # --- Reward 3: Length Penalty ---
        length_rewards = torch.zeros(B, device=device)
        for i in range(B):
            orig_len = len(original_queries[i].split())
            rewr_len = len(rewritten_queries[i].split())
            # Ideal: rewrite is 1-3x original length
            ratio = rewr_len / max(orig_len, 1)
            if 1.0 <= ratio <= 3.0:
                length_rewards[i] = 1.0
            elif ratio < 1.0:
                length_rewards[i] = ratio  # Penalty for too short
            else:
                length_rewards[i] = max(0, 1.0 - (ratio - 3.0) * 0.2)  # Gradual penalty
        rewards += self.weights["length_penalty"] * length_rewards

        return rewards

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages for GRPO.

        For each group of `group_size` candidate rewrites of the same query,
        normalize rewards within the group to get advantages.

        GRPO advantage: A_i = (R_i - mean(R_group)) / (std(R_group) + ε)

        Args:
            rewards: (B * group_size,) flat reward tensor.
            group_size: Number of candidates per query.

        Returns:
            advantages: (B * group_size,) normalized advantages.
        """
        # Reshape to (B, group_size)
        B = rewards.size(0) // group_size
        grouped = rewards.view(B, group_size)

        # Group statistics
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True) + 1e-8

        # Normalize within group
        advantages = (grouped - mean) / std

        return advantages.view(-1)  # Flatten back
