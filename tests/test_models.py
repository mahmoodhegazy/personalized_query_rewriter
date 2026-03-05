"""
test_models.py
==============
Unit tests for FT-Transformer, E2P, Gate, and loss functions.
Tests model components without loading the full LLM.
"""

import pytest
import torch
import torch.nn as nn

from personalized_query_rewriter.models.ft_transformer import (
    FTTransformer,
    FeatureTokenizer,
)
from personalized_query_rewriter.models.e2p_projector import E2PProjector
from personalized_query_rewriter.models.personalization_gate import PersonalizationGate
from personalized_query_rewriter.models.losses import InfoNCELoss, RewriteRewardModel


# =============================================================================
# FT-Transformer Tests
# =============================================================================

class TestFTTransformer:
    def test_feature_tokenizer_shape(self):
        tokenizer = FeatureTokenizer(num_features=55, d_model=128)
        x = torch.randn(4, 55)  # Batch of 4
        out = tokenizer(x)
        assert out.shape == (4, 55, 128)

    def test_ft_transformer_forward(self):
        model = FTTransformer(
            num_features=55, d_model=128, n_heads=4, n_layers=3,
            output_dim=128,
        )
        x = torch.randn(4, 55)
        out = model(x)
        assert out.shape == (4, 128)

    def test_ft_transformer_different_sizes(self):
        for n_feat in [10, 55, 100]:
            model = FTTransformer(num_features=n_feat, d_model=64, n_heads=2,
                                  n_layers=1, output_dim=32)
            x = torch.randn(2, n_feat)
            out = model(x)
            assert out.shape == (2, 32)

    def test_attention_weights(self):
        model = FTTransformer(num_features=55, d_model=128, n_heads=4, n_layers=2)
        x = torch.randn(2, 55)
        weights = model.get_attention_weights(x)
        assert len(weights) == 2  # One per layer
        # Each: (batch, heads_or_avg, seq, seq)
        assert weights[0].shape[0] == 2  # Batch size

    def test_gradient_flow(self):
        model = FTTransformer(num_features=10, d_model=32, n_heads=2,
                              n_layers=1, output_dim=16)
        x = torch.randn(2, 10, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# =============================================================================
# E2P Projector Tests
# =============================================================================

class TestE2PProjector:
    def test_output_shape_single_prefix(self):
        e2p = E2PProjector(
            user_embed_dim=128, llm_hidden_dim=1152,
            n_prefix_tokens=1,
        )
        user_embed = torch.randn(4, 128)
        prefix = e2p(user_embed)
        assert prefix.shape == (4, 1, 1152)

    def test_output_shape_multi_prefix(self):
        e2p = E2PProjector(
            user_embed_dim=128, llm_hidden_dim=1152,
            n_prefix_tokens=4,
        )
        user_embed = torch.randn(2, 128)
        prefix = e2p(user_embed)
        assert prefix.shape == (2, 4, 1152)

    def test_param_count(self):
        e2p = E2PProjector(
            user_embed_dim=128, llm_hidden_dim=1152,
            n_prefix_tokens=1, projection_layers=2,
        )
        count = e2p.get_param_count()
        # Should be ~100K parameters
        assert count > 50000
        assert count < 500000

    def test_gating(self):
        e2p = E2PProjector(
            user_embed_dim=128, llm_hidden_dim=256,
            n_prefix_tokens=1,
        )
        user_embed = torch.randn(2, 128)

        e2p.use_gating = True
        out_gated = e2p(user_embed)

        e2p.use_gating = False
        out_ungated = e2p(user_embed)

        # Outputs should differ with gating
        assert not torch.allclose(out_gated, out_ungated)


# =============================================================================
# Personalization Gate Tests
# =============================================================================

class TestPersonalizationGate:
    def test_forward_shape(self):
        gate = PersonalizationGate(
            user_embed_dim=128, query_embed_dim=128, hidden_dim=64,
        )
        user_embed = torch.randn(4, 128)
        query_embed = torch.randn(4, 128)
        score = gate(user_embed, query_embed)
        assert score.shape == (4, 1)
        # Scores should be in [0, 1]
        assert (score >= 0).all() and (score <= 1).all()

    def test_decide(self):
        gate = PersonalizationGate(threshold=0.5)
        user_embed = torch.randn(4, 128)
        query_embed = torch.randn(4, 128)
        decisions = gate.decide(user_embed, query_embed)
        assert decisions.dtype == torch.bool
        assert decisions.shape == (4,)

    def test_loss_computation(self):
        gate = PersonalizationGate()
        user_embed = torch.randn(4, 128)
        query_embed = torch.randn(4, 128)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = gate.compute_loss(user_embed, query_embed, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_query_projector(self):
        gate = PersonalizationGate(user_embed_dim=128, query_embed_dim=128)
        gate.set_query_projector(llm_hidden_dim=1152, query_embed_dim=128)
        user_embed = torch.randn(2, 128)
        query_embed = torch.randn(2, 1152)  # LLM hidden dim
        score = gate(user_embed, query_embed)
        assert score.shape == (2, 1)


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestInfoNCELoss:
    def test_loss_computation(self):
        loss_fn = InfoNCELoss(temperature=0.07, navlink_embed_dim=128, num_navlinks=50)
        user_embeds = torch.randn(16, 128)
        navlink_ids = torch.randint(0, 50, (16,))
        loss = loss_fn(user_embeds, navlink_ids)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_alignment(self):
        """Loss should be lower when embeddings align with their navlinks."""
        loss_fn = InfoNCELoss(temperature=0.1, navlink_embed_dim=8, num_navlinks=4)

        # Make user embeddings close to their navlink embeddings
        with torch.no_grad():
            navlink_embeds = loss_fn.navlink_embeddings.weight.clone()

        navlink_ids = torch.arange(4)
        user_embeds_aligned = navlink_embeds[navlink_ids] + torch.randn(4, 8) * 0.01
        user_embeds_random = torch.randn(4, 8)

        loss_aligned = loss_fn(user_embeds_aligned, navlink_ids)
        loss_random = loss_fn(user_embeds_random, navlink_ids)

        assert loss_aligned.item() < loss_random.item()


class TestRewriteRewardModel:
    def test_reward_computation(self):
        reward_model = RewriteRewardModel()

        original = ["rewards", "fees"]
        rewritten = ["sapphire reserve rewards balance", "checking account fees"]
        gold_navlinks = ["/rewards/balance", "/fees/checking"]

        orig_embeds = torch.randn(2, 128)
        rewr_embeds = torch.randn(2, 128)

        rewards = reward_model.compute_rewards(
            original, rewritten, gold_navlinks,
            orig_embeds, rewr_embeds,
        )
        assert rewards.shape == (2,)

    def test_rewards_with_retrieval(self):
        reward_model = RewriteRewardModel()

        rewards = reward_model.compute_rewards(
            original_queries=["rewards"],
            rewritten_queries=["sapphire rewards"],
            gold_navlinks=["/rewards/balance"],
            original_embeddings=torch.randn(1, 128),
            rewritten_embeddings=torch.randn(1, 128),
            retrieved_navlinks=[["/rewards/balance", "/rewards/redeem"]],
        )
        # Should get index hit reward (gold is at position 0)
        assert rewards[0].item() > 0

    def test_group_advantages(self):
        reward_model = RewriteRewardModel()
        rewards = torch.tensor([0.8, 0.2, 0.5, 0.9, 0.1, 0.6])
        advantages = reward_model.compute_group_advantages(rewards, group_size=3)
        assert advantages.shape == (6,)
        # Within each group of 3, advantages should sum to ~0
        group1 = advantages[:3]
        group2 = advantages[3:]
        assert abs(group1.mean().item()) < 0.01
        assert abs(group2.mean().item()) < 0.01


# =============================================================================
# Integration Test (without LLM)
# =============================================================================

class TestIntegration:
    def test_full_encoding_pipeline(self):
        """Test FT-Transformer → E2P → prefix tokens pipeline."""
        ft = FTTransformer(num_features=55, d_model=64, n_heads=2,
                           n_layers=1, output_dim=128)
        e2p = E2PProjector(user_embed_dim=128, llm_hidden_dim=256,
                           n_prefix_tokens=1)

        features = torch.randn(4, 55)
        user_embed = ft(features)
        prefix = e2p(user_embed)

        assert user_embed.shape == (4, 128)
        assert prefix.shape == (4, 1, 256)

    def test_full_gate_pipeline(self):
        """Test encoding + gate decision pipeline."""
        ft = FTTransformer(num_features=55, d_model=64, n_heads=2,
                           n_layers=1, output_dim=128)
        gate = PersonalizationGate(user_embed_dim=128, query_embed_dim=128)

        features = torch.randn(4, 55)
        query_embed = torch.randn(4, 128)

        user_embed = ft(features)
        decisions = gate.decide(user_embed, query_embed)

        assert decisions.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
