"""
stage3_grpo.py
==============
Stage 3: Group Relative Policy Optimization (GRPO) alignment.

Purpose:
    Align the rewriter with actual search system performance. The SFT model
    learns surface patterns; GRPO teaches it to produce rewrites that
    improve downstream retrieval quality.

Method (following WeWrite, GRAPE):
    1. For each query, generate `G` candidate rewrites using the SFT model.
    2. Compute rewards for each candidate:
       - Index Hit Rate: does the rewrite retrieve the gold navlink?
       - Semantic Fidelity: cosine similarity with original query.
       - Length appropriateness.
    3. Compute group-relative advantages: A_i = (R_i - mean(R)) / std(R)
    4. Update policy with clipped surrogate objective + KL penalty:
       L = -E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)] + β * KL(π_θ || π_ref)

    GRPO eliminates the need for a separate critic network by computing
    advantages within groups of candidates for the same query.

Reference:
    DeepSeekMath GRPO, WeWrite (2025), CardRewriter
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import GRPODataset, grpo_collate_fn
from ..data.feature_engineering import UserFeatureEncoder
from ..models.personalized_rewriter import PersonalizedQueryRewriter
from ..models.losses import RewriteRewardModel

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO alignment trainer.

    Trains all components (FT-Transformer + E2P + gate + LLM LoRA) using
    group-relative policy optimization with retrieval-based rewards.

    Args:
        model: PersonalizedQueryRewriter (post-SFT).
        config: Training config (config.yaml['training_grpo']).
        train_df: Training DataFrame with query, user products, gold navlink.
        feature_encoder: UserFeatureEncoder instance.
        search_index_fn: Optional callable for index hit computation.
            Signature: fn(query: str) -> List[str] (returns top-K navlinks).
    """

    def __init__(
        self,
        model: PersonalizedQueryRewriter,
        config: Dict[str, Any],
        train_df,
        feature_encoder: UserFeatureEncoder,
        search_index_fn=None,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Keep a frozen reference model for KL computation
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Set training mode: all components
        self.model.set_training_mode("grpo")

        # Dataset
        self.dataset = GRPODataset(
            tokenizer=model.tokenizer,
            df=train_df,
            feature_encoder=feature_encoder,
            max_length=config["max_length"],
        )

        pad_id = model.tokenizer.pad_token_id
        from functools import partial
        self.collate_fn = partial(grpo_collate_fn, pad_token_id=pad_id)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        # Reward model
        self.reward_model = RewriteRewardModel(
            weights=config.get("reward_weights", {}),
        )

        # Search index function (for index hit rate reward)
        self.search_index_fn = search_index_fn

        # GRPO hyperparameters
        self.num_generations = config["num_generations"]
        self.kl_coeff = config["kl_coeff"]
        self.temperature = config["temperature"]
        self.clip_eps = 0.2  # PPO-style clipping

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.grad_accum_steps = config["gradient_accumulation_steps"]
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self) -> Dict[str, List[float]]:
        """
        Run GRPO training loop.

        For each batch:
            1. Generate G candidate rewrites per query.
            2. Compute rewards for each candidate.
            3. Compute group-relative advantages.
            4. Update policy with clipped surrogate + KL penalty.

        Returns:
            Dict with losses, rewards, kl_divs per step.
        """
        history = {
            "losses": [],
            "avg_rewards": [],
            "avg_kl": [],
            "avg_gate_scores": [],
        }

        self.model.train()

        for epoch in range(self.config["epochs"]):
            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_kl = 0.0
            num_steps = 0

            self.optimizer.zero_grad()

            for step, batch in enumerate(self.dataloader):
                step_result = self._grpo_step(batch)

                if step_result is None:
                    continue

                loss = step_result["loss"] / self.grad_accum_steps
                loss.backward()

                epoch_loss += step_result["loss"].item()
                epoch_reward += step_result["avg_reward"]
                epoch_kl += step_result["avg_kl"]
                num_steps += 1

                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (step + 1) % (self.config.get("logging_steps", 10)) == 0:
                    logger.info(
                        "  GRPO [Epoch %d, Step %d] loss=%.4f, reward=%.3f, kl=%.4f",
                        epoch + 1, step + 1,
                        epoch_loss / max(num_steps, 1),
                        epoch_reward / max(num_steps, 1),
                        epoch_kl / max(num_steps, 1),
                    )

            # End of epoch
            if num_steps > 0:
                avg_loss = epoch_loss / num_steps
                avg_reward = epoch_reward / num_steps
                avg_kl = epoch_kl / num_steps

                history["losses"].append(avg_loss)
                history["avg_rewards"].append(avg_reward)
                history["avg_kl"].append(avg_kl)

                logger.info(
                    "GRPO Epoch %d — loss=%.4f, avg_reward=%.3f, avg_kl=%.4f",
                    epoch + 1, avg_loss, avg_reward, avg_kl,
                )

            # Save checkpoint
            self.model.save_components(
                os.path.join(self.output_dir, f"epoch_{epoch+1}")
            )

        # Save final model
        self.model.save_components(os.path.join(self.output_dir, "final"))
        return history

    def _grpo_step(self, batch: Dict) -> Optional[Dict]:
        """
        Single GRPO training step.

        Steps:
            1. Generate G candidates per query using current policy.
            2. Score each candidate with the reward model.
            3. Compute group-relative advantages.
            4. Compute policy loss with KL regularization.

        Returns:
            Dict with 'loss', 'avg_reward', 'avg_kl', or None if generation fails.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        user_features = batch["user_features"].to(self.device)
        gold_navlinks = batch["gold_navlinks"]
        original_queries = batch["original_queries"]

        B = input_ids.size(0)
        G = self.num_generations

        # --- Step 1: Generate G candidates per query ---
        self.model.eval()
        all_generated_ids = []
        all_generated_texts = []

        with torch.no_grad():
            # Encode user features
            user_embed = self.model.user_encoder(user_features)
            prefix_tokens = self.model.e2p(user_embed)

            embed_layer = self.model.llm.get_input_embeddings()
            input_embeds = embed_layer(input_ids)
            combined_embeds = torch.cat([prefix_tokens, input_embeds], dim=1)

            prefix_mask = torch.ones(
                B, prefix_tokens.size(1),
                dtype=attention_mask.dtype, device=self.device,
            )
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            try:
                outputs = self.model.llm.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    max_new_tokens=self.config.get("max_new_tokens", 50),
                    num_return_sequences=G,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                )
            except Exception as e:
                logger.warning("Generation failed: %s", e)
                return None

            for output in outputs:
                text = self.model.tokenizer.decode(output, skip_special_tokens=True)
                for prefix in ["rewrite:", "complete:"]:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                if "→" in text:
                    text = text.split("→", 1)[-1]
                all_generated_texts.append(text.strip())
                all_generated_ids.append(output)

        # --- Step 2: Compute rewards ---
        self.model.train()

        # Expand original queries and gold navlinks for group
        expanded_queries = []
        expanded_navlinks = []
        for i in range(B):
            expanded_queries.extend([original_queries[i]] * G)
            expanded_navlinks.extend([gold_navlinks[i]] * G)

        # Get retrieval results if search index is available
        retrieved_navlinks = None
        if self.search_index_fn is not None:
            retrieved_navlinks = [
                self.search_index_fn(q) for q in all_generated_texts
            ]

        # Simple embedding-based rewards (using mean of token embeddings)
        with torch.no_grad():
            orig_embeds = self._get_query_embeddings(expanded_queries)
            rewr_embeds = self._get_query_embeddings(all_generated_texts)

        rewards = self.reward_model.compute_rewards(
            original_queries=expanded_queries,
            rewritten_queries=all_generated_texts,
            gold_navlinks=expanded_navlinks,
            original_embeddings=orig_embeds,
            rewritten_embeddings=rewr_embeds,
            retrieved_navlinks=retrieved_navlinks,
        )

        # --- Step 3: Group-relative advantages ---
        advantages = self.reward_model.compute_group_advantages(rewards, G)

        # --- Step 4: Policy loss + KL ---
        # Compute log probabilities under current and reference policy
        # For simplicity, use a simplified version computing token-level loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl = 0.0

        for i, gen_ids in enumerate(all_generated_ids):
            gen_ids = gen_ids.unsqueeze(0).to(self.device)
            gen_mask = (gen_ids != self.model.tokenizer.pad_token_id).long()

            # Current policy log-prob
            query_idx = i // G
            uf = user_features[query_idx].unsqueeze(0)
            u_embed = self.model.user_encoder(uf)
            pref = self.model.e2p(u_embed)

            e_layer = self.model.llm.get_input_embeddings()
            gen_embeds = e_layer(gen_ids)
            comb = torch.cat([pref, gen_embeds], dim=1)
            comb_mask = torch.cat([
                torch.ones(1, pref.size(1), dtype=gen_mask.dtype, device=self.device),
                gen_mask,
            ], dim=1)

            # Labels: shift for causal LM
            prefix_len = pref.size(1)
            labels = gen_ids.clone()
            full_labels = torch.cat([
                torch.full((1, prefix_len), -100, dtype=labels.dtype, device=self.device),
                labels,
            ], dim=1)

            outputs = self.model.llm(
                inputs_embeds=comb,
                attention_mask=comb_mask,
                labels=full_labels,
            )

            # Weighted by advantage
            adv = advantages[i].detach()
            policy_loss = outputs.loss * adv

            # KL divergence with reference model
            with torch.no_grad():
                ref_outputs = self.ref_model.llm(
                    inputs_embeds=comb.detach(),
                    attention_mask=comb_mask,
                )
                ref_logits = ref_outputs.logits

            curr_logits = outputs.logits
            kl = F.kl_div(
                F.log_softmax(curr_logits[:, prefix_len:, :], dim=-1),
                F.softmax(ref_logits[:, prefix_len:, :], dim=-1),
                reduction="batchmean",
            )

            total_loss = total_loss - policy_loss + self.kl_coeff * kl
            total_kl += kl.item()

        total_loss = total_loss / (B * G)

        return {
            "loss": total_loss,
            "avg_reward": rewards.mean().item(),
            "avg_kl": total_kl / max(B * G, 1),
        }

    @torch.no_grad()
    def _get_query_embeddings(self, queries: List[str]) -> torch.Tensor:
        """Get simple query embeddings using mean of LLM input embeddings."""
        embeddings = []
        embed_layer = self.model.llm.get_input_embeddings()

        for q in queries:
            tokens = self.model.tokenizer(
                q, return_tensors="pt", truncation=True, max_length=64,
            ).to(self.device)
            embeds = embed_layer(tokens["input_ids"])
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            mean_embed = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(mean_embed.squeeze(0))

        return torch.stack(embeddings)
