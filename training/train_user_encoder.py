"""
train_user_encoder.py
=====================
Pre-train the FT-Transformer user encoder with contrastive learning.

Purpose:
    Learn user embeddings that capture search-relevant behavioral patterns.
    Users who click on similar features should have similar embeddings,
    regardless of their raw product feature differences.

Method (following CoPPS, KDD 2023):
    - Positive pairs: (user_features, clicked_navlink)
    - Negatives: In-batch negatives via InfoNCE loss
    - Hard negatives: Sample from same navlink category but different user
      segment to push the model to capture user-specific signals.

This pre-training step runs BEFORE Stage 2 (SFT) to give the E2P module
a well-initialized user embedding space to project from.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import UserEncoderContrastiveDataset
from ..data.feature_engineering import UserFeatureEncoder
from ..models.ft_transformer import FTTransformer
from ..models.losses import InfoNCELoss

logger = logging.getLogger(__name__)


class UserEncoderTrainer:
    """
    Contrastive pre-training for the FT-Transformer user encoder.

    Trains the encoder to produce user embeddings where users with similar
    search behavior (clicking similar navlinks) are close in embedding space.

    Args:
        user_encoder: FTTransformer instance.
        config: Training config (config.yaml['training_user_encoder']).
        train_df: Training DataFrame with clicked interactions.
        val_df: Validation DataFrame.
        feature_encoder: UserFeatureEncoder instance.
        navlink_vocab: Dict mapping navlink strings to IDs.
    """

    def __init__(
        self,
        user_encoder: FTTransformer,
        config: Dict[str, Any],
        train_df,
        val_df,
        feature_encoder: UserFeatureEncoder,
        navlink_vocab: Optional[Dict[str, int]] = None,
    ):
        self.user_encoder = user_encoder
        self.config = config
        self.device = next(user_encoder.parameters()).device

        # Build dataset
        self.train_dataset = UserEncoderContrastiveDataset(
            df=train_df,
            feature_encoder=feature_encoder,
            navlink_vocab=navlink_vocab,
        )

        # Capture navlink vocab for val dataset
        self.navlink_vocab = self.train_dataset.navlink_vocab

        self.val_dataset = UserEncoderContrastiveDataset(
            df=val_df,
            feature_encoder=feature_encoder,
            navlink_vocab=self.navlink_vocab,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,  # Important for InfoNCE with in-batch negatives
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # Loss
        self.criterion = InfoNCELoss(
            temperature=config["temperature"],
            navlink_embed_dim=user_encoder.output_dim,
            num_navlinks=self.train_dataset.num_navlinks,
        ).to(self.device)

        # Optimizer
        all_params = list(user_encoder.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self) -> Dict[str, List[float]]:
        """
        Run contrastive pre-training loop.

        Returns:
            Dict with train_losses and val_losses per epoch.
        """
        history = {"train_losses": [], "val_losses": []}
        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            train_loss = self._train_epoch(epoch)
            history["train_losses"].append(train_loss)

            val_loss = self._validate(epoch)
            history["val_losses"].append(val_loss)

            logger.info(
                "UserEncoder Epoch %d/%d — train=%.4f, val=%.4f",
                epoch + 1, self.config["epochs"], train_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.user_encoder.state_dict(),
                    os.path.join(self.output_dir, "best_user_encoder.pt"),
                )
                logger.info("  Saved best user encoder (val_loss=%.4f)", val_loss)

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.user_encoder.train()
        self.criterion.train()
        total_loss = 0.0
        num_steps = 0

        for batch in self.train_loader:
            user_features = batch["user_features"].to(self.device)
            navlink_ids = batch["navlink_id"].to(self.device)

            # Forward through user encoder
            user_embeddings = self.user_encoder(user_features)

            # Compute InfoNCE loss
            loss = self.criterion(user_embeddings, navlink_ids)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.user_encoder.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation."""
        self.user_encoder.eval()
        self.criterion.eval()
        total_loss = 0.0
        num_steps = 0

        for batch in self.val_loader:
            user_features = batch["user_features"].to(self.device)
            navlink_ids = batch["navlink_id"].to(self.device)

            user_embeddings = self.user_encoder(user_features)
            loss = self.criterion(user_embeddings, navlink_ids)

            total_loss += loss.item()
            num_steps += 1

        return total_loss / max(num_steps, 1)
