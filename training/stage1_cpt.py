"""
stage1_cpt.py
=============
Stage 1: Continual Pre-Training (CPT) on Chase domain text.

Purpose:
    Adapt Gemma3-1B to Chase-specific vocabulary and query patterns before
    any personalization is introduced. This ensures the model understands
    banking terminology (e.g., "Sapphire" = credit card, not gemstone).

Method:
    - Use the existing "complete: {partial_query}" prefix task from the
      Gemma3 notebook alongside a new "rewrite: {masked_query}" task.
    - Train only LoRA adapters (base LLM frozen).
    - Use the cleaned CDA utterance corpus (~10M queries).

Reference:
    R&R method: CPT with LoRA on a single NVIDIA 4090 produces significant
    gains on professional QA across finance, law, and education domains.
"""

import logging
import os
from functools import partial
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..data.dataset import CPTDataset, cpt_collate_fn
from ..models.personalized_rewriter import PersonalizedQueryRewriter

logger = logging.getLogger(__name__)


class CPTTrainer:
    """
    Continual Pre-Training trainer.

    Trains only the LLM's LoRA adapters on Chase domain text, using the
    existing completion task format plus a masked rewrite task.

    Args:
        model: PersonalizedQueryRewriter instance.
        config: Training configuration dict (from config.yaml['training_cpt']).
        train_texts: List of query strings for training.
        val_texts: List of query strings for validation.
    """

    def __init__(
        self,
        model: PersonalizedQueryRewriter,
        config: Dict[str, Any],
        train_texts: List[str],
        val_texts: List[str],
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Set training mode: only LoRA weights
        self.model.set_training_mode("cpt")

        # Create datasets
        self.train_dataset = CPTDataset(
            tokenizer=model.tokenizer,
            texts=train_texts,
            max_length=config["max_length"],
        )
        self.val_dataset = CPTDataset(
            tokenizer=model.tokenizer,
            texts=val_texts,
            max_length=config["max_length"],
        )

        # Collate function with pad token
        pad_id = model.tokenizer.pad_token_id
        self.collate_fn = partial(cpt_collate_fn, pad_token_id=pad_id)

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        # Optimizer (only LoRA params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Scheduler
        total_steps = (
            len(self.train_loader)
            // config["gradient_accumulation_steps"]
            * config["epochs"]
        )
        warmup_steps = int(total_steps * config["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.grad_accum_steps = config["gradient_accumulation_steps"]
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self) -> Dict[str, List[float]]:
        """
        Run CPT training loop.

        Returns:
            Dict with 'train_losses' and 'val_losses' per epoch.
        """
        history = {"train_losses": [], "val_losses": []}
        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            # --- Training ---
            train_loss = self._train_epoch(epoch)
            history["train_losses"].append(train_loss)

            # --- Validation ---
            val_loss = self._validate(epoch)
            history["val_losses"].append(val_loss)

            logger.info(
                "CPT Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                epoch + 1, self.config["epochs"], train_loss, val_loss,
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_components(
                    os.path.join(self.output_dir, "best")
                )
                logger.info("  Saved best model (val_loss=%.4f)", val_loss)

            # Save checkpoint
            self.model.save_components(
                os.path.join(self.output_dir, f"epoch_{epoch+1}")
            )

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward (no user features in CPT)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"] / self.grad_accum_steps
            loss.backward()

            total_loss += outputs["loss"].item()
            num_steps += 1

            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (step + 1) % self.config["logging_steps"] == 0:
                avg = total_loss / num_steps
                logger.info(
                    "  CPT [Epoch %d, Step %d/%d] loss=%.4f, lr=%.2e",
                    epoch + 1, step + 1, len(self.train_loader),
                    avg, self.scheduler.get_last_lr()[0],
                )

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_steps = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs["loss"].item()
            num_steps += 1

        return total_loss / max(num_steps, 1)
