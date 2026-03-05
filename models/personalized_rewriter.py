"""
personalized_rewriter.py
========================
End-to-end Personalized Query Rewriter combining:

    1. FT-Transformer   → User profile → dense embedding
    2. E2P Projector     → User embedding → soft prefix tokens
    3. Personalization Gate → Decide whether to personalize
    4. Gemma3-1B + LoRA  → Generate personalized query rewrite

The forward pass:
    user_features → FT-Transformer → user_embed
    user_embed → E2P → prefix_tokens (B, n_prefix, hidden_dim)
    user_embed + query_embed → Gate → should_personalize?

    If personalizing:
        [prefix_tokens ⊕ input_embeddings] → Gemma3-1B(LoRA) → rewritten query
    Else:
        input_embeddings → Gemma3-1B(LoRA) → passthrough completion
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .ft_transformer import FTTransformer
from .e2p_projector import E2PProjector
from .personalization_gate import PersonalizationGate

logger = logging.getLogger(__name__)


class PersonalizedQueryRewriter(nn.Module):
    """
    Full personalized query rewriting model.

    Components:
        - user_encoder: FT-Transformer encoding user profile features
        - e2p: E2P projector mapping user embedding to LLM prefix tokens
        - gate: Personalization gate deciding when to personalize
        - llm: Gemma3-1B with LoRA adapters for text generation

    Training modes:
        - "cpt":  Only LLM LoRA weights are trained (domain adaptation)
        - "sft":  user_encoder + e2p + LLM LoRA trained (personalized rewriting)
        - "grpo": All components fine-tuned with RL rewards
        - "gate": Only gate network trained (ambiguity classification)

    Args:
        config: Configuration dictionary (from config.yaml).
        load_llm: Whether to load the LLM (False for unit testing).
    """

    def __init__(self, config: Dict[str, Any], load_llm: bool = True):
        super().__init__()
        self.config = config

        # --- User Encoder (FT-Transformer) ---
        ft_cfg = config["ft_transformer"]
        self.user_encoder = FTTransformer(
            num_features=ft_cfg["num_features"],
            d_model=ft_cfg["d_model"],
            n_heads=ft_cfg["n_heads"],
            n_layers=ft_cfg["n_layers"],
            d_ffn_factor=ft_cfg["d_ffn_factor"],
            dropout=ft_cfg["dropout"],
            attention_dropout=ft_cfg["attention_dropout"],
            ffn_dropout=ft_cfg["ffn_dropout"],
            output_dim=ft_cfg["output_dim"],
        )

        # --- E2P Projector ---
        e2p_cfg = config["e2p"]
        self.e2p = E2PProjector(
            user_embed_dim=e2p_cfg["user_embed_dim"],
            llm_hidden_dim=e2p_cfg["llm_hidden_dim"],
            n_prefix_tokens=e2p_cfg["n_prefix_tokens"],
            projection_layers=e2p_cfg["projection_layers"],
            dropout=e2p_cfg["projection_dropout"],
        )

        # --- Personalization Gate ---
        gate_cfg = config["gate"]
        self.gate = PersonalizationGate(
            user_embed_dim=gate_cfg["input_dim"],
            query_embed_dim=gate_cfg["input_dim"],
            hidden_dim=gate_cfg["hidden_dim"],
            threshold=gate_cfg["threshold"],
            dropout=gate_cfg["dropout"],
        )

        # --- LLM (Gemma3-1B + LoRA) ---
        self.tokenizer = None
        self.llm = None
        if load_llm:
            self._load_llm(config["llm"])

        self._log_param_counts()

    def _load_llm(self, llm_cfg: Dict):
        """Load Gemma3-1B with LoRA adapters."""
        model_name = llm_cfg["model_name"]
        logger.info("Loading LLM: %s", model_name)

        # Determine dtype
        dtype_str = llm_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model.config.pad_token_id = self.tokenizer.eos_token_id

        # Apply LoRA
        lora_cfg = llm_cfg["lora"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        )
        self.llm = get_peft_model(base_model, lora_config)
        self.llm.enable_input_require_grads()

        # Set gate query projector to match LLM hidden dim
        llm_hidden = self.config["e2p"]["llm_hidden_dim"]
        query_embed_dim = self.config["gate"]["input_dim"]
        self.gate.set_query_projector(llm_hidden, query_embed_dim)

        logger.info("LLM loaded with LoRA adapters")

    def _log_param_counts(self):
        """Log parameter counts for each component."""
        counts = {
            "user_encoder": sum(p.numel() for p in self.user_encoder.parameters()),
            "e2p": self.e2p.get_param_count(),
            "gate": sum(p.numel() for p in self.gate.parameters()),
        }
        if self.llm is not None:
            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            counts["llm_trainable"] = trainable
            counts["llm_total"] = total

        for name, count in counts.items():
            logger.info("  %s: %s params", name, f"{count:,}")

    def set_training_mode(self, mode: str):
        """
        Configure which components are trainable for each training stage.

        Modes:
            'cpt'  : Only LLM LoRA weights
            'sft'  : user_encoder + e2p + LLM LoRA
            'grpo' : All components (user_encoder + e2p + gate + LLM LoRA)
            'gate' : Only gate network
            'user_encoder': Only FT-Transformer (for contrastive pre-training)
        """
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if mode == "cpt":
            # Only LoRA adapters
            if self.llm is not None:
                for name, param in self.llm.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        elif mode == "sft":
            # User encoder + E2P + LoRA
            for param in self.user_encoder.parameters():
                param.requires_grad = True
            for param in self.e2p.parameters():
                param.requires_grad = True
            if self.llm is not None:
                for name, param in self.llm.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        elif mode == "grpo":
            # Everything trainable
            for param in self.user_encoder.parameters():
                param.requires_grad = True
            for param in self.e2p.parameters():
                param.requires_grad = True
            for param in self.gate.parameters():
                param.requires_grad = True
            if self.llm is not None:
                for name, param in self.llm.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        elif mode == "gate":
            for param in self.gate.parameters():
                param.requires_grad = True

        elif mode == "user_encoder":
            for param in self.user_encoder.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown training mode: {mode}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Training mode '%s': %s / %s params trainable (%.2f%%)",
                     mode, f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (SFT mode).

        If user_features is provided, inject E2P prefix tokens. Otherwise,
        run standard LLM forward (for CPT stage).

        Args:
            input_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) attention mask.
            labels: (B, seq_len) target labels (-100 for masked).
            user_features: (B, num_features) raw user profile features.

        Returns:
            Dict with 'loss', 'logits', and optionally 'gate_score'.
        """
        result = {}

        if user_features is not None and self.llm is not None:
            # --- Personalized forward pass ---

            # 1. Encode user features
            user_embed = self.user_encoder(user_features)  # (B, output_dim)

            # 2. Generate prefix tokens via E2P
            prefix_tokens = self.e2p(user_embed)  # (B, n_prefix, llm_hidden)

            # 3. Get input embeddings from LLM
            embed_layer = self.llm.get_input_embeddings()
            input_embeds = embed_layer(input_ids)  # (B, seq_len, hidden)

            # 4. Concatenate prefix + input embeddings
            combined_embeds = torch.cat([prefix_tokens, input_embeds], dim=1)

            # 5. Extend attention mask for prefix tokens
            prefix_mask = torch.ones(
                input_ids.size(0),
                prefix_tokens.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # 6. Extend labels with -100 for prefix positions
            if labels is not None:
                prefix_labels = torch.full(
                    (labels.size(0), prefix_tokens.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                combined_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                combined_labels = None

            # 7. Forward through LLM
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
            )

            result["loss"] = outputs.loss
            result["logits"] = outputs.logits

            # 8. Compute gate score (for monitoring / GRPO training)
            # Use mean of input embeddings as query representation
            query_embed = input_embeds.mean(dim=1)  # (B, hidden)
            gate_score = self.gate(user_embed, query_embed)
            result["gate_score"] = gate_score

        else:
            # --- Standard forward pass (CPT or no user features) ---
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            result["loss"] = outputs.loss
            result["logits"] = outputs.logits

        return result

    @torch.no_grad()
    def generate(
        self,
        query: str,
        user_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        use_gate: bool = True,
    ) -> List[str]:
        """
        Generate personalized query rewrites.

        Args:
            query: Original query text.
            user_features: (num_features,) user profile tensor. If None, skip personalization.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Beam search width.
            num_return_sequences: Number of rewrites to return.
            use_gate: Whether to check the personalization gate.

        Returns:
            List of rewritten query strings.
        """
        self.eval()
        device = next(self.parameters()).device

        prompt = f"rewrite: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        if user_features is not None:
            user_features = user_features.unsqueeze(0).to(device)

            # Check gate
            if use_gate:
                user_embed = self.user_encoder(user_features)
                embed_layer = self.llm.get_input_embeddings()
                input_embeds = embed_layer(inputs["input_ids"])
                query_embed = input_embeds.mean(dim=1)
                should_personalize = self.gate.decide(user_embed, query_embed)

                if not should_personalize.item():
                    # Passthrough: return original query
                    return [query]

            # Generate with prefix injection
            user_embed = self.user_encoder(user_features)
            prefix_tokens = self.e2p(user_embed)

            embed_layer = self.llm.get_input_embeddings()
            input_embeds = embed_layer(inputs["input_ids"])
            combined_embeds = torch.cat([prefix_tokens, input_embeds], dim=1)

            prefix_mask = torch.ones(
                1, prefix_tokens.size(1),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            )
            combined_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )
        else:
            # Standard generation without personalization
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )

        # Decode outputs
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt prefix
            for prefix in ["rewrite:", "complete:"]:
                if text.startswith(prefix):
                    text = text[len(prefix):]
            # Remove arrow separator if present
            if "→" in text:
                text = text.split("→", 1)[-1]
            results.append(text.strip())

        return list(dict.fromkeys(results))  # Deduplicate preserving order

    def save_components(self, output_dir: str):
        """Save all model components separately."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        torch.save(self.user_encoder.state_dict(),
                   os.path.join(output_dir, "user_encoder.pt"))
        torch.save(self.e2p.state_dict(),
                   os.path.join(output_dir, "e2p_projector.pt"))
        torch.save(self.gate.state_dict(),
                   os.path.join(output_dir, "personalization_gate.pt"))

        if self.llm is not None:
            self.llm.save_pretrained(os.path.join(output_dir, "llm_lora"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        logger.info("All components saved to %s", output_dir)

    def load_components(self, output_dir: str):
        """Load all model components."""
        import os
        from peft import PeftModel

        self.user_encoder.load_state_dict(
            torch.load(os.path.join(output_dir, "user_encoder.pt"), weights_only=True)
        )
        self.e2p.load_state_dict(
            torch.load(os.path.join(output_dir, "e2p_projector.pt"), weights_only=True)
        )
        self.gate.load_state_dict(
            torch.load(os.path.join(output_dir, "personalization_gate.pt"), weights_only=True)
        )

        logger.info("All components loaded from %s", output_dir)
