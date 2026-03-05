"""
Personalized Query Rewriter for Chase UEP
==========================================

Architecture: FT-Transformer → E2P Projection → Gemma3-1B with LoRA
Training Pipeline: CPT → SFT → GRPO

Modules:
    data      - Data loading, feature engineering, PyTorch datasets
    models    - FT-Transformer, E2P, personalization gate, full rewriter
    training  - Three-stage training pipeline + user encoder training
    evaluation - Metrics, evaluator, position bias correction
    inference - End-to-end pipeline with semantic caching
"""

__version__ = "0.1.0"
