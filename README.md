# Personalized Query Rewriter for Chase UEP

## Architecture: FT-Transformer → E2P Projection → Gemma3-1B with LoRA

A production-ready personalized query rewriting system that conditions on user profile features to improve search relevance in the Chase Universal Entry Point (UEP). The system selectively rewrites only ambiguous queries, preserving latency for functional queries that don't benefit from personalization.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
│                                                                 │
│  User Query ──┐                                                 │
│               │                                                 │
│  User Profile ─┼──► FT-Transformer ──► User Embedding (128-d)  │
│  (47 features) │         │                    │                 │
│               │         │          ┌──────────┤                 │
│               │         │          ▼          ▼                 │
│               │         │    E2P Projector  Personalization     │
│               │         │    (128→1152-d)    Gate              │
│               │         │          │          │                 │
│               │         │          ▼          ▼                 │
│               │         │    Prefix Token   Personalize?       │
│               │         │          │        Yes / No            │
│               │         │          │          │                 │
│               │         │          ▼          │                 │
│               └─────────┼──► [Prefix ⊕ Query Tokens]           │
│                         │          │                            │
│                         │          ▼                            │
│                         │   Gemma3-1B + LoRA                   │
│                         │          │                            │
│                         │          ▼                            │
│                         │   Personalized Rewrite                │
│                         │   "rewards" → "sapphire reserve       │
│                         │    points balance"                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
personalized_query_rewriter/
├── config/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── data_loader.py           # Load SQL output, build training pairs
│   ├── feature_engineering.py   # User profile → fixed-size feature vector
│   └── dataset.py               # PyTorch datasets for CPT, SFT, GRPO
├── models/
│   ├── ft_transformer.py        # FT-Transformer user encoder (tabular→embedding)
│   ├── e2p_projector.py         # Embedding-to-Prefix projection
│   ├── personalization_gate.py  # WeWrite "When to Write" decision gate
│   ├── personalized_rewriter.py # Full model combining all components
│   └── losses.py                # InfoNCE contrastive loss + GRPO rewards
├── training/
│   ├── train_user_encoder.py    # Stage 0: Contrastive pre-training
│   ├── stage1_cpt.py            # Stage 1: Continual Pre-Training on Chase text
│   ├── stage2_sft.py            # Stage 2: Supervised Fine-Tuning with user context
│   └── stage3_grpo.py           # Stage 3: GRPO alignment with retrieval rewards
├── evaluation/
│   ├── metrics.py               # NDCG, MRR, BLEU, ROUGE-L, CTR, reformulation rate
│   ├── evaluator.py             # Full evaluation pipeline with stratified analysis
│   └── bias_correction.py       # IPW and relevance saturation correction
├── inference/
│   ├── pipeline.py              # Production pipeline with caching and latency tracking
│   └── cache.py                 # Semantic cache for query→rewrite
├── scripts/
│   ├── run_training.py          # Full training orchestrator (all stages)
│   ├── run_eval.py              # Evaluation runner
│   └── run_inference.py         # Inference (interactive, batch, benchmark)
├── tests/
│   ├── test_data.py             # Tests for data loading and features
│   ├── test_models.py           # Tests for model components
│   └── test_training.py         # Tests for metrics, caching, bias correction
├── config/config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## Training Pipeline

### Overview

The training proceeds in 4 stages, each building on the previous:

```
Stage 0: User Encoder Pre-training (Contrastive / InfoNCE)
    └──► Stage 1: Continual Pre-Training (CPT on Chase domain text)
            └──► Stage 2: Supervised Fine-Tuning (SFT with user context)
                    └──► Stage 3: GRPO Alignment (retrieval-based rewards)
```

### Stage 0: User Encoder Pre-training

**Purpose:** Learn user embeddings where users with similar search behavior are close in embedding space.

**Method:** InfoNCE contrastive loss with in-batch negatives.
- Positive pairs: (user_features, clicked_navlink)
- Negatives: Other users' clicks in the same batch

**Trains:** FT-Transformer only

### Stage 1: Continual Pre-Training (CPT)

**Purpose:** Adapt Gemma3-1B to Chase domain vocabulary ("Sapphire" = credit card, not gemstone).

**Method:** Two task prefixes on ~10M CDA utterances:
- `"complete: {partial_query}"` → existing auto-completion task
- `"rewrite: {masked_query}"` → domain reconstruction task

**Trains:** LoRA adapters only (base LLM frozen)

### Stage 2: Supervised Fine-Tuning (SFT)

**Purpose:** Train the full personalization pipeline on supervised rewriting pairs.

**Method:** User features injected via E2P prefix tokens:
- Input: `[E2P_prefix] rewrite: {original_query}` → `{target_rewrite}`
- Training pairs from session reformulations + click-through data

**Trains:** FT-Transformer + E2P + LoRA adapters

### Stage 3: GRPO Alignment

**Purpose:** Align the rewriter with actual search system performance.

**Method:** Group Relative Policy Optimization:
1. Generate G candidate rewrites per query
2. Score each with composite reward (index hit + semantic fidelity + length)
3. Compute group-relative advantages
4. Update with clipped surrogate objective + KL penalty vs SFT checkpoint

**Trains:** All components (FT-Transformer + E2P + Gate + LoRA)

---

## Data Pipeline

### Input Data Format

The system expects the output of the SQL join (see `query.md`):

```csv
ENTP_PRTY_ID,USR_PRFL_ID,ConvID,Query,User Products,Clicked,Selected_Navlink,Click_Sequence,Primary_Predicted_Intent,Predicted_Intent_Full
98765432,ABC123XYZ,123,Increase limit,"Sapphire Reserve, Chase Freedom Unlimited",True,/creditcard/limit,1,credit_limit_increase,credit_limit_increase:0.92
```

### Feature Engineering

The `UserFeatureEncoder` converts the "User Products" comma-separated string into a 55-dimensional feature vector:
- 47 binary product indicators (from `upd_cbds_hhld_prfl_fct`)
- 8 derived aggregate features (premium card holder, business user, digital active, diversity score, etc.)

### Training Data Construction

Three types of training pairs are automatically extracted:

1. **Session reformulation pairs:** Within a conversation, if query_i didn't result in a click but query_{i+1} did, the pair (query_i, query_{i+1}) becomes a training example.

2. **Click-through pairs:** (query, clicked_navlink) pairs with user context.

3. **Ambiguity labels:** Queries flagged as ambiguous via intent entropy analysis (for gate training).

---

## Running the Code

### Installation

```bash
pip install -e ".[full]"
```

### Full Training Pipeline

```bash
python -m scripts.run_training --config config/config.yaml
```

### Individual Stages

```bash
# Stage 0: User encoder
python -m scripts.run_training --config config/config.yaml --stage user_encoder

# Stage 1: CPT (resume from user encoder)
python -m scripts.run_training --config config/config.yaml --stage cpt

# Stage 2: SFT (resume from CPT)
python -m scripts.run_training --config config/config.yaml --stage sft --resume outputs/cpt/best

# Stage 3: GRPO (resume from SFT)
python -m scripts.run_training --config config/config.yaml --stage grpo --resume outputs/sft/best
```

### Evaluation

```bash
python -m scripts.run_eval --model outputs/final --repeats 5
```

### Inference

```bash
# Interactive mode
python -m scripts.run_inference --model outputs/final --interactive

# Single query
python -m scripts.run_inference --model outputs/final \
    --query "rewards" \
    --products "Sapphire Reserve, Mobile Banking"

# Batch mode
python -m scripts.run_inference --model outputs/final \
    --input data/test_queries.csv --output outputs/rewrites.csv

# Latency benchmark
python -m scripts.run_inference --model outputs/final --benchmark
```

### Tests

```bash
pytest tests/ -v
```

---

## Key Design Decisions

### 1. Selective Personalization (WeWrite Gate)

Not all queries benefit from personalization. The gate classifies queries as ambiguous or functional:
- **Functional** ("check balance", "pay credit card"): Pass through unchanged. No latency added.
- **Ambiguous** ("rewards", "fees", "travel"): Routed through personalized rewriter.

This prevents over-personalization (which degrades performance per TREC iKAT findings) and preserves the ~50ms latency budget for the majority of queries.

### 2. FT-Transformer for Tabular Features

Chase's user profile has hundreds of heterogeneous features (binary indicators, counts, ratios). The FT-Transformer treats each feature as a "token" with learned embeddings, then applies self-attention to capture feature interactions. This outperforms naive NL serialization (which would consume hundreds of LLM tokens) and handles mixed feature types natively.

### 3. E2P Prefix Injection

The user embedding is projected into a single soft prefix token in the LLM's hidden space (~100K parameters, <2ms latency). This is more parameter-efficient than cross-attention (USER-LLM) and more expressive than NL serialization (TabLLM). PEPNet-style gating modulates injection strength.

### 4. GRPO over PPO

GRPO eliminates the need for a separate critic network by computing advantages within groups of candidates for the same query. This halves the training compute and is validated by DeepSeekMath and WeWrite.

---

## Latency Budget

Target: <50ms end-to-end on CPU

| Component              | Budget  | Notes                              |
|------------------------|---------|------------------------------------|
| User embedding lookup  | <10ms   | Pre-computed, Redis cached         |
| E2P projection         | <2ms    | ~100K parameter MLP                |
| Personalization gate   | <1ms    | Small MLP, quick decision          |
| LLM inference (LoRA)   | <25ms   | INT8 quantized, speculative decode |
| Network overhead       | <12ms   | gRPC serialization                 |

For non-personalized queries (gate rejects), total overhead is <1ms (gate check only).

---

## Evaluation Framework

### Metrics

| Family              | Metrics                          | Purpose                                |
|---------------------|----------------------------------|----------------------------------------|
| Retrieval Quality   | NDCG@K, MRR@K, Recall@K         | Does the rewrite improve search?       |
| Rewrite Quality     | BLEU, ROUGE-L                    | How close to gold reformulations?      |
| Personalization     | Reformulation rate, CTR, dwell   | Does user behavior improve?            |
| Stratified          | Per-segment breakdown            | Detect domain seesaw effects           |

### Position Bias Correction

Click data has strong position bias. We implement:
- **IPW (Inverse Propensity Weighting):** Power-law propensity model corrects click weights
- **Relevance Saturation:** DualIPW correction for the ≤2 clicks per session problem

### Variance Reporting

Following SIGIR-AP 2025 findings, all evaluations are repeated 5-10 times with variance reported, since LLM generation stochasticity can cause multi-percentage-point differences across identical runs.

---

## References

Key papers informing this architecture:

- **WeWrite (2025):** "When to Write" selective personalization with GRPO
- **E2P (Spotify, RecSys 2025):** Embedding-to-Prefix for <2ms personalization
- **PEPNet (KDD 2023):** Gated personalization at 300M+ users (Kuaishou)
- **FT-Transformer (NeurIPS 2021):** Tabular data encoding via attention
- **RePair My Queries (WISE 2024):** User identity tokens for T5 rewriting
- **GRPO (DeepSeekMath):** Group Relative Policy Optimization
- **CoPPS (KDD 2023):** Contrastive user embedding pre-training
- **R&R (2024):** CPT + SFT for domain-specific query rewriting
- **DLA (SIGIR 2018):** Position bias correction
- **Netflix Search (ACM 2023):** Fetch vs explore mode personalization
