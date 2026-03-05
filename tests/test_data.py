"""
test_data.py
============
Unit tests for data loading, feature engineering, and dataset classes.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from personalized_query_rewriter.data.data_loader import (
    load_raw_data,
    split_data,
    build_session_reformulation_pairs,
    build_click_through_pairs,
    identify_ambiguous_queries,
)
from personalized_query_rewriter.data.feature_engineering import (
    UserFeatureEncoder,
    CANONICAL_PRODUCTS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV matching the SQL output format."""
    data = {
        "ENTP_PRTY_ID": ["U001", "U001", "U002", "U002", "U003"],
        "USR_PRFL_ID": ["S001", "S001", "S002", "S002", "S003"],
        "ConvID": ["C001", "C001", "C002", "C002", "C003"],
        "Query": [
            "Increase limit",
            "credit limit increase",
            "card activation",
            "activate debit card",
            "rewards",
        ],
        "User Products": [
            "Sapphire Reserve, Chase Freedom Unlimited",
            "Sapphire Reserve, Chase Freedom Unlimited",
            "ATM User, Debit Card",
            "ATM User, Debit Card",
            "Sapphire Reserve, Personal Checking, Ultimate Rewards",
        ],
        "Clicked": ["False", "True", "False", "True", "False"],
        "Selected_Navlink": ["NULL", "/creditcard/limit", "NULL", "/debit/activate", "NULL"],
        "Click_Sequence": ["NULL", "1", "NULL", "1", "NULL"],
        "Primary_Predicted_Intent": [
            "credit_limit_increase",
            "credit_limit_increase",
            "card_activation",
            "card_activation",
            "rewards_balance",
        ],
        "Predicted_Intent_Full": [
            "credit_limit_increase:0.85",
            "credit_limit_increase:0.92",
            "card_activation:0.88",
            "card_activation:0.95",
            "rewards_balance:0.55",
        ],
    }
    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_df(sample_csv):
    return load_raw_data(sample_csv)


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDataLoader:
    def test_load_raw_data(self, sample_csv):
        df = load_raw_data(sample_csv)
        assert len(df) == 5
        assert "query_clean" in df.columns
        assert "clicked_bool" in df.columns
        assert "user_product_list" in df.columns
        assert "product_count" in df.columns
        assert "intent_confidence" in df.columns

    def test_query_cleaning(self, sample_df):
        # Queries should be lowercased and stripped
        assert all(q == q.lower().strip() for q in sample_df["query_clean"])

    def test_clicked_parsing(self, sample_df):
        clicked = sample_df["clicked_bool"].tolist()
        assert clicked == [False, True, False, True, False]

    def test_product_list_parsing(self, sample_df):
        first_products = sample_df.iloc[0]["user_product_list"]
        assert isinstance(first_products, list)
        assert "Sapphire Reserve" in first_products
        assert "Chase Freedom Unlimited" in first_products

    def test_intent_confidence_parsing(self, sample_df):
        # First row: "credit_limit_increase:0.85"
        assert abs(sample_df.iloc[0]["intent_confidence"] - 0.85) < 0.01
        # Last row: "rewards_balance:0.55"
        assert abs(sample_df.iloc[4]["intent_confidence"] - 0.55) < 0.01

    def test_split_data_no_leakage(self, sample_df):
        train, val, test = split_data(sample_df, 0.6, 0.2, 0.2, seed=42)
        train_users = set(train["ENTP_PRTY_ID"])
        val_users = set(val["ENTP_PRTY_ID"])
        test_users = set(test["ENTP_PRTY_ID"])
        # No user overlap between splits
        assert not (train_users & val_users)
        assert not (train_users & test_users)
        assert not (val_users & test_users)

    def test_build_click_through_pairs(self, sample_df):
        pairs = build_click_through_pairs(sample_df)
        # Only clicked rows with valid navlinks
        assert len(pairs) == 2
        assert "query_clean" in pairs.columns
        assert "navlink" in pairs.columns

    def test_identify_ambiguous_queries(self, sample_df):
        df = identify_ambiguous_queries(sample_df)
        assert "is_ambiguous" in df.columns
        assert "intent_entropy" in df.columns


# =============================================================================
# Feature Engineering Tests
# =============================================================================

class TestFeatureEncoder:
    def test_encoder_initialization(self):
        encoder = UserFeatureEncoder(include_derived=True)
        assert encoder.num_base_features == len(CANONICAL_PRODUCTS)
        assert encoder.num_features == len(CANONICAL_PRODUCTS) + 8  # 8 derived

    def test_encode_empty(self):
        encoder = UserFeatureEncoder()
        vec = encoder.encode([])
        assert vec.shape == (encoder.num_features,)
        assert vec.sum() == 0.0

    def test_encode_known_products(self):
        encoder = UserFeatureEncoder()
        vec = encoder.encode(["Sapphire Reserve", "Personal Checking"])
        # Binary indicators should be set
        assert vec[0] == 1.0  # Sapphire Reserve is index 0
        assert vec[23] == 1.0  # Personal Checking is index 23
        # Derived: has_premium should be 1
        assert vec[encoder.num_base_features + 1] == 1.0

    def test_encode_unknown_product(self):
        encoder = UserFeatureEncoder()
        vec = encoder.encode(["Unknown Product"])
        # No binary indicators set
        assert vec[:encoder.num_base_features].sum() == 0.0
        # But total_product_count should be 1
        assert vec[encoder.num_base_features] == 1.0

    def test_encode_batch(self):
        encoder = UserFeatureEncoder()
        batch = encoder.encode_batch([
            ["Sapphire Reserve"],
            ["Personal Checking", "Debit Card"],
        ])
        assert batch.shape == (2, encoder.num_features)

    def test_encode_to_tensor(self):
        encoder = UserFeatureEncoder()
        tensor = encoder.encode_to_tensor(["Sapphire Reserve"])
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_derived_features(self):
        encoder = UserFeatureEncoder(include_derived=True)

        # Business user
        vec = encoder.encode(["Ink Business Card", "Business Checking"])
        offset = encoder.num_base_features
        assert vec[offset + 3] == 1.0  # has_business

        # Lending user
        vec = encoder.encode(["Mortgage", "Auto Loan"])
        assert vec[offset + 4] == 1.0  # has_lending

        # Digital user
        vec = encoder.encode(["Mobile Banking", "Online Banking"])
        assert vec[offset + 6] == 1.0  # is_digital_active

    def test_diversity_score(self):
        encoder = UserFeatureEncoder(include_derived=True)
        offset = encoder.num_base_features

        # User with products in many categories
        vec = encoder.encode([
            "Sapphire Reserve",      # premium
            "United Card",           # cobrand
            "Ink Business Card",     # business
            "Mortgage",              # lending
            "Investment Account",    # investment
            "Mobile Banking",        # digital
        ])
        # All 6 categories → diversity = 1.0
        assert abs(vec[offset + 7] - 1.0) < 0.01


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
