"""
feature_engineering.py
======================
Convert the comma-separated 'User Products' string into a fixed-size binary
feature vector suitable for the FT-Transformer user encoder.

The user profile from upd_cbds_hhld_prfl_fct contains hundreds of columns,
but via the SQL join we surface the most relevant product indicators as a
comma-separated string. This module converts that into a dense tensor.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Canonical product list (47 features matching config.yaml)
# Order matters — this defines the feature vector index mapping.
# -------------------------------------------------------------------------
CANONICAL_PRODUCTS = [
    "Sapphire Reserve",
    "Sapphire Preferred",
    "Chase Freedom",
    "Chase Freedom Unlimited",
    "Amazon Prime Card",
    "Amazon Card",
    "Southwest Card",
    "Southwest Business Card",
    "United Card",
    "United Business Card",
    "Marriott Card",
    "IHG Card",
    "Hyatt Card",
    "AARP Card",
    "Disney Card",
    "British Airways Card",
    "Starbucks Card",
    "World of Hyatt Card",
    "Ink Business Card",
    "Ink Business Points Card",
    "Business Credit Card",
    "Commercial Credit Card",
    "Personal Credit Card",
    "Personal Checking",
    "Personal Savings",
    "Chase Total Checking",
    "Chase Secure Checking",
    "Student Checking",
    "High School Checking",
    "Chase Private Client",
    "Certificate of Deposit",
    "Business Checking",
    "Business Savings",
    "Business Credit Line",
    "Business Trust",
    "Mortgage",
    "Home Equity",
    "Auto Loan",
    "Auto Lease",
    "Student Loan",
    "Personal Line of Credit",
    "Investment Account",
    "Personal Trust Account",
    "Debit Card",
    "Ultimate Rewards",
    "Mobile Banking",
    "Online Banking",
    "ATM User",
]

# Higher-order derived features (computed from binary indicators)
DERIVED_FEATURE_NAMES = [
    "total_product_count",
    "has_premium_card",        # Sapphire Reserve or Preferred
    "has_cobrand_card",        # Any co-branded airline/hotel card
    "has_business_product",    # Any business product
    "has_lending_product",     # Mortgage, Home Equity, Auto, Student, PLOC
    "has_investment",          # Investment or Trust
    "is_digital_active",       # Mobile or Online banking
    "product_diversity_score", # Ratio of product categories held
]

# Product category groupings for derived features
PREMIUM_CARDS = {"Sapphire Reserve", "Sapphire Preferred", "Chase Private Client"}
COBRAND_CARDS = {
    "Southwest Card", "Southwest Business Card", "United Card",
    "United Business Card", "Marriott Card", "IHG Card", "Hyatt Card",
    "British Airways Card", "Disney Card", "Starbucks Card", "World of Hyatt Card",
}
BUSINESS_PRODUCTS = {
    "Ink Business Card", "Ink Business Points Card", "Business Credit Card",
    "Commercial Credit Card", "Business Checking", "Business Savings",
    "Business Credit Line", "Business Trust", "Southwest Business Card",
    "United Business Card",
}
LENDING_PRODUCTS = {
    "Mortgage", "Home Equity", "Auto Loan", "Auto Lease",
    "Student Loan", "Personal Line of Credit",
}
INVESTMENT_PRODUCTS = {"Investment Account", "Personal Trust Account"}
DIGITAL_PRODUCTS = {"Mobile Banking", "Online Banking"}

# Total number of product categories for diversity score
NUM_CATEGORIES = 6  # premium, cobrand, business, lending, investment, digital


class UserFeatureEncoder:
    """
    Encode a user's product list into a fixed-size feature vector.

    Output vector structure:
        [0..46]   : Binary product indicators (47 features)
        [47..54]  : Derived aggregate features (8 features)
        Total     : 55 features

    Usage:
        encoder = UserFeatureEncoder()
        vec = encoder.encode(["Sapphire Reserve", "Personal Checking", "Mobile Banking"])
        # vec is a numpy array of shape (55,)
    """

    def __init__(self, include_derived: bool = True):
        self.include_derived = include_derived
        self.product_to_idx: Dict[str, int] = {
            p: i for i, p in enumerate(CANONICAL_PRODUCTS)
        }
        self.num_base_features = len(CANONICAL_PRODUCTS)
        self.num_derived_features = len(DERIVED_FEATURE_NAMES) if include_derived else 0
        self.num_features = self.num_base_features + self.num_derived_features

        logger.info(
            "UserFeatureEncoder: %d base + %d derived = %d total features",
            self.num_base_features, self.num_derived_features, self.num_features,
        )

    def encode(self, product_list: List[str]) -> np.ndarray:
        """
        Encode a list of product names into a fixed-size feature vector.

        Args:
            product_list: List of product name strings.

        Returns:
            np.ndarray of shape (num_features,) with float32 values.
        """
        vec = np.zeros(self.num_features, dtype=np.float32)

        # Binary indicators
        product_set = set(product_list)
        for product in product_list:
            idx = self.product_to_idx.get(product)
            if idx is not None:
                vec[idx] = 1.0

        # Derived features
        if self.include_derived:
            offset = self.num_base_features
            vec[offset + 0] = len(product_list)                              # total count
            vec[offset + 1] = float(bool(product_set & PREMIUM_CARDS))       # has premium
            vec[offset + 2] = float(bool(product_set & COBRAND_CARDS))       # has cobrand
            vec[offset + 3] = float(bool(product_set & BUSINESS_PRODUCTS))   # has business
            vec[offset + 4] = float(bool(product_set & LENDING_PRODUCTS))    # has lending
            vec[offset + 5] = float(bool(product_set & INVESTMENT_PRODUCTS)) # has investment
            vec[offset + 6] = float(bool(product_set & DIGITAL_PRODUCTS))    # is digital
            # Diversity = fraction of categories present
            categories_present = sum([
                bool(product_set & PREMIUM_CARDS),
                bool(product_set & COBRAND_CARDS),
                bool(product_set & BUSINESS_PRODUCTS),
                bool(product_set & LENDING_PRODUCTS),
                bool(product_set & INVESTMENT_PRODUCTS),
                bool(product_set & DIGITAL_PRODUCTS),
            ])
            vec[offset + 7] = categories_present / NUM_CATEGORIES

        return vec

    def encode_batch(self, product_lists: List[List[str]]) -> np.ndarray:
        """Encode a batch of product lists. Returns shape (batch, num_features)."""
        return np.stack([self.encode(pl) for pl in product_lists])

    def encode_to_tensor(self, product_list: List[str]) -> torch.Tensor:
        """Encode and return as a PyTorch tensor."""
        return torch.from_numpy(self.encode(product_list))

    def encode_batch_to_tensor(self, product_lists: List[List[str]]) -> torch.Tensor:
        """Encode batch and return as a PyTorch tensor."""
        return torch.from_numpy(self.encode_batch(product_lists))

    @property
    def feature_names(self) -> List[str]:
        """Return ordered list of all feature names."""
        names = list(CANONICAL_PRODUCTS)
        if self.include_derived:
            names.extend(DERIVED_FEATURE_NAMES)
        return names
