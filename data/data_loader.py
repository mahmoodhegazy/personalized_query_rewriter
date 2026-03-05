"""
data_loader.py
==============
Load the SQL-joined query+user-profile CSV and prepare train/val/test splits.

Expected input CSV columns (from the SQL join in query.md):
    ENTP_PRTY_ID          - Enterprise Party ID (unique customer)
    USR_PRFL_ID           - User Profile ID (session/device)
    ConvID                - Conversation ID
    Query                 - User search query text
    User Products         - Comma-separated list of Chase products
    Clicked               - True/False whether user clicked a result
    Selected_Navlink      - Deeplink destination (NULL if no click)
    Click_Sequence        - Click order within conversation
    Primary_Predicted_Intent   - Top NLU predicted intent
    Predicted_Intent_Full      - Full ranked intent list (e.g. "intent:0.92")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Column name constants
# -------------------------------------------------------------------------
COL_PARTY_ID = "ENTP_PRTY_ID"
COL_USER_PROFILE_ID = "USR_PRFL_ID"
COL_CONV_ID = "ConvID"
COL_QUERY = "Query"
COL_USER_PRODUCTS = "User Products"
COL_CLICKED = "Clicked"
COL_NAVLINK = "Selected_Navlink"
COL_CLICK_SEQ = "Click_Sequence"
COL_PRIMARY_INTENT = "Primary_Predicted_Intent"
COL_INTENT_FULL = "Predicted_Intent_Full"


def load_raw_data(
    csv_path: str,
    min_query_length: int = 3,
) -> pd.DataFrame:
    """
    Load the SQL output CSV and apply basic cleaning.

    Steps:
        1. Read CSV, handle NULL strings.
        2. Normalize query text (lowercase, strip).
        3. Parse 'Clicked' to boolean.
        4. Filter queries shorter than min_query_length.
        5. Parse the comma-separated 'User Products' into a list.
        6. Extract intent confidence from Predicted_Intent_Full.

    Returns:
        Cleaned DataFrame with additional derived columns:
            - query_clean        : lowercased, stripped query
            - user_product_list  : Python list of product strings
            - product_count      : number of products
            - intent_confidence  : float confidence of primary intent
            - clicked_bool       : boolean clicked flag
    """
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    # --- Basic cleaning ---
    df[COL_QUERY] = df[COL_QUERY].fillna("").astype(str)
    df["query_clean"] = df[COL_QUERY].str.strip().str.lower()

    # Parse clicked
    df["clicked_bool"] = df[COL_CLICKED].str.strip().str.lower() == "true"

    # Filter short queries
    initial_len = len(df)
    df = df[df["query_clean"].str.len() >= min_query_length].copy()
    logger.info("Filtered %d → %d rows (min query length=%d)",
                initial_len, len(df), min_query_length)

    # Parse user products into list
    df["user_product_list"] = (
        df[COL_USER_PRODUCTS]
        .fillna("")
        .apply(lambda x: [p.strip() for p in x.split(",") if p.strip()])
    )
    df["product_count"] = df["user_product_list"].apply(len)

    # Parse intent confidence from "intent_name:0.92" format
    df["intent_confidence"] = df[COL_INTENT_FULL].apply(_parse_top_confidence)

    # Replace NULL navlinks with None
    df[COL_NAVLINK] = df[COL_NAVLINK].replace({"NULL": None, "null": None, "": None})

    logger.info("Loaded %d records, %d unique users, %d unique queries",
                len(df),
                df[COL_PARTY_ID].nunique(),
                df["query_clean"].nunique())

    return df


def _parse_top_confidence(intent_str: str) -> float:
    """Extract the confidence score from 'intent_name:0.92' format."""
    if pd.isna(intent_str) or not isinstance(intent_str, str):
        return 0.0
    parts = intent_str.strip().split(":")
    if len(parts) >= 2:
        try:
            return float(parts[-1])
        except ValueError:
            return 0.0
    return 0.0


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_col: Optional[str] = "clicked_bool",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by user (no user leakage).

    We split at the *user level* (ENTP_PRTY_ID) to prevent data leakage:
    all queries from a user go into the same split.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    users = df[COL_PARTY_ID].unique()
    logger.info("Splitting %d users into train/val/test (%.0f/%.0f/%.0f%%)",
                len(users), train_ratio*100, val_ratio*100, test_ratio*100)

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_users, valtest_users = train_test_split(
        users, test_size=val_test_ratio, random_state=seed
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    val_users, test_users = train_test_split(
        valtest_users, test_size=relative_test, random_state=seed
    )

    train_df = df[df[COL_PARTY_ID].isin(set(train_users))].copy()
    val_df = df[df[COL_PARTY_ID].isin(set(val_users))].copy()
    test_df = df[df[COL_PARTY_ID].isin(set(test_users))].copy()

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(train_df), len(val_df), len(test_df))

    return train_df, val_df, test_df


def build_session_reformulation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract session reformulation pairs for SFT training data construction.

    Following WeWrite's posterior-based mining: within a conversation, if
    consecutive queries show intent refinement (user reformulated), the
    second query is the "gold" rewrite target.

    Logic:
        - Group by ConvID, order by Click_Sequence (or row order).
        - If query_i did NOT result in a click but query_{i+1} DID,
          treat (query_i, user_products, query_{i+1}) as a training pair.
        - Also extract pairs where the second query is more specific
          (longer, shares tokens with first).

    Returns:
        DataFrame with columns:
            - original_query   : the ambiguous query
            - rewritten_query  : the user's reformulation (target)
            - user_products    : user product list
            - entp_prty_id     : user ID
            - navlink          : the clicked navlink (if any)
    """
    pairs = []

    for conv_id, group in df.groupby(COL_CONV_ID):
        group = group.sort_values(COL_CLICK_SEQ, na_position="last")
        rows = group.to_dict("records")

        for i in range(len(rows) - 1):
            curr = rows[i]
            nxt = rows[i + 1]

            # Case 1: Current didn't click, next did → reformulation
            if not curr["clicked_bool"] and nxt["clicked_bool"]:
                pairs.append({
                    "original_query": curr["query_clean"],
                    "rewritten_query": nxt["query_clean"],
                    "user_product_list": curr["user_product_list"],
                    COL_PARTY_ID: curr[COL_PARTY_ID],
                    "navlink": nxt[COL_NAVLINK],
                    "intent": nxt[COL_PRIMARY_INTENT],
                })

            # Case 2: Both clicked, but next is more specific
            elif (curr["clicked_bool"] and nxt["clicked_bool"]
                  and len(nxt["query_clean"]) > len(curr["query_clean"])):
                pairs.append({
                    "original_query": curr["query_clean"],
                    "rewritten_query": nxt["query_clean"],
                    "user_product_list": curr["user_product_list"],
                    COL_PARTY_ID: curr[COL_PARTY_ID],
                    "navlink": nxt[COL_NAVLINK],
                    "intent": nxt[COL_PRIMARY_INTENT],
                })

    pairs_df = pd.DataFrame(pairs)
    logger.info("Built %d session reformulation pairs from %d conversations",
                len(pairs_df), df[COL_CONV_ID].nunique())
    return pairs_df


def build_click_through_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (query, user_products, navlink) click-through pairs for training.

    These are direct signal pairs: query + user context → clicked destination.
    Used for both SFT target construction and reward computation in GRPO.

    Returns:
        DataFrame with: query_clean, user_product_list, navlink, intent, entp_prty_id
    """
    clicked = df[df["clicked_bool"] & df[COL_NAVLINK].notna()].copy()

    pairs = clicked[[
        "query_clean", "user_product_list",
        COL_NAVLINK, COL_PRIMARY_INTENT, COL_PARTY_ID
    ]].rename(columns={
        COL_NAVLINK: "navlink",
        COL_PRIMARY_INTENT: "intent",
        COL_PARTY_ID: "entp_prty_id",
    })

    logger.info("Built %d click-through pairs", len(pairs))
    return pairs


def identify_ambiguous_queries(
    df: pd.DataFrame,
    intent_entropy_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Identify queries that are 'ambiguous' and would benefit from personalization.

    Following WeWrite's "When to Write" principle: only personalize queries
    where user context can disambiguate intent.

    Heuristics for ambiguity:
        1. Same query text maps to different intents across users.
        2. Same query text has low click-through rate.
        3. Query is short (≤ 3 tokens) and generic.

    Returns:
        DataFrame with additional column 'is_ambiguous' (bool).
    """
    df = df.copy()

    # Metric 1: Intent entropy per query
    intent_counts = (
        df.groupby("query_clean")[COL_PRIMARY_INTENT]
        .apply(lambda x: x.value_counts(normalize=True))
        .reset_index(name="prob")
    )

    # Compute per-query entropy
    def _entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    query_entropy = (
        df.groupby("query_clean")[COL_PRIMARY_INTENT]
        .apply(lambda x: _entropy(x.value_counts(normalize=True).values))
        .reset_index(name="intent_entropy")
    )

    df = df.merge(query_entropy, on="query_clean", how="left")

    # Metric 2: Click-through rate per query
    query_ctr = (
        df.groupby("query_clean")["clicked_bool"]
        .mean()
        .reset_index(name="query_ctr")
    )
    df = df.merge(query_ctr, on="query_clean", how="left")

    # Metric 3: Token count
    df["token_count"] = df["query_clean"].str.split().str.len()

    # Combine: ambiguous if high entropy OR low CTR with short query
    df["is_ambiguous"] = (
        (df["intent_entropy"] > intent_entropy_threshold)
        | ((df["query_ctr"] < 0.3) & (df["token_count"] <= 3))
    )

    n_ambiguous = df["is_ambiguous"].sum()
    logger.info("Identified %d/%d (%.1f%%) ambiguous queries",
                n_ambiguous, len(df), 100 * n_ambiguous / max(len(df), 1))

    return df
