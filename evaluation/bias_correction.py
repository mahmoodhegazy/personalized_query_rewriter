"""
bias_correction.py
==================
Position bias correction for click-derived training labels.

Users exhibit strong position bias: items shown at higher positions receive
more clicks regardless of relevance. This bias must be corrected before
using clicks as relevance labels.

Methods implemented:
    1. Inverse Propensity Weighting (IPW)
       - Classic approach from Joachims et al.
    2. Dual Learning Algorithm (DLA, SIGIR 2018)
       - Jointly learns unbiased ranking and propensity models.
    3. DualIPW correction for relevance saturation bias
       - Users click ≤2 results even when multiple are relevant.

Reference:
    DLA (SIGIR 2018), DualIPW (2025)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InversePropensityWeighting:
    """
    Inverse Propensity Weighting (IPW) for position bias correction.

    Position propensity: P(examine | position) decreases with rank.
    The true relevance weight for a click at position k is:
        w_k = 1 / P(examine | k)

    We estimate propensity from randomized experiments or assume a
    power-law decay model: P(examine | k) = 1 / k^η

    Args:
        method: 'power_law' or 'empirical'.
        eta: Power law exponent (for 'power_law' method).
        empirical_propensities: Dict mapping position → propensity
            (for 'empirical' method, from randomized experiments).
    """

    def __init__(
        self,
        method: str = "power_law",
        eta: float = 1.0,
        empirical_propensities: Optional[Dict[int, float]] = None,
        max_position: int = 20,
    ):
        self.method = method
        self.eta = eta
        self.max_position = max_position

        if method == "empirical" and empirical_propensities:
            self.propensities = empirical_propensities
        else:
            # Power law: P(examine | k) = 1 / k^η
            self.propensities = {
                k: 1.0 / (k ** eta) for k in range(1, max_position + 1)
            }

        # Normalize so max propensity = 1.0
        max_prop = max(self.propensities.values())
        self.propensities = {
            k: v / max_prop for k, v in self.propensities.items()
        }

    def get_weight(self, position: int) -> float:
        """
        Get IPW weight for a click at a given position.

        Args:
            position: 1-indexed position in the result list.

        Returns:
            Weight = 1 / propensity.
        """
        propensity = self.propensities.get(position, 1e-6)
        return 1.0 / max(propensity, 1e-6)

    def correct_click_labels(
        self,
        click_positions: List[int],
        max_weight: float = 10.0,
    ) -> List[float]:
        """
        Convert binary click labels at positions into IPW-corrected weights.

        Args:
            click_positions: List of positions where clicks occurred (1-indexed).
            max_weight: Maximum allowed weight (for clipping).

        Returns:
            List of corrected relevance weights.
        """
        weights = []
        for pos in click_positions:
            w = min(self.get_weight(pos), max_weight)
            weights.append(w)
        return weights

    def correct_training_data(
        self,
        queries: List[str],
        clicked_positions: List[int],
        navlinks: List[str],
    ) -> List[Tuple[str, str, float]]:
        """
        Produce IPW-corrected training triples: (query, navlink, weight).

        Higher-position clicks get lower weight (they would have been
        clicked regardless of relevance). Lower-position clicks get
        higher weight (user had to scroll past more items).

        Args:
            queries: Query texts.
            clicked_positions: Position of clicked result per query.
            navlinks: Clicked navlink per query.

        Returns:
            List of (query, navlink, weight) tuples.
        """
        triples = []
        for query, pos, navlink in zip(queries, clicked_positions, navlinks):
            weight = min(self.get_weight(pos), 10.0)
            triples.append((query, navlink, weight))

        logger.info(
            "IPW correction: %d triples, weight range [%.2f, %.2f]",
            len(triples),
            min(t[2] for t in triples) if triples else 0,
            max(t[2] for t in triples) if triples else 0,
        )

        return triples


class RelevanceSaturationCorrector:
    """
    Correct for relevance saturation bias (DualIPW, 2025).

    Users typically click only 1-2 results even when multiple are relevant,
    causing ~99% of sessions to have ≤2 clicks. This makes unclicked items
    at positions 3+ unreliable as negative examples.

    Solution:
        - For sessions with ≤2 clicks, discount negative labels at
          positions near the last click.
        - Use a "soft negative" weight that decays with distance from
          the last click position.

    Args:
        decay_rate: How quickly negative confidence increases with distance
            from last click position.
        min_confidence: Minimum confidence for a negative label.
    """

    def __init__(
        self,
        decay_rate: float = 0.5,
        min_confidence: float = 0.1,
    ):
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence

    def get_negative_weight(
        self,
        position: int,
        last_click_position: int,
        total_clicks: int,
    ) -> float:
        """
        Compute confidence weight for a negative (unclicked) item.

        Items just below the last click are less reliably negative
        than items far below.

        Args:
            position: Position of the unclicked item.
            last_click_position: Position of the last clicked item.
            total_clicks: Total clicks in the session.

        Returns:
            Weight in [min_confidence, 1.0] for the negative label.
        """
        if total_clicks >= 3:
            # Sessions with 3+ clicks are more reliable
            return 1.0

        distance = position - last_click_position
        if distance <= 0:
            return 1.0  # Items above last click are reliable negatives

        # Exponential decay: items close to last click are unreliable negatives
        confidence = 1.0 - (1.0 - self.min_confidence) * np.exp(
            -self.decay_rate * distance
        )

        return max(confidence, self.min_confidence)
