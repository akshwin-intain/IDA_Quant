"""
Base classes for behavioral models.
Extracted from original behavior.py — just the interface, no implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HazardForecast:
    """
    Monthly hazards for each loan for each projection month.

    cpr_annual and cdr_annual are annualized rates in [0,1].
    severity is loss severity in [0,1] (so recovery = (1-severity) * default_principal).
    """

    # Each should be shape (n_loans, n_months)
    cpr_annual: np.ndarray
    cdr_annual: np.ndarray
    severity: np.ndarray


class BehaviorModel:
    """Interface for generating monthly hazards (can be ML/statistical)."""

    def forecast(
        self,
        tape: pd.DataFrame,
        projection_months: int,
        *,
        as_of_date: pd.Timestamp,
    ) -> HazardForecast:
        raise NotImplementedError
