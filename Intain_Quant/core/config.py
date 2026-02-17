"""
Projection configuration.
Copied from original config.py — no changes.
Distribution parameters live in distributions/sampler.py (DistributionParams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd


@dataclass(frozen=True)
class ProjectionConfig:
    as_of_date: pd.Timestamp
    projection_months: int = 360
    n_paths: int = 200
    seed: int = 7

    # default / recovery modeling
    recovery_lag_months: int = 6

    # event priority if both happen in same month (rare in practice)
    event_priority: Tuple[Literal["default", "prepay"], Literal["default", "prepay"]] = (
        "default",
        "prepay",
    )

    # output size controls
    store_loan_level: bool = False  # portfolio-level by default (keeps outputs manageable)
