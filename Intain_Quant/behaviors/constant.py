"""
ConstantHazardModel — baseline model with same CPR/CDR/Severity for all loans/time.
Extracted from original behavior.py. Only change: import path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BehaviorModel, HazardForecast


@dataclass(frozen=True)
class ConstantHazardModel(BehaviorModel):
    """
    Baseline model: constant annual CPR/CDR and severity for all loans/time.
    Useful to validate the plumbing before plugging in ML models.
    """

    cpr_annual: float = 0.08
    cdr_annual: float = 0.02
    severity: float = 0.35

    def forecast(
        self, tape: pd.DataFrame, projection_months: int, *, as_of_date: pd.Timestamp
    ) -> HazardForecast:
        n = len(tape)
        cpr = np.full((n, projection_months), float(self.cpr_annual), dtype=float)
        cdr = np.full((n, projection_months), float(self.cdr_annual), dtype=float)
        sev = np.full((n, projection_months), float(self.severity), dtype=float)
        return HazardForecast(cpr_annual=cpr, cdr_annual=cdr, severity=sev)
