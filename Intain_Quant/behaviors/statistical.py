"""
StatisticalHazardModel — loan-level differentiation based on borrower/collateral attributes.

PHASE 2: Not yet implemented. This is where Siddhartha's 3 statistical models connect.

Instead of giving every loan the same CPR/CDR within a path, this model will:
  - Use FICO to differentiate default probability (low FICO → higher CDR for that loan)
  - Use LTV to differentiate severity (high LTV → higher severity for that loan)
  - Use DTI to further adjust default probability
  - Use Property State for geographic risk adjustment

The per-path sampled CPR/CDR/Severity from Monte Carlo become the POOL-LEVEL CENTER,
and this model distributes risk around that center based on loan characteristics.

Example:
  Path 42 sampled CDR = 4% (pool level)
  Loan A: FICO=750, LTV=60% → CDR_A = 1.5%  (much lower than pool)
  Loan B: FICO=620, LTV=95% → CDR_B = 9.1%  (much higher than pool)
  Pool average still ≈ 4%, but risk is allocated to where it belongs.

Dependencies: models/cpr_model.py, models/cdr_model.py, models/severity_model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BehaviorModel, HazardForecast


@dataclass(frozen=True)
class StatisticalHazardModel(BehaviorModel):
    """
    Placeholder for loan-level statistical behavioral model.

    Will be implemented once the 3 statistical models (CPR, CDR, Severity)
    are trained on the deal history data.
    """

    # Pool-level assumptions (from Monte Carlo path)
    cpr_annual: float = 0.10
    cdr_annual: float = 0.02
    severity: float = 0.35

    # Model artifacts (to be loaded from trained models)
    # cpr_model_path: Optional[str] = None
    # cdr_model_path: Optional[str] = None
    # severity_model_path: Optional[str] = None

    def forecast(
        self,
        tape: pd.DataFrame,
        projection_months: int,
        *,
        as_of_date: pd.Timestamp,
    ) -> HazardForecast:
        raise NotImplementedError(
            "StatisticalHazardModel is not yet implemented. "
            "Use ScenarioHazardModel for pool-level assumptions, "
            "or ConstantHazardModel for baseline."
        )
