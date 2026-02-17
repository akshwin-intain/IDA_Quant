"""
ScenarioHazardModel — accepts per-path sampled assumptions from Monte Carlo.

This is the REPLACEMENT for ConstantHazardModel in the corrected flow.

ConstantHazardModel: same CPR/CDR/Severity for ALL paths (old, wrong)
ScenarioHazardModel: DIFFERENT CPR/CDR/Severity for EACH path (new, correct)

The runner calls forecast() once per path, passing that path's sampled assumptions.
The model fills in (n_loans × n_months) hazard arrays using those assumptions.

For now, this model applies the same rate to all loans within a path (pool-level).
The StatisticalHazardModel (Phase 2) will differentiate by loan characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BehaviorModel, HazardForecast


@dataclass(frozen=True)
class ScenarioHazardModel(BehaviorModel):
    """
    Behavioral model that uses per-path assumptions from Monte Carlo sampling.

    Parameters are set per-path by the runner before calling forecast().
    Within a path, all loans get the same CPR/CDR/Severity (pool-level model).

    Usage by runner.py:
        for p in range(n_paths):
            path_assumptions = sampled_paths.get_path(p)
            model = ScenarioHazardModel(
                cpr_annual=path_assumptions["cpr"],
                cdr_annual=path_assumptions["cdr"],
                severity=path_assumptions["severity"],
            )
            hazards = model.forecast(tape, horizon, as_of_date=cfg.as_of_date)
            # ... run cashflow engine with these hazards
    """

    cpr_annual: float = 0.10
    cdr_annual: float = 0.02
    severity: float = 0.35

    def forecast(
        self,
        tape: pd.DataFrame,
        projection_months: int,
        *,
        as_of_date: pd.Timestamp,
    ) -> HazardForecast:
        """
        Generate hazard arrays for all loans × all months.

        Same structure as ConstantHazardModel, but the values come from
        Monte Carlo sampling rather than hardcoded inputs.

        Returns
        -------
        HazardForecast with shape (n_loans, projection_months) for each array.
        """
        n = len(tape)
        cpr = np.full((n, projection_months), float(self.cpr_annual), dtype=float)
        cdr = np.full((n, projection_months), float(self.cdr_annual), dtype=float)
        sev = np.full((n, projection_months), float(self.severity), dtype=float)
        return HazardForecast(cpr_annual=cpr, cdr_annual=cdr, severity=sev)


@dataclass(frozen=True)
class SeasonedScenarioHazardModel(BehaviorModel):
    """
    Extension of ScenarioHazardModel that applies a seasoning ramp.

    Industry convention: new loans default/prepay at lower rates than
    seasoned loans. The PSA (Public Securities Association) ramp increases
    CPR linearly from 0 to base over the first 30 months.

    Similar CDR ramp: defaults ramp up over first 24-36 months as
    borrowers who will default start missing payments.

    This is still a pool-level model (same for all loans at same age).
    """

    cpr_annual: float = 0.10
    cdr_annual: float = 0.02
    severity: float = 0.35
    cpr_ramp_months: int = 30    # PSA standard: 30 months to full CPR
    cdr_ramp_months: int = 24    # Default ramp: 24 months to full CDR

    def forecast(
        self,
        tape: pd.DataFrame,
        projection_months: int,
        *,
        as_of_date: pd.Timestamp,
    ) -> HazardForecast:
        n = len(tape)

        # Build seasoning ramps
        months = np.arange(projection_months, dtype=float)

        # CPR ramp: linear from 0 to full over cpr_ramp_months
        cpr_ramp = np.minimum(months / max(self.cpr_ramp_months, 1), 1.0)
        cpr_monthly = float(self.cpr_annual) * cpr_ramp  # (projection_months,)

        # CDR ramp: linear from 0 to full over cdr_ramp_months
        cdr_ramp = np.minimum(months / max(self.cdr_ramp_months, 1), 1.0)
        cdr_monthly = float(self.cdr_annual) * cdr_ramp

        # Broadcast to (n_loans, projection_months)
        cpr = np.broadcast_to(cpr_monthly[np.newaxis, :], (n, projection_months)).copy()
        cdr = np.broadcast_to(cdr_monthly[np.newaxis, :], (n, projection_months)).copy()
        sev = np.full((n, projection_months), float(self.severity), dtype=float)

        return HazardForecast(cpr_annual=cpr, cdr_annual=cdr, severity=sev)
