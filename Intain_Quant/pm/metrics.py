"""
Per-path PM metric computation.

Computes WAL, Expected Life, and Cumulative Loss for EACH path individually.
This is used by the aggregator to build distribution summaries.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.utils import require_columns


def compute_path_metrics(
    cashflows: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    original_balance: float,
    date_col: str = "date",
    path_col: str = "path_id",
) -> pd.DataFrame:
    """
    Compute WAL, Cumulative Loss %, and Expected Life for each path.

    Parameters
    ----------
    cashflows : pd.DataFrame
        Engine output with columns: path_id, date, principal, loss, recovery, etc.
    as_of_date : pd.Timestamp
        Projection start date
    original_balance : float
        Total original pool balance (denominator for loss %)

    Returns
    -------
    DataFrame with one row per path:
        path_id, wal_years, cum_loss_pct, cum_loss_abs, expected_life_years, total_principal
    """
    require_columns(cashflows, [path_col, date_col, "principal", "loss", "recovery"])
    df = cashflows.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    as_of = pd.Timestamp(as_of_date)

    # Compute t_years for each date
    unique_dates = df[date_col].unique()
    date_to_years = {d: max((pd.Timestamp(d) - as_of).days / 365.25, 0.0) for d in unique_dates}
    df["_t_years"] = df[date_col].map(date_to_years)

    # Group by path
    results = []
    for path_id, grp in df.groupby(path_col):
        total_principal = grp["principal"].sum()
        total_loss = grp["loss"].sum()
        total_recovery = grp["recovery"].sum()

        # WAL = Σ(principal_t × t) / Σ(principal_t)
        wal = (
            (grp["principal"] * grp["_t_years"]).sum() / total_principal
            if total_principal > 0 else 0.0
        )

        # Expected Life = Σ((principal_t + recovery_t) × t) / Σ(principal_t + recovery_t)
        total_returned = grp["principal"].values + grp["recovery"].values
        total_returned_sum = total_returned.sum()
        expected_life = (
            (total_returned * grp["_t_years"].values).sum() / total_returned_sum
            if total_returned_sum > 0 else 0.0
        )

        # Cumulative Loss %
        cum_loss_pct = total_loss / original_balance if original_balance > 0 else 0.0

        results.append({
            "path_id": path_id,
            "wal_years": float(wal),
            "cum_loss_pct": float(cum_loss_pct),
            "cum_loss_abs": float(total_loss),
            "expected_life_years": float(expected_life),
            "total_principal": float(total_principal),
            "total_recovery": float(total_recovery),
        })

    return pd.DataFrame(results)
