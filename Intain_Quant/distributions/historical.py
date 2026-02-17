"""
Compute CPR, CDR, and Loss Severity distributions from actual deal history.

Input:  Raw monthly performance CSVs (665-column deal files)
Output: Mean, std, min, max for each behavioral variable

This is Source A of the 3 distribution sources:
  A. Historical data from deals (this file)
  B. Industry benchmarks (benchmarks.py)
  C. Statistical models (models/ package, future)

The deal CSVs contain one row per loan per month. We:
  1. Sort by loan × month
  2. For each month, count transitions: performing → prepaid, performing → defaulted
  3. Compute monthly SMM → annualize to CPR
  4. Compute monthly MDR → annualize to CDR
  5. For defaulted loans, compute severity from losses vs defaulted balance
  6. Return summary statistics for use in Monte Carlo sampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class HistoricalDistribution:
    """Summary statistics for one behavioral variable computed from deal history."""
    variable: str
    mean: float
    std: float
    min_val: float
    max_val: float
    n_observations: int
    monthly_series: pd.Series  # time series of monthly observations

    def __repr__(self) -> str:
        return (
            f"HistoricalDistribution({self.variable}: "
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}], "
            f"n={self.n_observations})"
        )


def _parse_month(df: pd.DataFrame, month_col: str = "Month") -> pd.Series:
    """Parse 'Jun-25' style month labels to Timestamps."""
    return pd.to_datetime(df[month_col].astype(str).str.strip(), format="%b-%y", errors="coerce")


def _compute_monthly_cpr(
    df: pd.DataFrame,
    *,
    loan_id_col: str = "Loan ID",
    month_col: str = "Month",
    status_col: str = "Account Status",
    pif_col: str = "Principal Payment - PIF",
    begin_bal_col: str = "Beginning Loan Balance",
    prepay_status_col: str = "Prepayment Status",
) -> pd.DataFrame:
    """
    Compute monthly CPR from deal performance data.

    Logic: For each month, identify loans that prepaid in full (PIF > 0 or status
    indicates prepayment). Monthly SMM = prepaid_balance / beginning_pool_balance.
    Annualize: CPR = 1 - (1 - SMM)^12
    """
    data = df.copy()
    data["_month_ts"] = _parse_month(data, month_col)
    data = data.dropna(subset=["_month_ts"])

    # Identify prepayments: PIF > 0 or prepayment status indicates payoff
    data[pif_col] = pd.to_numeric(data.get(pif_col, 0), errors="coerce").fillna(0)
    data[begin_bal_col] = pd.to_numeric(data.get(begin_bal_col, 0), errors="coerce").fillna(0)

    monthly = data.groupby("_month_ts").agg(
        total_beginning_balance=(begin_bal_col, "sum"),
        total_pif=(pif_col, "sum"),
        n_loans=(loan_id_col, "nunique"),
    ).reset_index()

    monthly["smm"] = np.where(
        monthly["total_beginning_balance"] > 0,
        monthly["total_pif"] / monthly["total_beginning_balance"],
        0.0,
    )
    monthly["smm"] = monthly["smm"].clip(0, 1)
    monthly["cpr_annual"] = 1.0 - np.power(1.0 - monthly["smm"], 12)

    return monthly[["_month_ts", "smm", "cpr_annual", "n_loans"]].rename(
        columns={"_month_ts": "month"}
    )


def _compute_monthly_cdr(
    df: pd.DataFrame,
    *,
    loan_id_col: str = "Loan ID",
    month_col: str = "Month",
    status_col: str = "Account Status",
    default_amt_col: str = "Default Amount",
    begin_bal_col: str = "Beginning Loan Balance",
    days_arrears_col: str = "Number Of Days In Arrears",
) -> pd.DataFrame:
    """
    Compute monthly CDR from deal performance data.

    Logic: For each month, identify loans that transitioned to default
    (Default Amount > 0 or status changed to default-like).
    Monthly MDR = defaulted_balance / beginning_pool_balance.
    Annualize: CDR = 1 - (1 - MDR)^12
    """
    data = df.copy()
    data["_month_ts"] = _parse_month(data, month_col)
    data = data.dropna(subset=["_month_ts"])

    data[default_amt_col] = pd.to_numeric(data.get(default_amt_col, 0), errors="coerce").fillna(0)
    data[begin_bal_col] = pd.to_numeric(data.get(begin_bal_col, 0), errors="coerce").fillna(0)

    # Also flag loans with severe delinquency as proxy for defaults
    if days_arrears_col in data.columns:
        data["_days_arr"] = pd.to_numeric(data[days_arrears_col], errors="coerce").fillna(0)
    else:
        data["_days_arr"] = 0

    # A loan is considered to have defaulted this month if Default Amount > 0
    # OR if it transitioned to 90+ days delinquent
    data["_is_default_event"] = (data[default_amt_col] > 0)

    monthly = data.groupby("_month_ts").agg(
        total_beginning_balance=(begin_bal_col, "sum"),
        total_default_amount=(default_amt_col, "sum"),
        n_defaults=("_is_default_event", "sum"),
        n_loans=(loan_id_col, "nunique"),
    ).reset_index()

    monthly["mdr"] = np.where(
        monthly["total_beginning_balance"] > 0,
        monthly["total_default_amount"] / monthly["total_beginning_balance"],
        0.0,
    )
    monthly["mdr"] = monthly["mdr"].clip(0, 1)
    monthly["cdr_annual"] = 1.0 - np.power(1.0 - monthly["mdr"], 12)

    return monthly[["_month_ts", "mdr", "cdr_annual", "n_defaults", "n_loans"]].rename(
        columns={"_month_ts": "month"}
    )


def _compute_severity(
    df: pd.DataFrame,
    *,
    loan_id_col: str = "Loan ID",
    default_amt_col: str = "Default Amount",
    loss_col: str = "Allocated Losses",
    recovery_col: str = "Cumulative Recoveries",
) -> Dict[str, float]:
    """
    Compute loss severity from loans that actually defaulted.

    Severity = Total Losses / Total Defaulted Balance
    Recovery Rate = 1 - Severity

    If no defaults observed, returns NaN (use benchmarks as fallback).
    """
    data = df.copy()
    data[default_amt_col] = pd.to_numeric(data.get(default_amt_col, 0), errors="coerce").fillna(0)
    data[loss_col] = pd.to_numeric(data.get(loss_col, 0), errors="coerce").fillna(0)
    data[recovery_col] = pd.to_numeric(data.get(recovery_col, 0), errors="coerce").fillna(0)

    # Only look at loans with actual defaults
    defaulted = data[data[default_amt_col] > 0].copy()

    if defaulted.empty:
        return {
            "severity_mean": float("nan"),
            "severity_std": float("nan"),
            "n_defaults": 0,
            "total_default_balance": 0.0,
            "total_losses": 0.0,
            "total_recoveries": 0.0,
        }

    # Per-loan severity (for loans that have both default amount and loss data)
    # Take the latest record per loan to get final severity
    latest = defaulted.sort_values(loan_id_col).groupby(loan_id_col).tail(1)

    total_defaults = float(latest[default_amt_col].sum())
    total_losses = float(latest[loss_col].sum())
    total_recoveries = float(latest[recovery_col].sum())

    # Pool-level severity
    pool_severity = total_losses / total_defaults if total_defaults > 0 else float("nan")

    # Per-loan severity for std computation
    latest["_loan_sev"] = np.where(
        latest[default_amt_col] > 0,
        latest[loss_col] / latest[default_amt_col],
        float("nan"),
    )
    loan_sevs = latest["_loan_sev"].dropna()

    return {
        "severity_mean": float(pool_severity),
        "severity_std": float(loan_sevs.std()) if len(loan_sevs) > 1 else 0.10,
        "n_defaults": len(latest),
        "total_default_balance": total_defaults,
        "total_losses": total_losses,
        "total_recoveries": total_recoveries,
    }


def compute_historical_distributions(
    deal_history: pd.DataFrame,
    *,
    deal_name: Optional[str] = None,
) -> Dict[str, HistoricalDistribution]:
    """
    Main entry point: compute CPR, CDR, and Severity distributions from deal history.

    Parameters
    ----------
    deal_history : pd.DataFrame
        Raw monthly performance CSV (665 columns, one row per loan per month)
    deal_name : str, optional
        Label for this deal (e.g. "2024-CES1")

    Returns
    -------
    Dict with keys "cpr", "cdr", "severity" mapping to HistoricalDistribution objects.
    If a variable can't be computed (e.g. no defaults for severity), its mean/std
    will be NaN — caller should fall back to benchmarks.
    """
    label = deal_name or "unknown_deal"
    results: Dict[str, HistoricalDistribution] = {}

    # --- CPR ---
    cpr_monthly = _compute_monthly_cpr(deal_history)
    cpr_series = cpr_monthly["cpr_annual"]
    results["cpr"] = HistoricalDistribution(
        variable=f"CPR ({label})",
        mean=float(cpr_series.mean()) if len(cpr_series) > 0 else float("nan"),
        std=float(cpr_series.std()) if len(cpr_series) > 1 else 0.03,
        min_val=float(cpr_series.min()) if len(cpr_series) > 0 else 0.0,
        max_val=float(cpr_series.max()) if len(cpr_series) > 0 else 0.0,
        n_observations=len(cpr_series),
        monthly_series=cpr_series,
    )

    # --- CDR ---
    cdr_monthly = _compute_monthly_cdr(deal_history)
    cdr_series = cdr_monthly["cdr_annual"]
    results["cdr"] = HistoricalDistribution(
        variable=f"CDR ({label})",
        mean=float(cdr_series.mean()) if len(cdr_series) > 0 else float("nan"),
        std=float(cdr_series.std()) if len(cdr_series) > 1 else 0.015,
        min_val=float(cdr_series.min()) if len(cdr_series) > 0 else 0.0,
        max_val=float(cdr_series.max()) if len(cdr_series) > 0 else 0.0,
        n_observations=len(cdr_series),
        monthly_series=cdr_series,
    )

    # --- Severity ---
    sev_stats = _compute_severity(deal_history)
    # Severity doesn't have a monthly time series in the same way — it's per-default-event
    sev_series = pd.Series([sev_stats["severity_mean"]] if not np.isnan(sev_stats["severity_mean"]) else [])
    results["severity"] = HistoricalDistribution(
        variable=f"Severity ({label})",
        mean=sev_stats["severity_mean"],
        std=sev_stats["severity_std"],
        min_val=max(sev_stats["severity_mean"] - 2 * sev_stats["severity_std"], 0.0)
        if not np.isnan(sev_stats["severity_mean"])
        else 0.0,
        max_val=min(sev_stats["severity_mean"] + 2 * sev_stats["severity_std"], 1.0)
        if not np.isnan(sev_stats["severity_mean"])
        else 1.0,
        n_observations=sev_stats["n_defaults"],
        monthly_series=sev_series,
    )

    return results
