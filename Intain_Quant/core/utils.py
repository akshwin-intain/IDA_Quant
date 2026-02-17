from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def annual_to_monthly_hazard(annual_rate: np.ndarray) -> np.ndarray:
    """Convert annualized hazard to a simple monthly probability via 1-(1-r)^(1/12)."""
    annual_rate = np.clip(annual_rate, 0.0, 1.0)
    return 1.0 - np.power((1.0 - annual_rate), 1.0 / 12.0)


def month_starts(as_of_date: pd.Timestamp, n_months: int) -> pd.DatetimeIndex:
    """
    Generate month-start dates for projection periods after as_of_date.
    If as_of_date is mid-month, we still project starting next month-start.
    """
    as_of = pd.Timestamp(as_of_date)
    first = (as_of.to_period("M") + 1).to_timestamp(how="start")
    return pd.date_range(first, periods=n_months, freq="MS")


def excel_round(x, decimals: int = 2):
    """Excel ROUND: half away from zero (vectorized)."""
    m = 10 ** decimals
    x = np.asarray(x, dtype=float)
    return np.sign(x) * (np.floor(np.abs(x) * m + 0.5) / m)


def datedif_months_excel(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Excel DATEDIF(start, end, "m") — complete months between two dates."""
    if pd.isna(start) or pd.isna(end):
        return np.nan
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    months = (e.year - s.year) * 12 + (e.month - s.month)
    if e.day < s.day:
        months -= 1
    return int(months)


def period_dates_25th(global_cutoff: pd.Timestamp, n_periods: int) -> list:
    """Generate 25th-of-month projection dates starting from the cutoff month."""
    base_month = pd.Timestamp(year=global_cutoff.year, month=global_cutoff.month, day=1)
    base = base_month + pd.Timedelta(days=24)  # 25th
    dates = [base]
    for k in range(1, n_periods + 1):
        dates.append(base + relativedelta(months=k))
    return dates


