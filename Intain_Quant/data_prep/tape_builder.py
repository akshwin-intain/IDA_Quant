"""
Build static loan tape from monthly performance history.
Copied from original tape.py — only import paths changed:
  from .schema → from core.schema
  from .utils  → from core.utils
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.schema import CES1_TAPE_COLUMNS
from core.utils import require_columns


_COLUMN_ALIASES: Dict[str, str] = {
    # identifiers
    "LoanID": "Loan ID",
    "loan_id": "Loan ID",
    # dates
    "FirstPaymentDate": "First Payment Date",
    "MaturityDate": "Maturity Date",
    "OriginationDate": "Origination Date",
    "Data Cutoff Date": "Data Cut-Off Date",
    "DataCutoffDate": "Data Cut-Off Date",
    "DataCutOffDate": "Data Cut-Off Date",
    # balances/rates/terms
    "OriginalBalance": "Original Principal Balance",
    "Original Principal": "Original Principal Balance",
    "CurrentBalance": "Current Principal Balance",
    "Rate": "Current Interest Rate",
    "OriginalTerm": "Original Term",
    # text fields with common variants
    "Amortization Type": "Amortisation Type",
    # IO flag
    "IO Flag": "IOFlag",
    "IOFLAG": "IOFlag",
    "IO_Flag": "IOFlag",
    "io_flag": "IOFlag",
}

# Columns that are optional — select_tape_columns won't fail if they're missing
_OPTIONAL_COLUMNS: set = {"IOFlag", "Data Cut-Off Date"}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common column name aliases normalized and duplicates coalesced."""
    if df.empty:
        return df.copy()

    ren = {c: _COLUMN_ALIASES.get(c, c) for c in df.columns}
    out = df.rename(columns=ren).copy()

    # If renaming created duplicate column names (e.g. both "Amortization Type" and
    # "Amortisation Type" existed), coalesce duplicates by taking first non-null.
    if out.columns.duplicated().any():
        new_cols: List[str] = []
        parts: List[pd.Series] = []
        seen: set[str] = set()
        cols = list(out.columns)
        for name in cols:
            if name in seen:
                continue
            idxs = [i for i, c in enumerate(cols) if c == name]
            s = out.iloc[:, idxs[0]]
            for j in idxs[1:]:
                s = s.combine_first(out.iloc[:, j])
            new_cols.append(name)
            parts.append(s)
            seen.add(name)
        out = pd.concat(parts, axis=1)
        out.columns = new_cols

    return out


def select_tape_columns(
    df: pd.DataFrame,
    *,
    columns: Tuple[str, ...] = CES1_TAPE_COLUMNS,
) -> pd.DataFrame:
    """Return a copy containing ONLY the tape columns.

    Optional columns (IOFlag, Data Cut-Off Date) are included when present
    but do not cause an error when missing.
    """
    d2 = canonicalize_columns(df)
    required = [c for c in columns if c not in _OPTIONAL_COLUMNS]
    require_columns(d2, required)
    # Include optional columns only when they exist in the data
    present = [c for c in columns if c in d2.columns]
    return d2.loc[:, present].copy()


def parse_month_label(month_series: pd.Series) -> pd.Series:
    """Parse Month values like 'Jun-25' to a month-start Timestamp; NaT where parsing fails."""
    return pd.to_datetime(month_series.astype(str).str.strip(), format="%b-%y", errors="coerce")


def infer_as_of_date(
    perf_history: pd.DataFrame,
    *,
    month_col: str = "Month",
    payment_date_col: str = "Payment Date",
) -> pd.Timestamp:
    """
    Infer a reasonable 'as-of' date from a performance-history dataset.
    Prefers the max Payment Date if available, else max parsed Month label.
    """
    df = canonicalize_columns(perf_history)
    if payment_date_col in df.columns:
        p = pd.to_datetime(df[payment_date_col], errors="coerce")
        if p.notna().any():
            return pd.Timestamp(p.max())
    if month_col in df.columns:
        m = parse_month_label(df[month_col])
        if m.notna().any():
            # Use end-of-month as an as-of convention for month labels
            return pd.Timestamp(m.max()) + pd.offsets.MonthEnd(0)
    raise ValueError("Could not infer as-of date (no usable Payment Date or Month columns).")


def build_static_loan_tape_from_history(
    perf_history: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    *,
    loan_id_col: str = "Loan ID",
    month_col: str = "Month",
    prefer_payment_date_col: Optional[str] = "Payment Date",
    keep_columns: Optional[Tuple[str, ...]] = CES1_TAPE_COLUMNS,
) -> pd.DataFrame:
    """
    Build a one-row-per-loan "static tape" from a monthly performance history file.

    - If `as_of` is None, uses the latest available month per loan.
    - If `as_of` is provided, takes the latest row <= as_of per loan.
    """
    df = canonicalize_columns(perf_history)
    if loan_id_col not in df.columns:
        raise ValueError(f"Missing required column: {loan_id_col!r}")

    df["_month_ts"] = parse_month_label(df[month_col]) if month_col in df.columns else pd.NaT
    if prefer_payment_date_col and prefer_payment_date_col in df.columns:
        pay_dt = pd.to_datetime(df[prefer_payment_date_col], errors="coerce")
        df["_asof_ts"] = pay_dt.where(pay_dt.notna(), df["_month_ts"])
    else:
        df["_asof_ts"] = df["_month_ts"]

    if df["_asof_ts"].isna().all():
        raise ValueError("Could not infer as-of ordering. Provide Month or Payment Date.")

    if as_of is not None:
        as_of = pd.to_datetime(as_of)
        df = df[df["_asof_ts"] <= as_of].copy()
        if df.empty:
            raise ValueError("No rows at or before the provided as_of date.")

    df = df.sort_values([loan_id_col, "_asof_ts"])
    tape = df.groupby(loan_id_col, as_index=False).tail(1).copy()
    tape = tape.drop(columns=[c for c in ["_month_ts", "_asof_ts"] if c in tape.columns])
    tape = tape.reset_index(drop=True)

    if keep_columns is not None:
        tape = select_tape_columns(tape, columns=keep_columns)
    return tape
