"""
Deterministic cashflow computation helpers — Excel-matched logic.

Key design principles (matching the Excel Deal CE spreadsheet):
  1. Full precision per-loan for PI, Interest, CE (Principal)
  2. excel_round() only at pool-level SUM
  3. Loan age = datedif_months_excel(FPD, global_cutoff)
  4. Remaining term = Original Term - age
  5. IO flag: if "Y", principal contribution = 0
  6. Dates on 25th of each month
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.schema import CES1_TAPE_COLUMNS
from core.utils import (
    require_columns,
    excel_round,
    datedif_months_excel,
    period_dates_25th,
)
from data_prep.tape_builder import select_tape_columns


def level_payment(balance: float, monthly_rate: float, n_months: int) -> float:
    """Standard fully-amortizing level payment (PMT) with near-zero rate guard."""
    if n_months <= 0:
        return float(balance)
    if abs(monthly_rate) < 1e-12:
        return float(balance) / n_months
    return float(balance) * (monthly_rate * (1 + monthly_rate) ** n_months) / (
        (1 + monthly_rate) ** n_months - 1
    )


def prepare_tape_for_engine(
    tape: pd.DataFrame,
    global_cutoff_override: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Coerce key fields, compute loan age via datedif_months_excel, remaining term,
    full-precision PMT per loan, and IO flag.  Returns enriched tape.
    """
    df = select_tape_columns(tape, columns=CES1_TAPE_COLUMNS)
    require_columns(
        df,
        [
            "Loan ID",
            "First Payment Date",
            "Original Term",
            "Current Principal Balance",
            "Current Interest Rate",
        ],
    )

    out = df.copy()
    out["Loan ID"] = out["Loan ID"].astype(str)

    for dcol in ["Origination Date", "First Payment Date", "Maturity Date"]:
        if dcol in out.columns:
            out[dcol] = pd.to_datetime(out[dcol], errors="coerce")

    if "Data Cut-Off Date" in out.columns:
        out["Data Cut-Off Date"] = pd.to_datetime(out["Data Cut-Off Date"], errors="coerce")

    for ncol in [
        "Original Term",
        "Original Principal Balance",
        "Current Principal Balance",
        "Current Interest Rate",
    ]:
        out[ncol] = pd.to_numeric(out[ncol], errors="coerce")

    out["Current Interest Rate"] = out["Current Interest Rate"].fillna(0.0)
    out["Current Principal Balance"] = out["Current Principal Balance"].fillna(
        out["Original Principal Balance"]
    )
    out["Original Term"] = out["Original Term"].fillna(0).astype(float)

    if "Amortisation Type" not in out.columns:
        out["Amortisation Type"] = "OTHR"
    out["Amortisation Type"] = out["Amortisation Type"].astype(str).fillna("OTHR")

    # IO flag
    if "IOFlag" in out.columns:
        out["IOFlag"] = out["IOFlag"].astype(str).str.strip().str.upper()
        out["IOFlag"] = out["IOFlag"].where(out["IOFlag"] == "Y", "N")
    else:
        out["IOFlag"] = "N"

    # Global cutoff
    if global_cutoff_override is not None and not pd.isna(global_cutoff_override):
        global_cutoff = pd.Timestamp(global_cutoff_override).normalize()
    elif "Data Cut-Off Date" in out.columns and out["Data Cut-Off Date"].notna().any():
        global_cutoff = pd.Timestamp(out["Data Cut-Off Date"].max()).normalize()
    else:
        global_cutoff = None

    if global_cutoff is not None:
        # Loan age via Excel DATEDIF
        out["_loan_age"] = out["First Payment Date"].apply(
            lambda fpd: datedif_months_excel(fpd, global_cutoff)
            if pd.notna(fpd) else np.nan
        )

        # Fill missing _loan_age from other available date columns
        missing_age = out["_loan_age"].isna()
        if missing_age.any():
            # Try 1: Derive from Origination Date (FPD is typically ~1 month after origination)
            if "Origination Date" in out.columns:
                orig_age = out.loc[missing_age, "Origination Date"].apply(
                    lambda od: datedif_months_excel(od, global_cutoff)
                    if pd.notna(od) else np.nan
                )
                out.loc[missing_age, "_loan_age"] = orig_age
                missing_age = out["_loan_age"].isna()

            # Try 2: Derive from Maturity Date and Original Term
            #         age = Original Term - months_until_maturity
            if missing_age.any() and "Maturity Date" in out.columns:
                mat_remaining = out.loc[missing_age, "Maturity Date"].apply(
                    lambda md: datedif_months_excel(global_cutoff, md)
                    if pd.notna(md) else np.nan
                )
                inferred_age = out.loc[missing_age, "Original Term"] - mat_remaining
                fill_mask = inferred_age.notna()
                out.loc[missing_age & fill_mask.reindex(out.index, fill_value=False),
                        "_loan_age"] = inferred_age[fill_mask]
                missing_age = out["_loan_age"].isna()

            # Try 3: Fall back to 0 (treat as brand-new loan — full Original Term remaining)
            if missing_age.any():
                out.loc[missing_age, "_loan_age"] = 0

        out["_remaining_term"] = np.maximum(out["Original Term"] - out["_loan_age"], 0).astype(int)
    else:
        # Fallback: use Maturity Date if available, else Original Term
        out["_loan_age"] = np.nan
        if "Maturity Date" in out.columns:
            out["_remaining_term"] = out.apply(
                lambda row: max(
                    int(
                        (pd.Timestamp(row["Maturity Date"]).to_period("M")
                         - pd.Timestamp(row["First Payment Date"]).to_period("M")).n
                    )
                    if pd.notna(row["Maturity Date"]) and pd.notna(row["First Payment Date"])
                    else int(row["Original Term"]),
                    0,
                ),
                axis=1,
            )
        else:
            out["_remaining_term"] = out["Original Term"].clip(lower=0).astype(int)

    # Monthly rate and PMT
    out["_monthly_rate"] = out["Current Interest Rate"] / 12.0
    out["_pmt"] = out.apply(
        lambda row: level_payment(
            float(row["Current Principal Balance"]),
            float(row["_monthly_rate"]),
            int(row["_remaining_term"]),
        ),
        axis=1,
    )

    out["_global_cutoff"] = global_cutoff

    return out


def compute_contractual_cashflows(
    tape: pd.DataFrame,
    global_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Pure contractual (no CPR/CDR) cashflow engine matching Excel exactly.

    Full precision per-loan, pool-level rounding.
    Returns DataFrame with Period, Date, Principal, Interest, Cashflow, Balance + Totals row.
    """
    enriched = prepare_tape_for_engine(tape, global_cutoff_override=global_cutoff)

    bal = enriched["Current Principal Balance"].to_numpy(dtype=float)
    r_m = enriched["_monthly_rate"].to_numpy(dtype=float)
    pmt = enriched["_pmt"].to_numpy(dtype=float)
    remaining = enriched["_remaining_term"].to_numpy(dtype=int)
    ioflag = enriched["IOFlag"].to_numpy()

    # Filter to valid loans
    ok = np.isfinite(bal) & (bal > 0) & (remaining > 0)
    if ok.sum() == 0:
        raise ValueError("No valid loans found for contractual cashflow computation.")

    bal = bal[ok].copy()
    r_m = r_m[ok]
    pmt = pmt[ok]
    remaining = remaining[ok]
    ioflag = ioflag[ok]

    max_period = int(remaining.max())

    rows = []
    pool_beg = float(excel_round(bal.sum(), 2))

    # Period 0
    rows.append({
        "Period": "0",
        "Date": None,
        "Principal": 0.0,
        "Interest": 0.0,
        "Cashflow": 0.0,
        "Balance": pool_beg,
    })

    for p in range(1, max_period + 1):
        active = remaining >= p
        if not active.any():
            break

        b = bal[active]
        intr_raw = b * r_m[active]
        pay_raw = pmt[active]
        princ_raw = pay_raw - intr_raw

        # IO loans: principal = 0
        princ_raw = np.where(ioflag[active] == "Y", 0.0, princ_raw)
        princ_raw = np.clip(princ_raw, 0.0, b)

        end_raw = b - princ_raw
        bal[active] = end_raw

        principal_sum = float(excel_round(princ_raw.sum(), 2))
        interest_sum = float(excel_round(intr_raw.sum(), 2))
        cash_sum = float(excel_round(principal_sum + interest_sum, 2))
        balance_sum = float(excel_round(end_raw.sum(), 2))

        rows.append({
            "Period": str(p),
            "Date": None,
            "Principal": principal_sum,
            "Interest": interest_sum,
            "Cashflow": cash_sum,
            "Balance": balance_sum,
        })

    out = pd.DataFrame(rows)

    # Dates (25th of each month)
    n_periods = int(out["Period"].astype(int).max())
    dates = period_dates_25th(global_cutoff, n_periods=n_periods)
    out["Date"] = out["Period"].astype(int).apply(lambda k: dates[k].strftime("%m/%d/%Y"))

    # Totals row
    total_principal = pool_beg
    total_interest = float(
        excel_round(out.loc[out["Period"].astype(int) >= 1, "Interest"].sum(), 2)
    )
    total_cashflow = float(
        excel_round(out.loc[out["Period"].astype(int) >= 1, "Cashflow"].sum(), 2)
    )

    total_row = pd.DataFrame([{
        "Period": "",
        "Date": "Total",
        "Principal": total_principal,
        "Interest": total_interest,
        "Cashflow": total_cashflow,
        "Balance": np.nan,
    }])

    header_row = pd.DataFrame([{
        "Period": "Period",
        "Date": "Date",
        "Principal": np.nan,
        "Interest": np.nan,
        "Cashflow": np.nan,
        "Balance": np.nan,
    }])

    final = pd.concat([total_row, header_row, out], ignore_index=True)

    for c in ["Principal", "Interest", "Cashflow", "Balance"]:
        final[c] = final[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):,.2f}")
    final["Period"] = final["Period"].astype(str)

    return final
