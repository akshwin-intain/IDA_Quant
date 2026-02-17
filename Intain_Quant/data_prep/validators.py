"""
Data quality validation for loan tapes before they enter the engine.

Catches problems early:
- Missing critical fields
- Negative balances
- Rates outside plausible bounds
- Dates that don't make sense
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from core.schema import CES1_TAPE_COLUMNS
from core.utils import require_columns


@dataclass
class ValidationResult:
    """Collects all validation warnings/errors for a tape."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not lines:
            lines.append("✓ All checks passed.")
        return "\n".join(lines)


def validate_tape(
    tape: pd.DataFrame,
    *,
    columns: tuple = CES1_TAPE_COLUMNS,
) -> ValidationResult:
    """
    Run all validation checks on a static loan tape.
    Returns a ValidationResult with errors (blocking) and warnings (informational).
    """
    result = ValidationResult()

    # --- Schema checks ---
    missing = [c for c in columns if c not in tape.columns]
    if missing:
        result.errors.append(f"Missing required columns: {missing}")
        return result  # can't continue without columns

    n = len(tape)
    if n == 0:
        result.errors.append("Tape is empty (0 rows).")
        return result

    # --- Loan ID ---
    if tape["Loan ID"].isna().any():
        n_missing = int(tape["Loan ID"].isna().sum())
        result.errors.append(f"{n_missing} rows have null Loan ID.")

    n_dup = int(tape["Loan ID"].duplicated().sum())
    if n_dup > 0:
        result.warnings.append(f"{n_dup} duplicate Loan IDs found.")

    # --- Balances ---
    for col in ["Original Principal Balance", "Current Principal Balance"]:
        if col in tape.columns:
            vals = pd.to_numeric(tape[col], errors="coerce")
            n_neg = int((vals < 0).sum())
            n_null = int(vals.isna().sum())
            if n_neg > 0:
                result.errors.append(f"{n_neg} rows have negative {col}.")
            if n_null > 0:
                result.warnings.append(f"{n_null} rows have null/unparseable {col}.")

    # --- Interest Rate ---
    if "Current Interest Rate" in tape.columns:
        rates = pd.to_numeric(tape["Current Interest Rate"], errors="coerce")
        n_null = int(rates.isna().sum())
        # Rates should be in decimal form (e.g. 0.08 not 8.0)
        n_high = int((rates > 1.0).sum())
        n_neg = int((rates < 0).sum())
        if n_null > 0:
            result.warnings.append(f"{n_null} rows have null interest rate.")
        if n_high > 0:
            result.warnings.append(
                f"{n_high} rows have interest rate > 1.0 — check if rates are in "
                f"percent vs decimal form."
            )
        if n_neg > 0:
            result.errors.append(f"{n_neg} rows have negative interest rate.")

    # --- Dates ---
    for dcol in ["Origination Date", "First Payment Date", "Maturity Date"]:
        if dcol in tape.columns:
            dts = pd.to_datetime(tape[dcol], errors="coerce")
            n_null = int(dts.isna().sum())
            if n_null > 0:
                result.warnings.append(f"{n_null} rows have null/unparseable {dcol}.")

    # --- Original Term ---
    if "Original Term" in tape.columns:
        terms = pd.to_numeric(tape["Original Term"], errors="coerce")
        n_zero = int((terms <= 0).sum())
        if n_zero > 0:
            result.warnings.append(f"{n_zero} rows have zero or negative Original Term.")

    # --- LTV bounds ---
    for ltv_col in ["Original Loan-To-Value", "Current Loan-To-Value"]:
        if ltv_col in tape.columns:
            ltv = pd.to_numeric(tape[ltv_col], errors="coerce")
            n_over = int((ltv > 2.0).sum())  # >200% LTV is suspicious
            if n_over > 0:
                result.warnings.append(
                    f"{n_over} rows have {ltv_col} > 2.0 (200%) — verify units."
                )

    # --- FICO bounds ---
    if "Borrower FICO" in tape.columns:
        fico = pd.to_numeric(tape["Borrower FICO"], errors="coerce")
        n_low = int((fico < 300).sum())
        n_high = int((fico > 900).sum())
        if n_low > 0:
            result.warnings.append(f"{n_low} rows have FICO < 300.")
        if n_high > 0:
            result.warnings.append(f"{n_high} rows have FICO > 900.")

    return result
