from __future__ import annotations

from typing import Tuple

# Canonical "model tape" columns (from your `CES1_2024.ipynb` selection).
# The projection engine enforces that ONLY these columns are used as tape inputs.
CES1_TAPE_COLUMNS: Tuple[str, ...] = (
    "Loan ID",
    "Origination Date",
    "First Payment Date",
    "Maturity Date",
    "Amortisation Type",
    "Original Principal Balance",
    "Current Principal Balance",
    "Original Term",
    "Current Interest Rate",
    "Interest Rate Type",
    "Property Type",
    "Original Valuation Amount",
    "Current Valuation Amount",
    "Original Loan-To-Value",
    "Current Loan-To-Value",
    "Property State",
    "Borrower FICO",
    "Debt To Income Ratio",
    "Primary Income",
    "Employment Status",
    "IOFlag",
    "Data Cut-Off Date",
)


