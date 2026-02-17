"""
Loan-level event simulation — random draws to determine default/prepay per loan per month.

This is the SECOND layer of randomness (the first being Monte Carlo assumption sampling):
  Layer 1 (sampler.py): Which economic future are we in? (CPR=6%, CDR=4.5%, ...)
  Layer 2 (events.py):  Given that economic future, WHICH specific loans default/prepay?

Even with the same CPR=10%, different random draws produce different outcomes:
  - Path 1: Loans A, C, F prepay
  - Path 2: Loans B, D, G prepay
  - Both paths have roughly the same total prepayment, but different loans are affected.

This is appropriate and necessary — it captures idiosyncratic loan-level uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoanMonthEvent:
    """Result of simulating one loan for one month."""
    did_default: bool
    did_prepay: bool
    alive_after: bool
    prepay_amount: float
    default_principal: float
    loss_amount: float
    recovery_amount: float
    recovery_month_offset: int  # how many months until recovery arrives


def simulate_loan_month(
    *,
    balance_after_scheduled: float,
    smm: float,
    mdr: float,
    severity: float,
    recovery_lag_months: int,
    rng: np.random.Generator,
) -> LoanMonthEvent:
    """
    Simulate whether a loan defaults, prepays, or survives this month.

    Priority: default is checked first (if both would happen, default wins).
    This matches industry convention and your existing engine.

    Parameters
    ----------
    balance_after_scheduled : float
        Loan balance after scheduled principal payment
    smm : float
        Single Monthly Mortality (monthly prepay probability), already converted from CPR
    mdr : float
        Monthly Default Rate, already converted from CDR
    severity : float
        Loss severity in [0, 1]
    recovery_lag_months : int
        Months from default to recovery
    rng : np.random.Generator
        Random number generator
    """
    u_default = rng.random()
    u_prepay = rng.random()

    did_default = u_default < mdr
    did_prepay = (not did_default) and (u_prepay < smm)

    prepay_amount = 0.0
    default_principal = 0.0
    loss_amount = 0.0
    recovery_amount = 0.0

    if did_default:
        default_principal = balance_after_scheduled
        loss_amount = severity * default_principal
        recovery_amount = (1.0 - severity) * default_principal
    elif did_prepay:
        prepay_amount = balance_after_scheduled

    return LoanMonthEvent(
        did_default=did_default,
        did_prepay=did_prepay,
        alive_after=not (did_default or did_prepay),
        prepay_amount=prepay_amount,
        default_principal=default_principal,
        loss_amount=loss_amount,
        recovery_amount=recovery_amount,
        recovery_month_offset=recovery_lag_months,
    )
