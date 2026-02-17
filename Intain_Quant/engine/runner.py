"""
Projection runner — orchestrates Monte Carlo paths through the cashflow engine.

Uses Excel-matched base calculations (datedif_months_excel for loan age,
Original Term - age for remaining term, full-precision PMT, IO flag support,
period_dates_25th for dates, excel_round only at output level).

Two modes of operation:
  1. Single scenario: Pass a BehaviorModel directly (like ConstantHazardModel)
  2. Monte Carlo:     Pass SampledPaths — each path uses different assumptions

The behavioral overlay (SMM/MDR random draws, default/prepay logic, recovery lag)
is layered on top of the Excel-matched base calculations.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from behaviors.base import BehaviorModel
from behaviors.scenario import SeasonedScenarioHazardModel
from core.config import ProjectionConfig
from core.utils import (
    annual_to_monthly_hazard,
    datedif_months_excel,
    excel_round,
    period_dates_25th,
)
from distributions.sampler import SampledPaths

from .cashflow import prepare_tape_for_engine, level_payment


def run_projection(
    tape: pd.DataFrame,
    config: ProjectionConfig,
    *,
    behavior_model: Optional[BehaviorModel] = None,
    sampled_paths: Optional[SampledPaths] = None,
    aggregate: Literal["portfolio", "loan"] = "portfolio",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run cashflow projection — either single-scenario or full Monte Carlo.

    Parameters
    ----------
    tape : pd.DataFrame
        Static loan tape (one row per loan)
    config : ProjectionConfig
        Projection settings (as_of_date, n_paths, seed, etc.)
    behavior_model : BehaviorModel, optional
        Single behavior model applied to ALL paths (old mode, backwards compatible)
    sampled_paths : SampledPaths, optional
        Monte Carlo sampled assumptions — each path gets different CPR/CDR/Severity (new mode)
    aggregate : str
        "portfolio" (default) or "loan" level output

    Exactly one of behavior_model or sampled_paths must be provided.
    If sampled_paths is provided, config.n_paths is overridden by sampled_paths.n_paths.

    Returns
    -------
    (cashflows_df, raw_path_results)
    cashflows_df: DataFrame with columns per the engine output spec
    raw_path_results: dict with per-path WAL, Loss%, etc. for aggregation
    """
    # Validate inputs
    if behavior_model is None and sampled_paths is None:
        raise ValueError("Must provide either behavior_model or sampled_paths.")
    if behavior_model is not None and sampled_paths is not None:
        raise ValueError("Provide behavior_model OR sampled_paths, not both.")

    use_mc = sampled_paths is not None
    n_paths = sampled_paths.n_paths if use_mc else config.n_paths

    cfg = config
    rng = np.random.default_rng(cfg.seed)

    # Prepare tape with Excel-matched calculations
    tape2 = prepare_tape_for_engine(tape, global_cutoff_override=cfg.as_of_date)
    n_loans = len(tape2)
    if n_loans == 0:
        raise ValueError("Empty tape.")

    # Use Excel-matched remaining terms (computed in prepare_tape_for_engine)
    rem_terms = np.minimum(
        tape2["_remaining_term"].to_numpy(dtype=int),
        cfg.projection_months,
    )
    horizon = int(max(rem_terms.max(), 0))
    if horizon == 0:
        raise ValueError("All loans have zero remaining term at the as-of date.")

    # Use 25th-of-month dates
    dates_list = period_dates_25th(pd.Timestamp(cfg.as_of_date), horizon)
    dates = pd.DatetimeIndex(dates_list[1:])  # skip period 0 date

    loan_ids = tape2["Loan ID"].to_numpy()
    bal0 = tape2["Current Principal Balance"].to_numpy(dtype=float)
    rate_annual = tape2["Current Interest Rate"].to_numpy(dtype=float)
    r_m_all = tape2["_monthly_rate"].to_numpy(dtype=float)
    pmt_all = tape2["_pmt"].to_numpy(dtype=float)
    ioflag_all = tape2["IOFlag"].to_numpy()

    # Allocate portfolio-level arrays
    port_begin_bal = np.zeros((n_paths, horizon), dtype=float)
    port_end_bal = np.zeros((n_paths, horizon), dtype=float)
    port_sched_prin = np.zeros((n_paths, horizon), dtype=float)
    port_prepay = np.zeros((n_paths, horizon), dtype=float)
    port_default_prin = np.zeros((n_paths, horizon), dtype=float)
    port_recovery = np.zeros((n_paths, horizon), dtype=float)
    port_loss = np.zeros((n_paths, horizon), dtype=float)
    port_int = np.zeros((n_paths, horizon), dtype=float)
    port_cf = np.zeros((n_paths, horizon), dtype=float)
    port_pi = np.zeros((n_paths, horizon), dtype=float)
    port_prin = np.zeros((n_paths, horizon), dtype=float)

    # Per-path results for PM aggregation
    path_wal = np.zeros(n_paths, dtype=float)
    path_cum_loss = np.zeros(n_paths, dtype=float)
    path_total_principal = np.zeros(n_paths, dtype=float)

    original_balance = float(tape2["Original Principal Balance"].fillna(0.0).sum())
    current_balance = float(tape2["Current Principal Balance"].fillna(0.0).sum())

    # ========= MAIN PATH LOOP =========
    for p in range(n_paths):

        # --- Build per-path behavior model ---
        if use_mc:
            path_assumptions = sampled_paths.get_path(p)
            path_model = SeasonedScenarioHazardModel(
                cpr_annual=path_assumptions["cpr"],
                cdr_annual=path_assumptions["cdr"],
                severity=path_assumptions["severity"],
            )
            recovery_lag = path_assumptions["recovery_lag"]
        else:
            path_model = behavior_model
            recovery_lag = cfg.recovery_lag_months

        # Get hazards for this path's assumptions
        hazards = path_model.forecast(tape2, horizon, as_of_date=pd.Timestamp(cfg.as_of_date))
        smm = annual_to_monthly_hazard(hazards.cpr_annual)
        mdr = annual_to_monthly_hazard(hazards.cdr_annual)
        sev = np.clip(hazards.severity, 0.0, 1.0)

        recovery_bucket = np.zeros((n_loans, horizon + recovery_lag + 1), dtype=float)

        for i in range(n_loans):
            term_i = int(rem_terms[i])
            if term_i <= 0:
                continue

            bal = float(bal0[i]) if np.isfinite(bal0[i]) else 0.0
            if bal <= 0:
                continue

            r_m = float(r_m_all[i])
            pay = float(pmt_all[i])
            is_io = ioflag_all[i] == "Y"

            alive = True
            for t in range(term_i):
                if not alive:
                    break

                begin = bal
                interest = begin * r_m

                # IO handling: if IO flag is "Y", scheduled principal = 0
                if is_io:
                    sched_prin = 0.0
                else:
                    sched_prin = min(max(pay - interest, 0.0), begin)

                bal_after_sched = begin - sched_prin

                recovery = float(recovery_bucket[i, t])

                u_def = rng.random()
                u_pre = rng.random()
                did_default = u_def < float(mdr[i, t])
                did_prepay = (not did_default) and (u_pre < float(smm[i, t]))

                prepay_amt = 0.0
                default_prin = 0.0
                loss_amt = 0.0

                if did_default:
                    default_prin = bal_after_sched
                    loss_amt = float(sev[i, t]) * default_prin
                    rec = (1.0 - float(sev[i, t])) * default_prin
                    rec_t = t + recovery_lag
                    if rec_t < recovery_bucket.shape[1]:
                        recovery_bucket[i, rec_t] += rec
                    bal = 0.0
                    alive = False
                elif did_prepay:
                    prepay_amt = bal_after_sched
                    bal = 0.0
                    alive = False
                else:
                    bal = bal_after_sched

                end = bal
                total_cf = interest + sched_prin + prepay_amt + recovery
                pi_cf = interest + sched_prin

                port_cf[p, t] += total_cf
                port_int[p, t] += interest
                port_prin[p, t] += (sched_prin + prepay_amt + recovery)
                port_begin_bal[p, t] += begin
                port_end_bal[p, t] += end
                port_sched_prin[p, t] += sched_prin
                port_prepay[p, t] += prepay_amt
                port_default_prin[p, t] += default_prin
                port_loss[p, t] += loss_amt
                port_recovery[p, t] += recovery
                port_pi[p, t] += pi_cf

        # --- Compute per-path metrics ---
        path_total_principal[p] = port_prin[p, :].sum()
        path_cum_loss[p] = port_loss[p, :].sum()

        # WAL for this path
        t_years = np.array([(dates[t] - cfg.as_of_date).days / 365.25 for t in range(horizon)])
        t_years = np.clip(t_years, 0, None)
        total_prin_p = port_prin[p, :].sum()
        if total_prin_p > 0:
            path_wal[p] = (port_prin[p, :] * t_years).sum() / total_prin_p
        else:
            path_wal[p] = 0.0

    # ========= BUILD OUTPUT DATAFRAME =========
    cashflows = (
        pd.DataFrame(
            {
                "path_id": np.repeat(np.arange(n_paths), horizon),
                "date": np.tile(dates.values, n_paths),
                "begin_balance": port_begin_bal.reshape(-1),
                "pi": port_pi.reshape(-1),
                "interest": port_int.reshape(-1),
                "scheduled_principal": port_sched_prin.reshape(-1),
                "prepayment": port_prepay.reshape(-1),
                "default_principal": port_default_prin.reshape(-1),
                "loss": port_loss.reshape(-1),
                "recovery": port_recovery.reshape(-1),
                "end_balance": port_end_bal.reshape(-1),
                "principal": port_prin.reshape(-1),
                "total_cashflow": port_cf.reshape(-1),
            }
        )
        .sort_values(["path_id", "date"])
        .reset_index(drop=True)
    )

    # Per-path results (for PM aggregation)
    raw_path_results = {
        "path_wal": path_wal,
        "path_cum_loss_pct": path_cum_loss / original_balance if original_balance > 0 else path_cum_loss,
        "path_cum_loss_abs": path_cum_loss,
        "path_total_principal": path_total_principal,
        "original_balance": original_balance,
        "current_balance": current_balance,
        "n_paths": n_paths,
        "horizon": horizon,
        "dates": dates,
        "cashflows_by_path": cashflows,
    }

    # If MC mode, attach the sampled assumptions for traceability
    if use_mc:
        raw_path_results["sampled_assumptions"] = sampled_paths.to_dataframe()

    return cashflows, raw_path_results
