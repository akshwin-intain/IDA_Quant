"""
Aggregate N path results into PM-consumable distribution summaries.

THIS IS WHERE THE CORRECTED FLOW DELIVERS VALUE.

Instead of: "WAL = 4.5 years" (one number, no context)
The PM gets: "WAL: mean=4.5, median=4.3, 5th=3.2, 95th=7.1 years"

Instead of: "Cumulative Loss = 2.6%"
The PM gets: "Loss: mean=2.6%, 95th percentile=6.4%, worst case=9.8%"

This enables the PM's core decision framework:
  Q1: "Am I getting paid enough for base case?" → compare mean loss vs yield
  Q2: "What breaks this deal?" → look at 95th/99th percentile losses
  Q3: "How sensitive?" → compare spread between 5th and 95th percentiles
  Q4: "How long is my money stuck?" → WAL distribution spread
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def aggregate_path_results(
    path_metrics: pd.DataFrame,
    *,
    percentiles: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-path metrics into distribution summaries.

    Parameters
    ----------
    path_metrics : pd.DataFrame
        Output of pm.metrics.compute_path_metrics() with one row per path.
        Required columns: path_id, wal_years, cum_loss_pct, expected_life_years
    percentiles : tuple of float
        Percentile levels to report

    Returns
    -------
    Dict with:
      "summary_table":     One row per metric (WAL, Loss, Life) with mean/median/percentiles
      "wal_distribution":  Array of all WAL values (for histograms)
      "loss_distribution": Array of all loss values (for histograms)
      "life_distribution": Array of all expected life values (for histograms)
    """

    metrics_to_summarize = {
        "WAL (years)": "wal_years",
        "Cumulative Loss (%)": "cum_loss_pct",
        "Expected Life (years)": "expected_life_years",
    }

    rows = []
    for label, col in metrics_to_summarize.items():
        if col not in path_metrics.columns:
            continue

        values = path_metrics[col].dropna().values
        if len(values) == 0:
            continue

        row = {
            "Metric": label,
            "Mean": float(np.mean(values)),
            "Std Dev": float(np.std(values)),
            "Min": float(np.min(values)),
        }
        for p in percentiles:
            pct_label = f"P{int(p * 100):02d}"
            row[pct_label] = float(np.percentile(values, p * 100))
        row["Max"] = float(np.max(values))
        rows.append(row)

    summary_table = pd.DataFrame(rows)

    # Format loss as percentage for readability
    if "cum_loss_pct" in path_metrics.columns:
        loss_vals = path_metrics["cum_loss_pct"].dropna().values
    else:
        loss_vals = np.array([])

    return {
        "summary_table": summary_table,
        "wal_distribution": path_metrics["wal_years"].dropna().values
        if "wal_years" in path_metrics.columns
        else np.array([]),
        "loss_distribution": loss_vals,
        "life_distribution": path_metrics["expected_life_years"].dropna().values
        if "expected_life_years" in path_metrics.columns
        else np.array([]),
        "n_paths": len(path_metrics),
    }


def aggregate_cashflows_by_date(
    cashflows: pd.DataFrame,
    *,
    date_col: str = "date",
    path_col: str = "path_id",
    percentiles: Tuple[float, ...] = (0.05, 0.50, 0.95),
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate monthly cashflow columns across paths → expected + percentile bands.

    Returns time series of expected cashflows with confidence bands.
    Useful for cashflow waterfall charts.
    """
    df = cashflows.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    value_cols = [
        "begin_balance", "interest", "scheduled_principal", "prepayment",
        "default_principal", "loss", "recovery", "end_balance",
        "principal", "total_cashflow",
    ]
    value_cols = [c for c in value_cols if c in df.columns]

    # Expected (mean across paths) by date
    by_path_date = df.groupby([path_col, date_col], as_index=False)[value_cols].sum()
    expected = by_path_date.groupby(date_col, as_index=False)[value_cols].mean()
    expected = expected.sort_values(date_col).reset_index(drop=True)

    # Percentile bands for total_cashflow
    pct_data = {}
    if "total_cashflow" in by_path_date.columns:
        for p in percentiles:
            label = f"p{int(p * 100):02d}"
            pct_series = (
                by_path_date.groupby(date_col)["total_cashflow"]
                .quantile(p)
                .reset_index()
                .rename(columns={"total_cashflow": label})
            )
            pct_data[label] = pct_series

    # Merge percentile bands
    pct_df = None
    for label, pdf in pct_data.items():
        if pct_df is None:
            pct_df = pdf
        else:
            pct_df = pct_df.merge(pdf, on=date_col, how="outer")

    if pct_df is not None:
        pct_df = pct_df.sort_values(date_col).reset_index(drop=True)

    return {
        "expected_by_date": expected,
        "percentile_bands": pct_df,
    }
