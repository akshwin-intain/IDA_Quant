"""
PM Decision Support — probability statements, stress flags, and investment signals.

Translates distribution summaries into answers a PM can act on:
  Q1: "Am I getting paid enough?" → compare mean loss vs offered yield
  Q2: "What breaks this deal?"   → tail risk: P(loss > yield)
  Q3: "How sensitive is this?"   → spread between 5th and 95th percentile
  Q4: "How long is my money stuck?" → WAL extension risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DecisionReport:
    """Structured PM decision output."""
    deal_name: str
    yield_offered: Optional[float]  # annual yield on the tranche being evaluated

    # Core metrics
    mean_wal: float
    mean_loss_pct: float
    mean_expected_life: float

    # Tail risk
    p95_loss_pct: float
    p99_loss_pct: float
    worst_case_loss_pct: float

    # Probability statements
    prob_loss_exceeds_yield: Optional[float]  # P(cum loss > yield)
    prob_loss_exceeds_5pct: float
    prob_loss_exceeds_10pct: float

    # Extension risk
    p95_wal: float
    wal_spread: float  # P95 WAL - P05 WAL

    # Sensitivity (spread = width of distribution)
    loss_spread_5_95: float  # P95 loss - P05 loss

    # Flags
    flags: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a display-friendly table."""
        rows = [
            {"Metric": "Deal", "Value": self.deal_name, "Unit": ""},
            {"Metric": "Mean WAL", "Value": f"{self.mean_wal:.2f}", "Unit": "years"},
            {"Metric": "Mean Expected Life", "Value": f"{self.mean_expected_life:.2f}", "Unit": "years"},
            {"Metric": "Mean Cumulative Loss", "Value": f"{self.mean_loss_pct:.2%}", "Unit": ""},
            {"Metric": "95th Pctl Loss", "Value": f"{self.p95_loss_pct:.2%}", "Unit": ""},
            {"Metric": "99th Pctl Loss", "Value": f"{self.p99_loss_pct:.2%}", "Unit": ""},
            {"Metric": "Worst Case Loss", "Value": f"{self.worst_case_loss_pct:.2%}", "Unit": ""},
            {"Metric": "P(Loss > 5%)", "Value": f"{self.prob_loss_exceeds_5pct:.1%}", "Unit": ""},
            {"Metric": "P(Loss > 10%)", "Value": f"{self.prob_loss_exceeds_10pct:.1%}", "Unit": ""},
            {"Metric": "95th Pctl WAL", "Value": f"{self.p95_wal:.2f}", "Unit": "years"},
            {"Metric": "WAL Spread (P05-P95)", "Value": f"{self.wal_spread:.2f}", "Unit": "years"},
            {"Metric": "Loss Spread (P05-P95)", "Value": f"{self.loss_spread_5_95:.2%}", "Unit": ""},
        ]
        if self.yield_offered is not None:
            rows.insert(1, {"Metric": "Yield Offered", "Value": f"{self.yield_offered:.2%}", "Unit": ""})
            rows.append({
                "Metric": "P(Loss > Yield)",
                "Value": f"{self.prob_loss_exceeds_yield:.1%}" if self.prob_loss_exceeds_yield is not None else "N/A",
                "Unit": "",
            })
        if self.flags:
            rows.append({"Metric": "FLAGS", "Value": " | ".join(self.flags), "Unit": ""})
        return pd.DataFrame(rows)


def generate_decision_report(
    path_metrics: pd.DataFrame,
    *,
    deal_name: str = "Unknown Deal",
    yield_offered: Optional[float] = None,
) -> DecisionReport:
    """
    Generate a PM decision report from per-path metrics.

    Parameters
    ----------
    path_metrics : pd.DataFrame
        Output of pm.metrics.compute_path_metrics().
        Required columns: wal_years, cum_loss_pct, expected_life_years
    deal_name : str
        Deal identifier for the report
    yield_offered : float, optional
        Annual yield on the tranche (e.g., 0.06 for 6%).
        If provided, computes P(loss > yield).
    """
    wal = path_metrics["wal_years"].dropna().values
    loss = path_metrics["cum_loss_pct"].dropna().values
    life = path_metrics["expected_life_years"].dropna().values

    n = len(loss)
    if n == 0:
        raise ValueError("No path metrics to generate report from.")

    # Probability statements
    prob_loss_gt_5 = float(np.mean(loss > 0.05))
    prob_loss_gt_10 = float(np.mean(loss > 0.10))
    prob_loss_gt_yield = float(np.mean(loss > yield_offered)) if yield_offered is not None else None

    # Percentiles
    p05_wal = float(np.percentile(wal, 5)) if len(wal) > 0 else 0.0
    p95_wal = float(np.percentile(wal, 95)) if len(wal) > 0 else 0.0
    p05_loss = float(np.percentile(loss, 5))
    p95_loss = float(np.percentile(loss, 95))
    p99_loss = float(np.percentile(loss, 99))

    # Flags
    flags = []
    if p95_loss > 0.05:
        flags.append("HIGH_TAIL_LOSS: 95th pctl loss exceeds 5%")
    if yield_offered is not None and prob_loss_gt_yield is not None and prob_loss_gt_yield > 0.10:
        flags.append(f"YIELD_AT_RISK: {prob_loss_gt_yield:.0%} chance loss exceeds yield")
    if (p95_wal - p05_wal) > 3.0:
        flags.append("EXTENSION_RISK: WAL spread exceeds 3 years")
    if (p95_loss - p05_loss) > 0.04:
        flags.append("HIGH_SENSITIVITY: Loss spread (P05-P95) exceeds 4%")

    return DecisionReport(
        deal_name=deal_name,
        yield_offered=yield_offered,
        mean_wal=float(np.mean(wal)) if len(wal) > 0 else 0.0,
        mean_loss_pct=float(np.mean(loss)),
        mean_expected_life=float(np.mean(life)) if len(life) > 0 else 0.0,
        p95_loss_pct=p95_loss,
        p99_loss_pct=p99_loss,
        worst_case_loss_pct=float(np.max(loss)),
        prob_loss_exceeds_yield=prob_loss_gt_yield,
        prob_loss_exceeds_5pct=prob_loss_gt_5,
        prob_loss_exceeds_10pct=prob_loss_gt_10,
        p95_wal=p95_wal,
        wal_spread=p95_wal - p05_wal,
        loss_spread_5_95=p95_loss - p05_loss,
        flags=flags,
    )
