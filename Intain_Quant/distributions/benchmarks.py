"""
Industry benchmark distributions for RMBS behavioral assumptions.

Source B: When historical data is unavailable or insufficient (e.g., only 1-2 years
of history in a calm economy won't show crisis behavior), use these benchmarks
derived from rating agency publications, Intex assumption libraries, and
Fed research papers.

Usage:
  - Day 1: Use benchmarks as primary source
  - Once historical.py computes real distributions, blend with benchmarks
  - For tail scenarios (beyond observed history), always extend with benchmarks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BenchmarkDistribution:
    """A single benchmark distribution for one assumption variable."""
    variable: str
    mean: float
    std: float
    min_val: float
    max_val: float
    shape: str  # "normal", "lognormal", "beta"
    source: str  # where this came from


# ----- CES (Closed-End Second Lien) specific benchmarks -----
# These are calibrated for non-QM / CES deals like JP Morgan CES
# Second liens have higher CPR/CDR/Severity than prime first liens

CES_BENCHMARKS: Dict[str, BenchmarkDistribution] = {
    "cpr": BenchmarkDistribution(
        variable="CPR",
        mean=0.10,
        std=0.03,
        min_val=0.02,
        max_val=0.25,
        shape="normal",
        source="Rating agency CES presale reports (Moody's/S&P/Fitch 2023-2024)",
    ),
    "cdr": BenchmarkDistribution(
        variable="CDR",
        mean=0.02,
        std=0.015,
        min_val=0.005,
        max_val=0.12,
        shape="lognormal",
        source="Rating agency CES presale reports; skewed right (defaults can spike)",
    ),
    "severity": BenchmarkDistribution(
        variable="Severity",
        mean=0.35,
        std=0.10,
        min_val=0.15,
        max_val=0.75,
        shape="beta",
        source="Moody's LGD studies for second liens; bounded [0,1]",
    ),
    "recovery_lag": BenchmarkDistribution(
        variable="Recovery Lag (months)",
        mean=12.0,
        std=4.0,
        min_val=6.0,
        max_val=24.0,
        shape="normal",
        source="Industry average foreclosure-to-liquidation timeline",
    ),
}

# ----- Named scenario sets (for deterministic scenario analysis) -----
NAMED_SCENARIOS: Dict[str, Dict[str, float]] = {
    "base": {
        "cpr": 0.10,
        "cdr": 0.02,
        "severity": 0.35,
        "recovery_lag": 12,
    },
    "fast_prepay": {
        "cpr": 0.15,
        "cdr": 0.015,
        "severity": 0.30,
        "recovery_lag": 10,
    },
    "slow_prepay_high_loss": {
        "cpr": 0.05,
        "cdr": 0.05,
        "severity": 0.50,
        "recovery_lag": 15,
    },
    "severe_stress": {
        "cpr": 0.03,
        "cdr": 0.08,
        "severity": 0.60,
        "recovery_lag": 18,
    },
    "crisis": {
        "cpr": 0.02,
        "cdr": 0.12,
        "severity": 0.75,
        "recovery_lag": 24,
    },
}


def get_benchmark_distributions(
    deal_type: str = "ces",
) -> Dict[str, BenchmarkDistribution]:
    """
    Return benchmark distributions for a given deal type.

    Parameters
    ----------
    deal_type : str
        One of "ces" (closed-end second), "prime", "nonqm".
        Currently only "ces" is implemented.

    Returns
    -------
    Dict mapping variable names to BenchmarkDistribution objects.
    """
    if deal_type.lower() == "ces":
        return CES_BENCHMARKS.copy()
    else:
        # TODO: add prime and nonqm benchmarks
        raise NotImplementedError(
            f"Benchmarks for deal_type='{deal_type}' not yet implemented. "
            f"Available: 'ces'"
        )


def get_named_scenario(name: str) -> Dict[str, float]:
    """
    Return a named deterministic scenario.

    Parameters
    ----------
    name : str
        One of: "base", "fast_prepay", "slow_prepay_high_loss",
        "severe_stress", "crisis"
    """
    if name not in NAMED_SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{name}'. "
            f"Available: {list(NAMED_SCENARIOS.keys())}"
        )
    return NAMED_SCENARIOS[name].copy()


def blend_with_benchmarks(
    historical_mean: float,
    historical_std: float,
    historical_n: int,
    benchmark: BenchmarkDistribution,
    *,
    min_observations: int = 12,
) -> tuple:
    """
    Blend historical estimates with benchmarks based on sample size.

    If historical data has fewer than min_observations months, weight
    the benchmark more heavily. As historical data grows, rely more
    on observed values.

    Returns
    -------
    (blended_mean, blended_std)
    """
    import math

    if historical_n <= 0 or math.isnan(historical_mean):
        return benchmark.mean, benchmark.std

    # Weight historical data by its sample size relative to threshold
    weight_hist = min(historical_n / min_observations, 1.0)
    weight_bench = 1.0 - weight_hist

    blended_mean = weight_hist * historical_mean + weight_bench * benchmark.mean
    blended_std = weight_hist * historical_std + weight_bench * benchmark.std

    return blended_mean, blended_std
