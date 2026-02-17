"""
Distributions package — estimate, store, and sample behavioral assumption distributions.

This is the NEW layer that grounds Monte Carlo in data:
  1. historical.py   — compute CPR/CDR/Severity from actual deal performance
  2. benchmarks.py   — industry-standard fallback distributions
  3. correlation.py  — estimate correlation structure between assumptions
  4. sampler.py      — generate N correlated (CPR, CDR, Severity, RecoveryLag) paths
"""

from .historical import compute_historical_distributions
from .benchmarks import get_benchmark_distributions
from .correlation import compute_correlation_matrix
from .sampler import DistributionParams, MonteCarloSampler, SampledPaths

__all__ = [
    "compute_historical_distributions",
    "get_benchmark_distributions",
    "compute_correlation_matrix",
    "DistributionParams",
    "MonteCarloSampler",
    "SampledPaths",
]
