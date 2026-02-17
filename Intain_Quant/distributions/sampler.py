"""
Monte Carlo Sampler — generates N sets of correlated (CPR, CDR, Severity, RecoveryLag).

THIS IS THE CORE OF THE CORRECTED FLOW.

Input:  Distribution parameters (mean, std for each variable) + correlation matrix
Output: (N × 4) table of sampled assumption sets — one row per path

Each row represents one plausible economic future:
  Path 1: CPR=9.2%, CDR=1.8%, Severity=33%, RecoveryLag=11mo  (mild)
  Path 2: CPR=6.1%, CDR=4.5%, Severity=48%, RecoveryLag=16mo  (recession)
  Path 3: CPR=11.4%, CDR=1.1%, Severity=28%, RecoveryLag=10mo (good economy)

The correlation matrix ensures these move together realistically.

Method:
  1. Sample from multivariate normal (correlated standard normal draws)
  2. Transform marginals to match target distributions:
     - CPR: Normal, clipped to [min, max]
     - CDR: LogNormal (skewed right — defaults can spike but can't go below 0)
     - Severity: Beta (bounded 0-1)
     - Recovery Lag: Normal, rounded to integer months, clipped
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .correlation import BENCHMARK_CORRELATION_MATRIX, _ensure_positive_definite


@dataclass(frozen=True)
class DistributionParams:
    """
    Complete specification of assumption distributions for Monte Carlo.

    This is what feeds the sampler. Can be built from:
    - benchmarks.get_benchmark_distributions()
    - historical.compute_historical_distributions()
    - A blend of both via benchmarks.blend_with_benchmarks()
    """
    cpr_mean: float = 0.10
    cpr_std: float = 0.03
    cpr_min: float = 0.02
    cpr_max: float = 0.25

    cdr_mean: float = 0.02
    cdr_std: float = 0.015
    cdr_min: float = 0.005
    cdr_max: float = 0.12

    severity_mean: float = 0.35
    severity_std: float = 0.10
    severity_min: float = 0.15
    severity_max: float = 0.75

    recovery_lag_mean: float = 12.0
    recovery_lag_std: float = 4.0
    recovery_lag_min: float = 6.0
    recovery_lag_max: float = 24.0

    correlation_matrix: Optional[np.ndarray] = None

    @property
    def corr(self) -> np.ndarray:
        if self.correlation_matrix is not None:
            return self.correlation_matrix
        return BENCHMARK_CORRELATION_MATRIX.copy()

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all distribution parameters."""
        return pd.DataFrame([
            {"Variable": "CPR", "Mean": self.cpr_mean, "StdDev": self.cpr_std,
             "Min": self.cpr_min, "Max": self.cpr_max},
            {"Variable": "CDR", "Mean": self.cdr_mean, "StdDev": self.cdr_std,
             "Min": self.cdr_min, "Max": self.cdr_max},
            {"Variable": "Severity", "Mean": self.severity_mean, "StdDev": self.severity_std,
             "Min": self.severity_min, "Max": self.severity_max},
            {"Variable": "Recovery Lag", "Mean": self.recovery_lag_mean, "StdDev": self.recovery_lag_std,
             "Min": self.recovery_lag_min, "Max": self.recovery_lag_max},
        ])


@dataclass
class SampledPaths:
    """
    Output of Monte Carlo sampling: N paths of (CPR, CDR, Severity, RecoveryLag).

    This is the (N × 4) table that feeds into the engine runner.
    """
    cpr: np.ndarray        # shape (n_paths,)
    cdr: np.ndarray        # shape (n_paths,)
    severity: np.ndarray   # shape (n_paths,)
    recovery_lag: np.ndarray  # shape (n_paths,) — integer months

    @property
    def n_paths(self) -> int:
        return len(self.cpr)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "path_id": np.arange(self.n_paths),
            "cpr": self.cpr,
            "cdr": self.cdr,
            "severity": self.severity,
            "recovery_lag": self.recovery_lag,
        })

    def get_path(self, path_idx: int) -> dict:
        """Return assumptions for a single path as a dict."""
        return {
            "cpr": float(self.cpr[path_idx]),
            "cdr": float(self.cdr[path_idx]),
            "severity": float(self.severity[path_idx]),
            "recovery_lag": int(self.recovery_lag[path_idx]),
        }

    def summary(self) -> pd.DataFrame:
        """Percentile summary of sampled paths."""
        pcts = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        rows = []
        for name, arr in [("CPR", self.cpr), ("CDR", self.cdr),
                          ("Severity", self.severity), ("Recovery Lag", self.recovery_lag)]:
            row = {"Variable": name, "Mean": np.mean(arr), "Std": np.std(arr)}
            for p in pcts:
                row[f"P{int(p*100):02d}"] = np.percentile(arr, p * 100)
            rows.append(row)
        return pd.DataFrame(rows)


class MonteCarloSampler:
    """
    Generates N correlated assumption paths from distribution parameters.

    Usage:
        params = DistributionParams(cpr_mean=0.10, cpr_std=0.03, ...)
        sampler = MonteCarloSampler(params, n_paths=1000, seed=42)
        paths = sampler.sample()
        # paths.cpr → array of 1000 CPR values
        # paths.to_dataframe() → nice table
    """

    def __init__(
        self,
        params: DistributionParams,
        n_paths: int = 1000,
        seed: int = 42,
    ):
        self.params = params
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def sample(self) -> SampledPaths:
        """
        Generate N correlated assumption paths.

        Steps:
        1. Draw N samples from multivariate standard normal with correlation
        2. Transform each marginal from standard normal to target distribution
        3. Clip to bounds
        """
        p = self.params

        # Step 1: Correlated standard normal draws
        # Shape: (n_paths, 4) — columns are [CPR, CDR, Severity, RecoveryLag]
        corr = _ensure_positive_definite(p.corr)
        z = self.rng.multivariate_normal(
            mean=np.zeros(4),
            cov=corr,
            size=self.n_paths,
        )
        # z[:, 0] = correlated standard normal for CPR
        # z[:, 1] = correlated standard normal for CDR
        # z[:, 2] = correlated standard normal for Severity
        # z[:, 3] = correlated standard normal for Recovery Lag

        # Step 2: Transform marginals

        # CPR: Normal distribution, clip to bounds
        cpr = p.cpr_mean + p.cpr_std * z[:, 0]
        cpr = np.clip(cpr, p.cpr_min, p.cpr_max)

        # CDR: LogNormal transformation (skewed right — defaults can spike)
        # We use the correlated normal draw and transform it
        # LogNormal: if X ~ N(mu, sigma), then exp(X) ~ LogNormal
        # We want: E[CDR] = cdr_mean, so we solve for lognormal params
        cdr_var = p.cdr_std ** 2
        cdr_mu_sq = p.cdr_mean ** 2
        if cdr_var > 0 and cdr_mu_sq > 0:
            sigma_ln = np.sqrt(np.log(1 + cdr_var / cdr_mu_sq))
            mu_ln = np.log(p.cdr_mean) - 0.5 * sigma_ln ** 2
            cdr = np.exp(mu_ln + sigma_ln * z[:, 1])
        else:
            cdr = p.cdr_mean + p.cdr_std * z[:, 1]
        cdr = np.clip(cdr, p.cdr_min, p.cdr_max)

        # Severity: Normal for now (could use Beta), clip to [0, 1]
        severity = p.severity_mean + p.severity_std * z[:, 2]
        severity = np.clip(severity, p.severity_min, p.severity_max)

        # Recovery Lag: Normal, rounded to integer months
        recovery_lag = p.recovery_lag_mean + p.recovery_lag_std * z[:, 3]
        recovery_lag = np.clip(recovery_lag, p.recovery_lag_min, p.recovery_lag_max)
        recovery_lag = np.round(recovery_lag).astype(int)

        return SampledPaths(
            cpr=cpr,
            cdr=cdr,
            severity=severity,
            recovery_lag=recovery_lag,
        )
