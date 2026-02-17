"""
Compute and manage correlation structure between behavioral assumptions.

WHY THIS MATTERS:
Without correlations, Monte Carlo can generate nonsense paths like:
  CPR=15% + CDR=10% + Severity=70%
  (everyone refinancing AND defaulting AND losing everything — impossible)

With correlations, paths make economic sense:
  Bad economy:  low CPR + high CDR + high severity + long recovery
  Good economy: high CPR + low CDR + low severity + short recovery

Sources for correlation estimates:
  1. Compute from historical data (when enough months available)
  2. Use industry benchmark correlations (always available)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Industry benchmark correlation matrix for CES/RMBS
# Order: [CPR, CDR, Severity, Recovery Lag]
#
# Rationale:
#   CPR ↔ CDR:      -0.60  (bad economy → less refi, more defaults)
#   CPR ↔ Severity: -0.40  (bad economy → less refi, higher losses)
#   CPR ↔ RecovLag: -0.20  (bad economy → less refi, slower recovery)
#   CDR ↔ Severity: +0.70  (defaults spike with housing crash → deeper losses)
#   CDR ↔ RecovLag: +0.50  (more defaults → courts/servicers overwhelmed → slower)
#   Severity ↔ RecovLag: +0.40  (deeper losses → more complex workouts → slower)

BENCHMARK_CORRELATION_MATRIX = np.array([
    # CPR    CDR    Sev    Lag
    [ 1.00, -0.60, -0.40, -0.20],  # CPR
    [-0.60,  1.00,  0.70,  0.50],  # CDR
    [-0.40,  0.70,  1.00,  0.40],  # Severity
    [-0.20,  0.50,  0.40,  1.00],  # Recovery Lag
])

VARIABLE_ORDER = ["cpr", "cdr", "severity", "recovery_lag"]


def _ensure_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Force a correlation matrix to be positive semi-definite.
    
    Historical correlations from small samples can sometimes produce
    matrices that aren't valid for multivariate sampling. This uses
    eigenvalue clipping to fix that.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip negative eigenvalues to small positive number
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-normalize to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def compute_correlation_matrix(
    historical_distributions: Dict,
    *,
    min_common_months: int = 6,
) -> np.ndarray:
    """
    Compute correlation matrix from historical CPR/CDR time series.

    Parameters
    ----------
    historical_distributions : dict
        Output of distributions.historical.compute_historical_distributions()
        Keys: "cpr", "cdr", "severity" with .monthly_series attribute
    min_common_months : int
        Minimum overlapping months needed to estimate correlations.
        If fewer, falls back to benchmark correlations.

    Returns
    -------
    4×4 numpy array (order: CPR, CDR, Severity, RecoveryLag)
    """
    # Try to build correlation from historical CPR and CDR series
    cpr_series = historical_distributions.get("cpr")
    cdr_series = historical_distributions.get("cdr")

    if (
        cpr_series is None
        or cdr_series is None
        or len(cpr_series.monthly_series) < min_common_months
        or len(cdr_series.monthly_series) < min_common_months
    ):
        # Not enough data — use benchmark correlations
        return BENCHMARK_CORRELATION_MATRIX.copy()

    # Align the two series by length (they should be from same deal)
    n = min(len(cpr_series.monthly_series), len(cdr_series.monthly_series))
    cpr_vals = cpr_series.monthly_series.values[:n]
    cdr_vals = cdr_series.monthly_series.values[:n]

    # Compute CPR-CDR correlation from data
    if np.std(cpr_vals) > 0 and np.std(cdr_vals) > 0:
        corr_cpr_cdr = float(np.corrcoef(cpr_vals, cdr_vals)[0, 1])
        # Clamp to reasonable range
        corr_cpr_cdr = np.clip(corr_cpr_cdr, -0.95, 0.95)
    else:
        corr_cpr_cdr = BENCHMARK_CORRELATION_MATRIX[0, 1]  # fallback

    # Build full matrix: use observed CPR-CDR, benchmark for the rest
    # (Severity and RecoveryLag correlations need more data than we typically have)
    matrix = BENCHMARK_CORRELATION_MATRIX.copy()
    matrix[0, 1] = corr_cpr_cdr
    matrix[1, 0] = corr_cpr_cdr

    # Ensure the blended matrix is still valid
    matrix = _ensure_positive_definite(matrix)

    return matrix


def get_benchmark_correlation_matrix() -> np.ndarray:
    """Return the industry benchmark correlation matrix."""
    return BENCHMARK_CORRELATION_MATRIX.copy()


def correlation_matrix_to_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    """Convert correlation matrix to a labeled DataFrame for display."""
    labels = ["CPR", "CDR", "Severity", "Recovery Lag"]
    return pd.DataFrame(matrix, index=labels, columns=labels)
