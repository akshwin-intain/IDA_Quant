"""
PM (Portfolio Manager) outputs — metrics, aggregation, and decision support.
"""

from .metrics import compute_path_metrics
from .aggregator import aggregate_path_results
from .decisions import generate_decision_report

__all__ = [
    "compute_path_metrics",
    "aggregate_path_results",
    "generate_decision_report",
]
