"""
Core package — schema definitions, configuration, and shared utilities.
No business logic lives here.
"""

from .schema import CES1_TAPE_COLUMNS
from .config import ProjectionConfig
from .utils import require_columns, annual_to_monthly_hazard, month_starts

__all__ = [
    "CES1_TAPE_COLUMNS",
    "ProjectionConfig",
    "require_columns",
    "annual_to_monthly_hazard",
    "month_starts",
]
