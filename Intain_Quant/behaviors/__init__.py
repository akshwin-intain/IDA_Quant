"""
Behavioral models — convert assumptions into loan-level monthly hazard forecasts.
"""

from .base import HazardForecast, BehaviorModel
from .constant import ConstantHazardModel
from .scenario import ScenarioHazardModel

__all__ = [
    "HazardForecast",
    "BehaviorModel",
    "ConstantHazardModel",
    "ScenarioHazardModel",
]
