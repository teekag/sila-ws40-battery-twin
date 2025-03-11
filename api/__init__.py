"""
Sila WS40 Battery Digital Twin API

This package provides the FastAPI backend for the battery digital twin,
including real-time simulation, data processing, and predictive analytics.
"""

from .models import (
    SensorData,
    BatteryState,
    SimulationConfig,
    PredictionRequest,
    PredictionResponse
)
from .services import BatteryService

__version__ = "1.0.0"
