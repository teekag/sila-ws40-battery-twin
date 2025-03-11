"""
Machine Learning package for Sila WS40 Battery Digital Twin.

This package provides ML models and training utilities for:
- Battery state prediction using LSTM
- Degradation estimation
- Uncertainty quantification using GPR
- Remaining useful life prediction
"""

from .models import BatteryPredictor, BatteryLSTM, BatteryGPR
from .training import BatteryModelTrainer

__version__ = "1.0.0"
