"""
ML models for battery state prediction and degradation estimation.

This module provides:
- LSTM model for sequence-based predictions
- GPR model for uncertainty quantification
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BatteryLSTM(nn.Module):
    """LSTM model for battery state prediction and degradation estimation."""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers for different predictions
        self.fc_soc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.fc_voltage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc_temperature = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc_soh = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.fc_rul = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # RUL should be non-negative
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the network."""
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out shape: [batch, seq_len, hidden_size]
        
        # Get predictions from the last timestep
        last_hidden = lstm_out[:, -1, :]  # shape: [batch, hidden_size]
        
        # Get predictions for each output
        soc = self.fc_soc(last_hidden)
        voltage = self.fc_voltage(last_hidden)
        temperature = self.fc_temperature(last_hidden)
        soh = self.fc_soh(last_hidden)
        rul = self.fc_rul(last_hidden)
        
        return {
            'soc': soc,
            'voltage': voltage,
            'temperature': temperature,
            'soh': soh,
            'rul': rul,
            'hidden': hidden
        }
        
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

class BatteryGPR:
    """Gaussian Process Regression for uncertainty quantification in battery predictions."""
    
    def __init__(self):
        # Define kernels for different prediction tasks
        self.kernels = {
            'soc': C(1.0) * RBF([1.0]),
            'voltage': C(1.0) * RBF([1.0]),
            'temperature': C(1.0) * RBF([1.0]),
            'soh': C(1.0) * RBF([1.0])
        }
        
        # Initialize GPR models
        self.models = {
            target: GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            for target, kernel in self.kernels.items()
        }
        
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Fit GPR models for each target variable."""
        try:
            for target, model in self.models.items():
                if target in y:
                    model.fit(X, y[target])
            logger.info("Successfully fitted GPR models")
        except Exception as e:
            logger.error(f"Error fitting GPR models: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with uncertainty estimates."""
        predictions = {}
        
        try:
            for target, model in self.models.items():
                mean, std = model.predict(X, return_std=True)
                predictions[target] = (mean, std)
            return predictions
        except Exception as e:
            logger.error(f"Error making GPR predictions: {str(e)}")
            raise

class BatteryPredictor:
    """Combined predictor using LSTM and GPR for battery state prediction."""
    
    def __init__(self, input_size: int = 5, device: str = 'cpu'):
        self.device = device
        self.lstm = BatteryLSTM(input_size=input_size).to(device)
        self.gpr = BatteryGPR()
        self.scaler = None  # Will be set during training
        
    def preprocess_data(self, data: Dict[str, np.ndarray]) -> torch.Tensor:
        """Preprocess input data for LSTM."""
        # Combine features
        features = np.stack([
            data['current'],
            data['voltage'],
            data['temperature'],
            data['soc'],
            data.get('capacity', np.ones_like(data['current']))  # Optional capacity
        ], axis=-1)
        
        # Convert to tensor
        return torch.FloatTensor(features).to(self.device)
        
    def predict(self, 
               input_data: Dict[str, np.ndarray],
               horizon: int,
               include_uncertainty: bool = True
               ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Make predictions with uncertainty estimates."""
        try:
            # Preprocess data
            x = self.preprocess_data(input_data)
            x = x.unsqueeze(0)  # Add batch dimension
            
            # Initialize LSTM hidden state
            hidden = self.lstm.init_hidden(batch_size=1, device=self.device)
            
            # Make LSTM predictions
            with torch.no_grad():
                lstm_preds = self.lstm(x, hidden)
            
            # Convert predictions to numpy
            predictions = {
                k: v.cpu().numpy().squeeze()
                for k, v in lstm_preds.items()
                if k != 'hidden'
            }
            
            # Get uncertainty estimates if requested
            if include_uncertainty:
                # Use last few timesteps for GPR
                recent_data = x[:, -10:, :].cpu().numpy()
                gpr_preds = self.gpr.predict(recent_data.reshape(-1, x.shape[-1]))
                
                # Combine LSTM predictions with GPR uncertainty
                return {
                    k: (predictions[k], gpr_preds.get(k, (None, None))[1])
                    for k in predictions.keys()
                }
            
            return {k: (v, None) for k, v in predictions.items()}
            
        except Exception as e:
            logger.error(f"Error in battery prediction: {str(e)}")
            raise
            
    def save_model(self, path: str):
        """Save model weights and parameters."""
        try:
            torch.save({
                'lstm_state_dict': self.lstm.state_dict(),
                'device': self.device
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str):
        """Load model weights and parameters."""
        try:
            checkpoint = torch.load(path)
            self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
            self.device = checkpoint['device']
            self.lstm.to(self.device)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
