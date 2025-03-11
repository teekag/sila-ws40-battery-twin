import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from .models import BatteryPredictor

logger = logging.getLogger(__name__)

class BatteryModelTrainer:
    """Trainer for battery prediction models."""
    
    def __init__(self, 
                 model: BatteryPredictor,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 sequence_length: int = 100):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.lstm.parameters(),
            lr=learning_rate
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def prepare_sequences(self, 
                         data: Dict[str, np.ndarray],
                         sequence_length: int
                         ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare sequential data for training."""
        # Get data dimensions
        n_samples = len(data['current'])
        n_sequences = n_samples - sequence_length
        
        # Prepare input sequences
        X = []
        y = {k: [] for k in ['soc', 'voltage', 'temperature', 'soh', 'rul']}
        
        for i in range(n_sequences):
            # Input sequence
            seq = np.stack([
                data['current'][i:i+sequence_length],
                data['voltage'][i:i+sequence_length],
                data['temperature'][i:i+sequence_length],
                data['soc'][i:i+sequence_length],
                data.get('capacity', np.ones(n_samples))[i:i+sequence_length]
            ], axis=-1)
            X.append(seq)
            
            # Target values (next timestep)
            y['soc'].append(data['soc'][i+sequence_length])
            y['voltage'].append(data['voltage'][i+sequence_length])
            y['temperature'].append(data['temperature'][i+sequence_length])
            y['soh'].append(data.get('soh', np.ones(n_samples))[i+sequence_length])
            y['rul'].append(data.get('rul', np.zeros(n_samples))[i+sequence_length])
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(X)).to(self.model.device)
        y = {k: torch.FloatTensor(np.array(v)).to(self.model.device)
             for k, v in y.items()}
        
        return X, y
        
    def train_epoch(self, 
                    train_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
                    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.lstm.train()
        total_loss = 0
        losses = {k: 0.0 for k in ['soc', 'voltage', 'temperature', 'soh', 'rul']}
        
        X, y = train_data
        n_batches = len(X) // self.batch_size
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            
            batch_X = X[start_idx:end_idx]
            batch_y = {k: v[start_idx:end_idx] for k, v in y.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model.lstm(batch_X)
            
            # Calculate losses
            loss = 0
            for k in losses.keys():
                if k in ['soc', 'soh']:
                    # Binary cross entropy for bounded [0,1] values
                    k_loss = self.bce_loss(
                        predictions[k].squeeze(),
                        batch_y[k]
                    )
                else:
                    # MSE for unbounded values
                    k_loss = self.mse_loss(
                        predictions[k].squeeze(),
                        batch_y[k]
                    )
                losses[k] += k_loss.item()
                loss += k_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # Average losses
        n_samples = n_batches * self.batch_size
        return {
            'total_loss': total_loss / n_samples,
            **{k: v / n_samples for k, v in losses.items()}
        }
        
    def train(self,
              train_data: Dict[str, np.ndarray],
              val_data: Optional[Dict[str, np.ndarray]] = None,
              n_epochs: int = 100,
              patience: int = 10,
              model_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the model."""
        try:
            logger.info("Starting model training...")
            
            # Prepare data
            train_sequences = self.prepare_sequences(
                train_data,
                self.sequence_length
            )
            
            if val_data is not None:
                val_sequences = self.prepare_sequences(
                    val_data,
                    self.sequence_length
                )
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [] if val_data is not None else None
            }
            
            # Early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            for epoch in range(n_epochs):
                # Train
                train_losses = self.train_epoch(train_sequences)
                history['train_loss'].append(train_losses['total_loss'])
                
                # Validate
                if val_data is not None:
                    val_losses = self.validate(val_sequences)
                    history['val_loss'].append(val_losses['total_loss'])
                    
                    # Early stopping check
                    if val_losses['total_loss'] < best_val_loss:
                        best_val_loss = val_losses['total_loss']
                        patience_counter = 0
                        
                        # Save best model
                        if model_dir is not None:
                            self.save_checkpoint(
                                Path(model_dir) / 'best_model.pt',
                                epoch,
                                val_losses['total_loss']
                            )
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{n_epochs} - "
                        f"Train Loss: {train_losses['total_loss']:.4f}"
                        + (f" - Val Loss: {val_losses['total_loss']:.4f}"
                           if val_data is not None else "")
                    )
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def validate(self,
                val_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
                ) -> Dict[str, float]:
        """Validate the model."""
        self.model.lstm.eval()
        
        with torch.no_grad():
            X, y = val_data
            predictions = self.model.lstm(X)
            
            losses = {}
            total_loss = 0
            
            # Calculate losses
            for k in y.keys():
                if k in ['soc', 'soh']:
                    loss = self.bce_loss(
                        predictions[k].squeeze(),
                        y[k]
                    ).item()
                else:
                    loss = self.mse_loss(
                        predictions[k].squeeze(),
                        y[k]
                    ).item()
                losses[k] = loss
                total_loss += loss
            
            losses['total_loss'] = total_loss / len(y)
            
        return losses
        
    def save_checkpoint(self,
                       path: Path,
                       epoch: int,
                       val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.lstm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.model.lstm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
