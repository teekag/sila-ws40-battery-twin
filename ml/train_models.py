"""
Train ML models for battery state prediction and degradation estimation.

This script trains the LSTM and GPR models using synthetic battery data
that mimics the behavior of silicon anode batteries like the Sila WS40.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.models import BatteryPredictor
from ml.training import BatteryModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir: str = "datasets/synthetic") -> pd.DataFrame:
    """Load synthetic battery data."""
    data_dir = Path(data_dir)
    data_file = data_dir / "synthetic_data.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    return pd.read_csv(data_file)

def prepare_data(df: pd.DataFrame) -> dict:
    """Prepare data for training."""
    # Group by cell and cycle
    features = {}
    
    for cell_id in df['cell_id'].unique():
        cell_data = df[df['cell_id'] == cell_id]
        
        # Get sequence data for each cycle
        cycle_data = []
        for cycle in cell_data['cycle'].unique():
            cycle_seq = cell_data[cell_data['cycle'] == cycle].sort_values('time')
            
            # Extract features
            seq_data = {
                'current': cycle_seq['current'].values,
                'voltage': cycle_seq['voltage'].values,
                'temperature': cycle_seq['temperature'].values,
                'soc': cycle_seq['soc'].values,
                'capacity': cycle_seq['capacity'].values,
                'soh': cycle_seq['soh'].values,
                'rul': cycle_seq['rul'].values,
                'stress': cycle_seq['stress'].values,
                'strain': cycle_seq['strain'].values
            }
            cycle_data.append(seq_data)
            
        features[cell_id] = cycle_data
        
    return features

def split_data(features: dict, test_size: float = 0.2, val_size: float = 0.1) -> tuple:
    """Split data into train, validation, and test sets."""
    # Split cells
    cell_ids = list(features.keys())
    train_cells, test_cells = train_test_split(cell_ids, test_size=test_size, random_state=42)
    train_cells, val_cells = train_test_split(train_cells, test_size=val_size/(1-test_size), random_state=42)
    
    # Combine sequences for each split
    splits = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for cell_id in train_cells:
        splits['train'][cell_id] = features[cell_id]
    
    for cell_id in val_cells:
        splits['val'][cell_id] = features[cell_id]
        
    for cell_id in test_cells:
        splits['test'][cell_id] = features[cell_id]
        
    return splits['train'], splits['val'], splits['test']

def prepare_sequences(data: dict, sequence_length: int = 100) -> dict:
    """Prepare sequences for training."""
    sequences = {
        'current': [],
        'voltage': [],
        'temperature': [],
        'soc': [],
        'capacity': [],
        'soh': [],
        'rul': []
    }
    
    # Combine sequences from all cells and cycles
    for cell_data in data.values():
        cell_sequences = {k: [] for k in sequences.keys()}
        
        # Concatenate all cycles for this cell
        for cycle_data in cell_data:
            for key in sequences:
                cell_sequences[key].extend(cycle_data[key])
                
        # Convert to numpy arrays
        cell_sequences = {k: np.array(v) for k, v in cell_sequences.items()}
        
        # Add to main sequences
        for key in sequences:
            sequences[key].append(cell_sequences[key])
    
    # Concatenate all cell sequences
    return {k: np.concatenate(v) for k, v in sequences.items()}

def train_models(
    train_data: dict,
    val_data: dict,
    model_dir: str,
    device: str = 'cpu',
    **kwargs
) -> BatteryPredictor:
    """Train LSTM and GPR models."""
    try:
        logger.info("Starting model training...")
        
        # Create model directory
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        predictor = BatteryPredictor(
            input_size=5,  # current, voltage, temperature, soc, capacity
            device=device
        )
        
        # Separate trainer params from training params
        trainer_params = {
            'learning_rate': kwargs.get('learning_rate', 1e-3),
            'batch_size': kwargs.get('batch_size', 32),
            'sequence_length': kwargs.get('sequence_length', 100)
        }
        
        training_params = {
            'n_epochs': kwargs.get('n_epochs', 100),
            'patience': kwargs.get('patience', 10)
        }
        
        # Initialize trainer
        trainer = BatteryModelTrainer(
            model=predictor,
            **trainer_params
        )
        
        # Prepare sequences
        train_sequences = prepare_sequences(train_data)
        val_sequences = prepare_sequences(val_data)
        
        # Train models
        history = trainer.train(
            train_sequences,
            val_sequences,
            model_dir=model_dir,
            **training_params
        )
        
        # Save training history
        history_file = model_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        logger.info(f"Training history saved to {history_file}")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def main():
    # Training parameters
    params = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'sequence_length': 20,  # Reduced from 100 to allow for more sequences
        'n_epochs': 100,
        'patience': 10
    }
    
    try:
        # Load data
        logger.info("Loading synthetic battery data...")
        df = load_data()
        
        # Prepare features
        logger.info("Preparing features...")
        features = prepare_data(df)
        
        # Split data
        logger.info("Splitting data...")
        train_data, val_data, test_data = split_data(features)
        
        # Train models
        logger.info("Training models...")
        model_dir = Path("models/battery")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        predictor = train_models(
            train_data,
            val_data,
            model_dir=model_dir,
            device=device,
            **params
        )
        
        # Save trained model
        model_path = model_dir / "battery_predictor.pt"
        predictor.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save parameters
        params_file = model_dir / "model_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"Parameters saved to {params_file}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
