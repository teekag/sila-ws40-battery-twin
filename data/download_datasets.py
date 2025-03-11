"""
Dataset downloader and preprocessor for battery digital twin training.

This script downloads and preprocesses battery datasets from:
1. MIT Battery Degradation Dataset (data.matr.io)
2. NASA Battery Data
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from scipy.interpolate import interp1d
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatteryDatasetDownloader:
    """Downloader and preprocessor for battery datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_mit_data(self):
        """
        Download MIT Battery Degradation Dataset.
        Source: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
        """
        logger.info("Downloading MIT Battery Dataset...")
        
        # Dataset URLs
        urls = {
            'cycles': 'https://data.matr.io/1/api/v1/file/5c86bd79c625d700019f3cd2/download',
            'summary': 'https://data.matr.io/1/api/v1/file/5c86bd73c625d700019f3cd1/download'
        }
        
        try:
            # Download files
            for name, url in urls.items():
                output_file = self.raw_dir / f"mit_{name}.csv"
                if not output_file.exists():
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded {name} data to {output_file}")
                else:
                    logger.info(f"{name} data already exists at {output_file}")
                    
        except Exception as e:
            logger.error(f"Error downloading MIT data: {str(e)}")
            raise
            
    def download_nasa_data(self):
        """
        Download NASA Battery Data.
        Source: https://www.nasa.gov/content/battery-data-set
        Note: Manual download required. This function processes local files.
        """
        logger.info("NASA Battery Dataset requires manual download.")
        logger.info("Please download the data from: https://www.nasa.gov/content/battery-data-set")
        logger.info(f"Place the files in: {self.raw_dir / 'nasa'}")
        
    def preprocess_mit_data(self) -> pd.DataFrame:
        """Preprocess MIT Battery Dataset."""
        try:
            # Load data
            cycles_file = self.raw_dir / "mit_cycles.csv"
            summary_file = self.raw_dir / "mit_summary.csv"
            
            if not (cycles_file.exists() and summary_file.exists()):
                raise FileNotFoundError("MIT dataset files not found. Run download_mit_data() first.")
                
            cycles_data = pd.read_csv(cycles_file)
            summary_data = pd.read_csv(summary_file)
            
            # Process cycle data
            processed_data = []
            
            for battery_id in cycles_data['battery'].unique():
                battery_cycles = cycles_data[cycles_data['battery'] == battery_id]
                battery_summary = summary_data[summary_data['battery'] == battery_id]
                
                for _, cycle in battery_cycles.iterrows():
                    # Extract time series data
                    V = np.array(json.loads(cycle['V']))
                    I = np.array(json.loads(cycle['I']))
                    T = np.array(json.loads(cycle['T']))
                    t = np.array(json.loads(cycle['t']))
                    
                    # Resample to uniform time grid (1-second intervals)
                    t_uniform = np.linspace(t[0], t[-1], num=int(t[-1]-t[0]))
                    
                    V_interp = interp1d(t, V, kind='linear')(t_uniform)
                    I_interp = interp1d(t, I, kind='linear')(t_uniform)
                    T_interp = interp1d(t, T, kind='linear')(t_uniform)
                    
                    # Calculate derived features
                    cycle_idx = cycle['cycle_index']
                    capacity = cycle['Qd']  # Discharge capacity
                    soh = capacity / battery_summary['QD'].iloc[0]  # Normalize by initial capacity
                    
                    # Create samples
                    for i in range(len(t_uniform)):
                        processed_data.append({
                            'battery_id': battery_id,
                            'cycle': cycle_idx,
                            'time': t_uniform[i],
                            'voltage': V_interp[i],
                            'current': I_interp[i],
                            'temperature': T_interp[i],
                            'capacity': capacity,
                            'soh': soh,
                            'rul': max(0, 1000 - cycle_idx)  # Simplified RUL based on cycle count
                        })
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
            
            # Save processed data
            output_file = self.processed_dir / "mit_processed.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed MIT data to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing MIT data: {str(e)}")
            raise
            
    def preprocess_nasa_data(self) -> pd.DataFrame:
        """
        Preprocess NASA Battery Data.
        Requires manual download of data files.
        """
        try:
            nasa_dir = self.raw_dir / "nasa"
            if not nasa_dir.exists():
                raise FileNotFoundError(
                    f"NASA data directory not found at {nasa_dir}. "
                    "Please download the data manually."
                )
                
            processed_data = []
            
            # Process each battery directory
            for battery_dir in nasa_dir.glob("B*"):
                battery_id = battery_dir.name
                
                # Process each measurement file
                for data_file in battery_dir.glob("*.txt"):
                    data = pd.read_csv(
                        data_file,
                        delimiter="\t",
                        names=['time', 'voltage', 'current', 'temperature']
                    )
                    
                    cycle = int(data_file.stem.split("_")[1])
                    
                    # Calculate capacity and SOH
                    discharge_mask = data['current'] < 0
                    capacity = abs(np.trapz(
                        y=data.loc[discharge_mask, 'current'],
                        x=data.loc[discharge_mask, 'time']
                    )) / 3600  # Convert to Ah
                    
                    # Get initial capacity for SOH calculation
                    initial_file = next(battery_dir.glob("*_0.txt"))
                    initial_data = pd.read_csv(
                        initial_file,
                        delimiter="\t",
                        names=['time', 'voltage', 'current', 'temperature']
                    )
                    initial_capacity = abs(np.trapz(
                        y=initial_data.loc[initial_data['current'] < 0, 'current'],
                        x=initial_data.loc[initial_data['current'] < 0, 'time']
                    )) / 3600
                    
                    soh = capacity / initial_capacity
                    
                    # Add to processed data
                    for _, row in data.iterrows():
                        processed_data.append({
                            'battery_id': battery_id,
                            'cycle': cycle,
                            'time': row['time'],
                            'voltage': row['voltage'],
                            'current': row['current'],
                            'temperature': row['temperature'],
                            'capacity': capacity,
                            'soh': soh,
                            'rul': max(0, 2000 - cycle)  # Simplified RUL based on cycle count
                        })
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
            
            # Save processed data
            output_file = self.processed_dir / "nasa_processed.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed NASA data to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing NASA data: {str(e)}")
            raise
            
    def combine_datasets(self) -> pd.DataFrame:
        """Combine and normalize all processed datasets."""
        try:
            datasets = []
            
            # Load MIT data if available
            mit_file = self.processed_dir / "mit_processed.csv"
            if mit_file.exists():
                mit_data = pd.read_csv(mit_file)
                mit_data['source'] = 'MIT'
                datasets.append(mit_data)
                
            # Load NASA data if available
            nasa_file = self.processed_dir / "nasa_processed.csv"
            if nasa_file.exists():
                nasa_data = pd.read_csv(nasa_file)
                nasa_data['source'] = 'NASA'
                datasets.append(nasa_data)
                
            if not datasets:
                raise FileNotFoundError("No processed datasets found")
                
            # Combine datasets
            combined_df = pd.concat(datasets, ignore_index=True)
            
            # Normalize numerical columns
            for col in ['voltage', 'current', 'temperature', 'capacity']:
                mean = combined_df[col].mean()
                std = combined_df[col].std()
                combined_df[f'{col}_normalized'] = (combined_df[col] - mean) / std
                
            # Save combined dataset
            output_file = self.processed_dir / "combined_dataset.csv"
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved combined dataset to {output_file}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize downloader
    downloader = BatteryDatasetDownloader()
    
    # Download datasets
    downloader.download_mit_data()
    downloader.download_nasa_data()  # Will prompt for manual download
    
    # Preprocess datasets
    mit_data = downloader.preprocess_mit_data()
    nasa_data = downloader.preprocess_nasa_data()
    
    # Combine datasets
    combined_data = downloader.combine_datasets()
