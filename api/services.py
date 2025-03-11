from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from simulation.ecm_model import EquivalentCircuitModel
from simulation.thermal_model import ThermalModel
from simulation.mechanical_model import MechanicalModel
from .models import SensorData, BatteryState, SimulationConfig

logger = logging.getLogger(__name__)

class BatteryService:
    """Service for managing battery simulation and data processing."""
    
    def __init__(self):
        """Initialize battery service with simulation models."""
        self.ecm = EquivalentCircuitModel()
        self.thermal = ThermalModel()
        self.mechanical = MechanicalModel()
        
        # Initialize data storage
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Track simulation state
        self.cycle_count = 0
        self.total_charge_throughput = 0.0
        
    def process_sensor_data(self, data: SensorData) -> BatteryState:
        """Process incoming sensor data and update battery state."""
        try:
            # Update charge throughput
            self.total_charge_throughput += abs(data.current)
            
            # Update cycle count (assuming 1 cycle = full capacity throughput)
            nominal_capacity = self.ecm.params.nominal_capacity * 3600  # Convert to Coulombs
            self.cycle_count = int(self.total_charge_throughput / (2 * nominal_capacity))
            
            # Update models
            voltage, soc, ecm_temp = self.ecm.update(
                current=data.current,
                dt=1.0,  # 1-second update
                ambient_temp=data.temperature
            )
            
            temp_avg, temp_max, temp_min = self.thermal.update(
                current=data.current,
                voltage=voltage,
                internal_resistance=self.ecm.params.r_internal,
                ambient_temp=data.temperature
            )
            
            mech_state = self.mechanical.update(
                soc=soc,
                temperature=temp_avg
            )
            
            # Calculate SOH based on capacity fade
            # Simple linear model for now
            cycle_life = 1000  # Expected cycle life
            soh = max(0.0, 1.0 - (self.cycle_count / cycle_life))
            
            return BatteryState(
                timestamp=data.timestamp,
                voltage=voltage,
                current=data.current,
                soc=soc,
                internal_resistance=self.ecm.params.r_internal,
                temperature_avg=temp_avg,
                temperature_max=temp_max,
                temperature_min=temp_min,
                cooling_active=False,
                volume_change=mech_state['volume_change'],
                von_mises_stress=mech_state['von_mises_stress'],
                max_principal_stress=mech_state['max_principal_stress'],
                safety_factor=mech_state['safety_factor'],
                cycle_count=self.cycle_count,
                soh=soh
            )
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {str(e)}")
            raise
            
    def run_simulation(self, config: SimulationConfig) -> List[BatteryState]:
        """Run battery simulation with given configuration."""
        try:
            results = []
            
            # Reset models to initial conditions
            self.ecm.soc = config.initial_soc
            self.thermal.T.fill(config.ambient_temp)
            self.mechanical.reset_state()
            
            # Calculate timesteps
            total_steps = int(config.duration_hours * 3600 / config.sampling_rate_sec)
            
            for step in range(total_steps):
                # Get current from profile (cycle if needed)
                current_idx = step % len(config.current_profile)
                current = config.current_profile[current_idx]
                
                # Create sensor data
                sensor_data = SensorData(
                    timestamp=datetime.now() + timedelta(seconds=step*config.sampling_rate_sec),
                    voltage=self.ecm.params.nominal_voltage,  # Will be updated
                    current=current,
                    temperature=config.ambient_temp,
                    strain=0.0,  # Will be updated
                    soc=self.ecm.soc
                )
                
                # Process data
                state = self.process_sensor_data(sensor_data)
                results.append(state)
                
            return results
            
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            raise
            
    def calculate_degradation_metrics(self) -> Dict[str, float]:
        """Calculate battery degradation metrics."""
        try:
            # Calculate capacity fade
            initial_capacity = self.ecm.params.nominal_capacity
            current_capacity = initial_capacity * (1.0 - self.cycle_count * 0.0002)  # Simple linear model
            
            # Calculate power fade (increase in internal resistance)
            initial_resistance = self.ecm.params.r_internal
            current_resistance = initial_resistance * (1.0 + self.cycle_count * 0.0001)
            
            # Calculate mechanical degradation
            safety_metrics = self.mechanical.get_safety_metrics()
            
            return {
                "capacity_fade_percent": (1 - current_capacity/initial_capacity) * 100,
                "power_fade_percent": (current_resistance/initial_resistance - 1) * 100,
                "mechanical_stress_ratio": safety_metrics["stress_ratio"],
                "safety_factor": safety_metrics["safety_factor"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating degradation metrics: {str(e)}")
            raise
            
    def get_state_history(self, hours: Optional[float] = None) -> List[BatteryState]:
        """Get battery state history."""
        try:
            history_file = self.processed_dir / "state_history.json"
            if not history_file.exists():
                return []
                
            df = pd.read_json(history_file)
            
            if hours is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                df = df[df["timestamp"] >= cutoff_time]
                
            return [BatteryState(**row) for row in df.to_dict("records")]
            
        except Exception as e:
            logger.error(f"Error retrieving state history: {str(e)}")
            raise

    def get_safety_status(self) -> Dict[str, bool]:
        """Get battery safety status."""
        try:
            # Get current state
            safety_metrics = self.mechanical.get_safety_metrics()
            temp_avg = np.mean(self.thermal.T)
            
            return {
                "temperature_safe": self.thermal.params.temp_min <= temp_avg <= self.thermal.params.temp_max,
                "stress_safe": safety_metrics["stress_ratio"] < 1.0,
                "strain_safe": safety_metrics["strain_ratio"] < 1.0,
                "voltage_safe": self.ecm.params.voltage_min <= self.ecm.params.nominal_voltage <= self.ecm.params.voltage_max
            }
            
        except Exception as e:
            logger.error(f"Error checking safety status: {str(e)}")
            raise
