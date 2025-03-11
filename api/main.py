from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from .models import (
    SensorData, BatteryState, SimulationConfig,
    PredictionRequest, PredictionResponse
)
from simulation.ecm_model import EquivalentCircuitModel
from simulation.thermal_model import ThermalModel
from simulation.mechanical_model import MechanicalModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sila WS40 Battery Digital Twin",
    description="Digital twin for Sila Nanotechnologies WS40 silicon anode lithium-ion battery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
ecm_model = EquivalentCircuitModel()
thermal_model = ThermalModel()
mechanical_model = MechanicalModel()

# Data storage
DATA_DIR = Path("data")
SENSOR_DATA_FILE = DATA_DIR / "raw" / "sensor_data.json"
STATE_HISTORY_FILE = DATA_DIR / "processed" / "state_history.json"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)

def load_data(file_path: Path) -> List[Dict]:
    """Load data from JSON file."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_data(data: List[Dict], file_path: Path):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Sila WS40 Battery Digital Twin API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/sensor-data", response_model=BatteryState)
async def process_sensor_data(data: SensorData, background_tasks: BackgroundTasks):
    """Process incoming sensor data and update battery state."""
    try:
        # Update ECM
        voltage, soc, ecm_temp = ecm_model.update(
            current=data.current,
            dt=1.0
        )
        
        # Update thermal model
        temp_avg, temp_max, temp_min = thermal_model.update(
            current=data.current,
            voltage=voltage,
            internal_resistance=ecm_model.params.r_internal,
            ambient_temp=25.0
        )
        
        # Update mechanical model
        mech_state = mechanical_model.update(
            soc=soc,
            temperature=temp_avg
        )
        
        # Create battery state
        state = BatteryState(
            timestamp=data.timestamp,
            voltage=voltage,
            current=data.current,
            soc=soc,
            internal_resistance=ecm_model.params.r_internal,
            temperature_avg=temp_avg,
            temperature_max=temp_max,
            temperature_min=temp_min,
            cooling_active=False,
            volume_change=mech_state['volume_change'],
            von_mises_stress=mech_state['von_mises_stress'],
            max_principal_stress=mech_state['max_principal_stress'],
            safety_factor=mech_state['safety_factor'],
            cycle_count=ecm_model.cycle_count,
            soh=1.0  # TODO: Implement SOH estimation
        )
        
        # Save data asynchronously
        background_tasks.add_task(save_sensor_data, data)
        background_tasks.add_task(save_battery_state, state)
        
        return state
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate", response_model=List[BatteryState])
async def run_simulation(config: SimulationConfig):
    """Run battery simulation with specified configuration."""
    try:
        # Calculate number of steps
        total_steps = int(config.duration_hours * 3600 / config.sampling_rate_sec)
        
        # Initialize results
        results = []
        
        # Reset models to initial conditions
        ecm_model.soc = config.initial_soc
        thermal_model.T.fill(config.ambient_temp)
        mechanical_model.reset_state()
        
        # Run simulation
        for step in range(total_steps):
            # Get current from profile (cycle if needed)
            current_idx = step % len(config.current_profile)
            current = config.current_profile[current_idx]
            
            # Update models
            voltage, soc, ecm_temp = ecm_model.update(
                current=current,
                dt=config.sampling_rate_sec
            )
            
            temp_avg, temp_max, temp_min = thermal_model.update(
                current=current,
                voltage=voltage,
                internal_resistance=ecm_model.params.r_internal,
                ambient_temp=config.ambient_temp,
                cooling_active=config.cooling_enabled
            )
            
            mech_state = mechanical_model.update(
                soc=soc,
                temperature=temp_avg
            )
            
            # Create state
            state = BatteryState(
                timestamp=datetime.now() + timedelta(seconds=step*config.sampling_rate_sec),
                voltage=voltage,
                current=current,
                soc=soc,
                internal_resistance=ecm_model.params.r_internal,
                temperature_avg=temp_avg,
                temperature_max=temp_max,
                temperature_min=temp_min,
                cooling_active=config.cooling_enabled,
                volume_change=mech_state['volume_change'],
                von_mises_stress=mech_state['von_mises_stress'],
                max_principal_stress=mech_state['max_principal_stress'],
                safety_factor=mech_state['safety_factor'],
                cycle_count=ecm_model.cycle_count,
                soh=1.0  # TODO: Implement SOH estimation
            )
            
            results.append(state)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_battery_behavior(request: PredictionRequest):
    """Predict future battery behavior."""
    try:
        # Calculate number of prediction steps
        steps = int(request.horizon_hours * 3600)
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(0, steps, 60)  # 1-minute intervals
        ]
        
        # For now, return simple predictions
        # TODO: Implement ML models for actual predictions
        soc_prediction = [
            max(0.0, ecm_model.soc - i * 0.001) 
            for i in range(len(timestamps))
        ]
        
        voltage_prediction = [
            3.7 - (1 - soc) * 0.5 
            for soc in soc_prediction
        ]
        
        temperature_prediction = [
            np.mean(thermal_model.T) + np.random.normal(0, 0.1)
            for _ in range(len(timestamps))
        ]
        
        # Simple uncertainty bounds (±5%)
        if request.include_uncertainty:
            uncertainty = {
                "soc_bounds": [0.05] * len(timestamps),
                "voltage_bounds": [0.1] * len(timestamps),
                "temperature_bounds": [2.0] * len(timestamps)
            }
        else:
            uncertainty = None
        
        return PredictionResponse(
            timestamps=timestamps,
            soc_prediction=soc_prediction,
            voltage_prediction=voltage_prediction,
            temperature_prediction=temperature_prediction,
            soh_prediction=0.95,  # Placeholder
            rul_hours=1000.0,     # Placeholder
            uncertainty=uncertainty
        )
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_sensor_data(data: SensorData):
    """Save sensor data to file."""
    sensor_data = load_data(SENSOR_DATA_FILE)
    sensor_data.append(data.dict())
    save_data(sensor_data, SENSOR_DATA_FILE)

async def save_battery_state(state: BatteryState):
    """Save battery state to file."""
    state_history = load_data(STATE_HISTORY_FILE)
    state_history.append(state.dict())
    save_data(state_history, STATE_HISTORY_FILE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
