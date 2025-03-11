from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class SensorData(BaseModel):
    """Battery sensor data model."""
    timestamp: datetime = Field(..., description="Measurement timestamp")
    voltage: float = Field(..., description="Terminal voltage (V)")
    current: float = Field(..., description="Applied current (A)")
    temperature: float = Field(..., description="Average temperature (°C)")
    strain: float = Field(..., description="Mechanical strain")
    soc: float = Field(..., description="State of charge (0-1)")
    
class BatteryState(BaseModel):
    """Complete battery state including all physics-based models."""
    timestamp: datetime = Field(..., description="State timestamp")
    # Electrical state
    voltage: float = Field(..., description="Terminal voltage (V)")
    current: float = Field(..., description="Applied current (A)")
    soc: float = Field(..., description="State of charge (0-1)")
    internal_resistance: float = Field(..., description="Internal resistance (Ω)")
    
    # Thermal state
    temperature_avg: float = Field(..., description="Average temperature (°C)")
    temperature_max: float = Field(..., description="Maximum temperature (°C)")
    temperature_min: float = Field(..., description="Minimum temperature (°C)")
    cooling_active: bool = Field(..., description="Cooling system status")
    
    # Mechanical state
    volume_change: float = Field(..., description="Volume change (%)")
    von_mises_stress: float = Field(..., description="Von Mises stress (Pa)")
    max_principal_stress: float = Field(..., description="Maximum principal stress (Pa)")
    safety_factor: float = Field(..., description="Mechanical safety factor")
    
    # Health metrics
    cycle_count: int = Field(..., description="Battery cycle count")
    soh: float = Field(..., description="State of health (0-1)")
    
class SimulationConfig(BaseModel):
    """Configuration for battery simulation."""
    duration_hours: float = Field(..., description="Simulation duration in hours")
    sampling_rate_sec: int = Field(..., description="Sampling rate in seconds")
    initial_soc: float = Field(1.0, description="Initial state of charge")
    ambient_temp: float = Field(25.0, description="Ambient temperature (°C)")
    current_profile: List[float] = Field(..., description="Current profile (A)")
    cooling_enabled: bool = Field(False, description="Enable active cooling")

class PredictionRequest(BaseModel):
    """Request model for battery predictions."""
    horizon_hours: float = Field(..., description="Prediction horizon in hours")
    current_profile: Optional[List[float]] = Field(None, description="Future current profile")
    include_uncertainty: bool = Field(True, description="Include prediction uncertainty")

class PredictionResponse(BaseModel):
    """Response model for battery predictions."""
    timestamps: List[datetime] = Field(..., description="Prediction timestamps")
    soc_prediction: List[float] = Field(..., description="SOC predictions")
    voltage_prediction: List[float] = Field(..., description="Voltage predictions")
    temperature_prediction: List[float] = Field(..., description="Temperature predictions")
    soh_prediction: float = Field(..., description="SOH prediction")
    rul_hours: float = Field(..., description="Remaining useful life in hours")
    uncertainty: Optional[Dict[str, List[float]]] = Field(None, description="Prediction uncertainties")
