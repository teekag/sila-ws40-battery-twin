import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryParameters:
    """Parameters for Sila WS40 Silicon Anode Battery."""
    
    # Nominal specifications
    nominal_capacity: float = 5.0  # Ah
    nominal_voltage: float = 3.7   # V
    
    # Voltage limits
    voltage_max: float = 4.2  # V
    voltage_min: float = 2.8  # V
    
    # Internal resistance parameters
    r_internal: float = 0.01  # Ohm
    r_sei: float = 0.002     # SEI layer resistance
    
    # Temperature coefficients
    temp_coeff: float = 0.004  # 1/°C
    
    # Silicon anode specific parameters
    silicon_content: float = 0.45  # 45% silicon in anode
    expansion_coeff: float = 0.04  # Volume expansion per unit lithiation

class EquivalentCircuitModel:
    """ECM for Sila WS40 battery with silicon anode considerations."""
    
    def __init__(self, params: Optional[BatteryParameters] = None):
        self.params = params or BatteryParameters()
        
        # Initialize state
        self.soc = 1.0  # State of Charge
        self.temperature = 25.0  # °C
        self.cycle_count = 0
        
        # RC pairs for dynamic behavior
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0
        self.r1 = 0.015  # Ohm
        self.c1 = 2000   # F
        self.r2 = 0.025  # Ohm
        self.c2 = 5000   # F
        
    def _update_temperature_effects(self):
        """Update resistance based on temperature."""
        temp_factor = 1.0 + self.params.temp_coeff * (self.temperature - 25.0)
        return self.params.r_internal / temp_factor
    
    def _calculate_ocv(self) -> float:
        """Calculate open circuit voltage considering silicon anode."""
        # Simplified OCV-SOC relationship for Si-graphite anode
        base_ocv = (
            2.8 + 
            1.4 * self.soc +  # Linear component
            0.2 * np.sin(np.pi * self.soc) +  # First plateau
            0.2 * np.sin(2 * np.pi * self.soc)  # Second plateau
        )
        
        # Add silicon anode voltage contribution
        si_voltage = 0.1 * np.log(self.soc + 0.1) * self.params.silicon_content
        
        return min(base_ocv + si_voltage, self.params.voltage_max)
    
    def update(self, 
              current: float, 
              dt: float, 
              ambient_temp: float = 25.0) -> Tuple[float, float, float]:
        """Update battery state for one timestep.
        
        Args:
            current: Applied current (A), positive for discharge
            dt: Timestep (s)
            ambient_temp: Ambient temperature (°C)
            
        Returns:
            Tuple of (terminal voltage, state of charge, temperature)
        """
        # Update SOC
        self.soc -= (current * dt) / (3600 * self.params.nominal_capacity)
        self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Update RC pair voltages
        self.v_rc1 *= np.exp(-dt / (self.r1 * self.c1))
        self.v_rc1 += self.r1 * current * (1 - np.exp(-dt / (self.r1 * self.c1)))
        
        self.v_rc2 *= np.exp(-dt / (self.r2 * self.c2))
        self.v_rc2 += self.r2 * current * (1 - np.exp(-dt / (self.r2 * self.c2)))
        
        # Calculate heat generation
        r_temp = self._update_temperature_effects()
        joule_heat = current * current * r_temp
        
        # Simple thermal model
        temp_diff = ambient_temp - self.temperature
        self.temperature += (joule_heat * 0.1 + temp_diff * 0.05) * dt
        
        # Calculate terminal voltage
        v_ocv = self._calculate_ocv()
        v_terminal = (v_ocv - 
                     current * r_temp - 
                     self.v_rc1 - 
                     self.v_rc2 - 
                     current * self.params.r_sei)
        
        # Log warnings for voltage limits
        if v_terminal > self.params.voltage_max:
            logger.warning(f"Terminal voltage ({v_terminal:.2f}V) exceeds maximum")
        elif v_terminal < self.params.voltage_min:
            logger.warning(f"Terminal voltage ({v_terminal:.2f}V) below minimum")
        
        return v_terminal, self.soc, self.temperature
