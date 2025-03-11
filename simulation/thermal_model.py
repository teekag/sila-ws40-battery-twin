import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermalParameters:
    """Thermal parameters for Sila WS40 Silicon Anode Battery."""
    
    # Physical dimensions (m)
    length: float = 0.065    # Cell length
    width: float = 0.035     # Cell width
    height: float = 0.007    # Cell thickness
    
    # Material properties
    density: float = 2500.0          # kg/m³
    specific_heat: float = 1000.0    # J/(kg·K)
    thermal_conductivity: float = 3.0  # W/(m·K)
    
    # Thermal limits
    temp_max: float = 60.0  # °C
    temp_min: float = 0.0   # °C
    
    # Cooling parameters
    cooling_coefficient: float = 10.0  # W/(m²·K)
    
    # Silicon-specific parameters
    si_thermal_conductivity: float = 150.0  # W/(m·K)
    si_specific_heat: float = 700.0        # J/(kg·K)
    si_content: float = 0.45              # 45% silicon in anode

class ThermalModel:
    """Thermal model for Sila WS40 battery with silicon anode considerations."""
    
    def __init__(self, params: Optional[ThermalParameters] = None):
        """Initialize thermal model.
        
        Args:
            params: Thermal parameters
        """
        self.params = params or ThermalParameters()
        
        # Initialize temperature distribution (°C)
        # Using a 3D grid for better spatial resolution
        self.nx, self.ny, self.nz = 5, 3, 3
        self.T = np.ones((self.nx, self.ny, self.nz)) * 25.0
        
        # Calculate derived parameters
        self.volume = (self.params.length * 
                      self.params.width * 
                      self.params.height)
        self.surface_area = 2 * (
            self.params.length * self.params.width +
            self.params.length * self.params.height +
            self.params.width * self.params.height
        )
        
        # Effective thermal properties considering silicon content
        self._calculate_effective_properties()
        
    def _calculate_effective_properties(self):
        """Calculate effective thermal properties with silicon content."""
        # Simple weighted average for thermal conductivity
        self.k_eff = (
            self.params.si_content * self.params.si_thermal_conductivity +
            (1 - self.params.si_content) * self.params.thermal_conductivity
        )
        
        # Effective specific heat
        self.cp_eff = (
            self.params.si_content * self.params.si_specific_heat +
            (1 - self.params.si_content) * self.params.specific_heat
        )
        
    def _calculate_heat_generation(self,
                                 current: float,
                                 voltage: float,
                                 internal_resistance: float) -> float:
        """Calculate heat generation rate.
        
        Args:
            current: Applied current (A)
            voltage: Terminal voltage (V)
            internal_resistance: Internal resistance (Ω)
            
        Returns:
            Heat generation rate (W)
        """
        # Joule heating
        joule_heat = current * current * internal_resistance
        
        # Entropic heating (simplified)
        entropic_heat = abs(current * voltage * 0.05)
        
        # Additional heat from silicon phase changes
        si_heat = (abs(current) * 
                  self.params.si_content * 
                  0.1)  # Approximate heat from Si phase change
        
        return joule_heat + entropic_heat + si_heat
    
    def _apply_cooling(self, ambient_temp: float, cooling_active: bool):
        """Apply cooling effects to the battery surface."""
        cooling_coeff = (self.params.cooling_coefficient * 2.0 
                        if cooling_active else 
                        self.params.cooling_coefficient)
        
        # Apply cooling to surface cells
        surface_temp_diff = self.T - ambient_temp
        cooling_rate = cooling_coeff * surface_temp_diff
        
        # Apply stronger cooling to edges and corners
        self.T[0, :, :] -= cooling_rate[0, :, :] * 0.1
        self.T[-1, :, :] -= cooling_rate[-1, :, :] * 0.1
        self.T[:, 0, :] -= cooling_rate[:, 0, :] * 0.1
        self.T[:, -1, :] -= cooling_rate[:, -1, :] * 0.1
    
    def update(self,
              current: float,
              voltage: float,
              internal_resistance: float,
              ambient_temp: float,
              cooling_active: bool = False,
              dt: Optional[float] = None) -> Tuple[float, float, float]:
        """Update thermal state for one timestep.
        
        Args:
            current: Applied current (A)
            voltage: Terminal voltage (V)
            internal_resistance: Internal resistance (Ω)
            ambient_temp: Ambient temperature (°C)
            cooling_active: Whether active cooling is enabled
            dt: Timestep (s), default 1.0
            
        Returns:
            Tuple of (average temp, max temp, min temp)
        """
        if dt is None:
            dt = 1.0
        
        # Calculate heat generation
        q_gen = self._calculate_heat_generation(
            current, voltage, internal_resistance)
        
        # Distribute heat generation (higher at center)
        q_dist = np.ones_like(self.T) * q_gen / self.T.size
        q_dist[1:-1, 1:-1, 1:-1] *= 1.5  # More heat generated in core
        
        # 3D heat diffusion (simplified)
        dx = self.params.length / self.nx
        dy = self.params.width / self.ny
        dz = self.params.height / self.nz
        
        # Thermal diffusivity
        alpha = self.k_eff / (self.params.density * self.cp_eff)
        
        # Update temperature distribution
        T_new = self.T.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # 3D heat equation discretization
                    d2T_dx2 = (self.T[i+1,j,k] - 2*self.T[i,j,k] + self.T[i-1,j,k]) / dx**2
                    d2T_dy2 = (self.T[i,j+1,k] - 2*self.T[i,j,k] + self.T[i,j-1,k]) / dy**2
                    d2T_dz2 = (self.T[i,j,k+1] - 2*self.T[i,j,k] + self.T[i,j,k-1]) / dz**2
                    
                    T_new[i,j,k] = self.T[i,j,k] + dt * (
                        alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2) +
                        q_dist[i,j,k] / (self.params.density * self.cp_eff)
                    )
        
        self.T = T_new
        
        # Apply cooling effects
        self._apply_cooling(ambient_temp, cooling_active)
        
        # Temperature limits and warnings
        if np.max(self.T) > self.params.temp_max:
            logger.warning(
                f"Temperature ({np.max(self.T):.1f}°C) exceeds maximum "
                f"({self.params.temp_max}°C)")
        elif np.min(self.T) < self.params.temp_min:
            logger.warning(
                f"Temperature ({np.min(self.T):.1f}°C) below minimum "
                f"({self.params.temp_min}°C)")
        
        return (float(np.mean(self.T)), 
                float(np.max(self.T)), 
                float(np.min(self.T)))
