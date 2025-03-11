import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MechanicalParameters:
    """Mechanical parameters for the Sila WS40 Silicon Anode."""
    
    # Material properties
    youngs_modulus: float = 80.0e9    # Pa (Silicon-based anode)
    poissons_ratio: float = 0.22      # Dimensionless
    yield_strength: float = 120.0e6    # Pa
    density: float = 2500.0           # kg/m³
    
    # Geometry
    length: float = 0.065    # m
    width: float = 0.035     # m
    height: float = 0.007    # m
    
    # Silicon expansion coefficients
    volume_expansion_ratio: float = 0.04  # Per unit lithiation
    thermal_expansion_coeff: float = 2.6e-6  # Per °C
    
    # Operating limits
    max_strain: float = 0.15      # Maximum allowable strain
    max_stress: float = 150.0e6   # Pa (Maximum allowable stress)

class MechanicalModel:
    """Mechanical model for battery deformation and stress analysis."""
    
    def __init__(self, params: Optional[MechanicalParameters] = None):
        """Initialize mechanical model.
        
        Args:
            params: Mechanical parameters
        """
        self.params = params or MechanicalParameters()
        
        # Initialize state variables
        self.reset_state()
        
        # Calculate initial volume
        self.initial_volume = (self.params.length * 
                             self.params.width * 
                             self.params.height)
    
    def reset_state(self):
        """Reset mechanical state variables."""
        self.strain_tensor = np.zeros((3, 3))  # Engineering strain tensor
        self.stress_tensor = np.zeros((3, 3))  # Stress tensor (Pa)
        self.displacement = np.zeros(3)        # Displacement vector (m)
        self.current_volume = self.initial_volume
        
    def _calculate_lithiation_strain(self, soc: float) -> float:
        """Calculate strain due to lithiation.
        
        Args:
            soc: State of charge (0 to 1)
            
        Returns:
            Volumetric strain due to lithiation
        """
        # Linear approximation of volume expansion with lithiation
        return self.params.volume_expansion_ratio * soc
    
    def _calculate_thermal_strain(self, 
                                temperature: float,
                                reference_temp: float = 25.0) -> float:
        """Calculate thermal strain.
        
        Args:
            temperature: Current temperature (°C)
            reference_temp: Reference temperature (°C)
            
        Returns:
            Thermal strain
        """
        return self.params.thermal_expansion_coeff * (temperature - reference_temp)
    
    def _calculate_stress_tensor(self, strain_tensor: np.ndarray) -> np.ndarray:
        """Calculate stress tensor using Hooke's law.
        
        Args:
            strain_tensor: 3x3 strain tensor
            
        Returns:
            3x3 stress tensor
        """
        E = self.params.youngs_modulus
        v = self.params.poissons_ratio
        
        # Lame parameters
        lambda_param = E * v / ((1 + v) * (1 - 2*v))
        mu = E / (2 * (1 + v))
        
        # Calculate stress tensor
        trace_strain = np.trace(strain_tensor)
        stress = 2 * mu * strain_tensor
        stress += lambda_param * trace_strain * np.eye(3)
        
        return stress
    
    def update(self,
              soc: float,
              temperature: float,
              external_pressure: float = 101325.0,  # 1 atm
              reference_temp: float = 25.0) -> Dict:
        """Update mechanical state.
        
        Args:
            soc: State of charge (0 to 1)
            temperature: Current temperature (°C)
            external_pressure: External pressure (Pa)
            reference_temp: Reference temperature (°C)
            
        Returns:
            Dictionary containing mechanical state variables
        """
        # Calculate strains
        e_lith = self._calculate_lithiation_strain(soc)
        e_thermal = self._calculate_thermal_strain(temperature, reference_temp)
        
        # Total volumetric strain
        e_vol = e_lith + e_thermal
        
        # Assume isotropic expansion for simplification
        e_linear = e_vol / 3.0
        
        # Update strain tensor (diagonal components)
        self.strain_tensor = np.eye(3) * e_linear
        
        # Calculate stress tensor
        self.stress_tensor = self._calculate_stress_tensor(self.strain_tensor)
        
        # Add external pressure
        self.stress_tensor -= np.eye(3) * external_pressure
        
        # Calculate displacement (simplified)
        self.displacement = np.array([
            self.params.length * e_linear,
            self.params.width * e_linear,
            self.params.height * e_linear
        ])
        
        # Update current volume
        self.current_volume = self.initial_volume * (1 + e_vol)
        
        # Calculate von Mises stress
        s = self.stress_tensor
        von_mises = np.sqrt(0.5 * (
            (s[0,0] - s[1,1])**2 +
            (s[1,1] - s[2,2])**2 +
            (s[2,2] - s[0,0])**2 +
            6*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2)
        ))
        
        # Check for yield criterion
        if von_mises > self.params.max_stress:
            logger.warning(f"Von Mises stress ({von_mises/1e6:.1f} MPa) "
                         f"exceeds maximum allowable stress "
                         f"({self.params.max_stress/1e6:.1f} MPa)")
        
        return {
            'strain_tensor': self.strain_tensor.copy(),
            'stress_tensor': self.stress_tensor.copy(),
            'displacement': self.displacement.copy(),
            'volume_change': (self.current_volume / self.initial_volume - 1) * 100,
            'von_mises_stress': von_mises,
            'max_principal_stress': np.max(np.linalg.eigvals(self.stress_tensor)),
            'yielding': von_mises > self.params.max_stress
        }
    
    def get_safety_metrics(self) -> Dict:
        """Calculate safety-related metrics.
        
        Returns:
            Dictionary of safety metrics
        """
        # Calculate von Mises stress
        s = self.stress_tensor
        von_mises = np.sqrt(0.5 * (
            (s[0,0] - s[1,1])**2 +
            (s[1,1] - s[2,2])**2 +
            (s[2,2] - s[0,0])**2 +
            6*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2)
        ))
        
        # Maximum shear stress (Tresca criterion)
        principal_stresses = np.linalg.eigvals(self.stress_tensor)
        max_shear = (np.max(principal_stresses) - np.min(principal_stresses)) / 2
        
        # Maximum strain
        max_strain = np.max(np.abs(self.strain_tensor))
        
        return {
            'stress_ratio': von_mises / self.params.max_stress,
            'strain_ratio': max_strain / self.params.max_strain,
            'max_shear_stress': max_shear,
            'safety_factor': self.params.max_stress / von_mises if von_mises > 0 else float('inf')
        }
