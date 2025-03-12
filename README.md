# Sila WS40 Silicon Anode Li-ion Battery Digital Twin

A comprehensive digital twin implementation for the Sila Nanotechnologies WS40 silicon anode lithium-ion battery, integrating multi-physics modeling with real-time monitoring and predictive analytics.

![Outline_DigTwin_WS40](https://github.com/user-attachments/assets/ecb6ff37-7f37-44f3-908f-b7c06e527503)



## Overview

This project creates a digital replica of the Sila WS40 battery, incorporating:
- Equivalent Circuit Model (ECM) for electrical behavior
- Thermal modeling for temperature distribution
- Mechanical modeling for stress and deformation
- Machine learning models for State of Health (SoH) and Remaining Useful Life (RUL) prediction

## Key Features

- **Multi-Physics Simulation**
  - Advanced ECM implementation
  - Thermal distribution modeling
  - Mechanical stress and strain analysis
  - Silicon anode-specific deformation modeling

- **Real-time Monitoring**
  - FastAPI backend for data ingestion
  - Streamlit dashboard for visualization
  - Sensor data processing pipeline

- **Predictive Analytics**
  - LSTM-based degradation prediction
  - Gaussian Process Regression for uncertainty quantification
  - Real-time SoH estimation

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- Data Processing: NumPy, Pandas
- ML Framework: PyTorch, Scikit-learn
- Deployment: Docker, AWS ECS

## Getting Started

1. Clone the repository
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Access the services:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Dashboard: http://localhost:8501

## Project Structure

```
.
├── api/                 # FastAPI backend
├── data/               # Data storage
│   ├── raw/           # Raw sensor data
│   └── processed/     # Processed datasets
├── datasets/          # Public battery datasets
├── frontend/          # Streamlit dashboard
├── models/            # ML model implementations
├── notebooks/         # Jupyter notebooks
├── simulation/        # Physics-based models
└── tests/             # Test suite
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sila Nanotechnologies for the WS40 battery specifications
- NASA Battery Dataset
- MIT Battery Aging Dataset
