# Federated Learning with Differential Privacy and Advanced Aggregation Strategies

This project provides a robust and flexible framework for conducting federated learning (FL) experiments, focusing on the interplay between differential privacy (DP) and various aggregation strategies. It is built using the Flower framework and Opacus for PyTorch-based federated learning with differential privacy guarantees.

## Core Features

- **Differential Privacy**: Implements DP-SGD using Opacus library with multiple noise mechanisms
  - **Gaussian Noise Mechanism**: Sample-level differential privacy with configurable privacy budgets (standard implementation)
  - **Laplace Noise Mechanism**: Experimental differential privacy implementation for comparative analysis
  - **Flexible Privacy Levels**: Low (ε=15), Medium (ε=8), and High (ε=4) privacy configurations
  - **Privacy Budget Management**: Automatic calculation of noise scales based on target epsilon values
  - **Noise Mechanism Comparison**: Framework supports switching between Gaussian and Laplace mechanisms for research purposes
- **Advanced Aggregation Strategies**: 
  - **FedAvg**: Standard federated averaging algorithm
  - **FedAdam**: Adaptive server-side optimization with momentum
  - **Krum/Multi-Krum**: Byzantine-robust aggregation methods for adversarial environments
- **Modular and Configurable Architecture**:
  - **Code/Config Separation**: All experimental parameters centralized in `experiment_configs.py`
  - **Automated Experiment Management**: `run_experiment.py` orchestrates experiments and saves results
  - **Environment Variable Control**: Dynamic configuration through environment variables
- **Efficient Implementation**: 
  - Dataset caching for faster repeated experiments
  - GPU acceleration support
  - Memory-efficient client implementations

## Project Structure

```
.
├── client.py                 # Flower client implementation with local training and evaluation
├── server.py                 # Flower server with strategy implementations (FedAvg, FedAdam, Krum)
├── opacus_client_dp.py       # Differential privacy wrapper using Opacus
├── utils.py                  # Utility functions: models (CNN, MLP), data handling, training loops
├── experiment_configs.py     # Central configuration hub for all experiments
├── run_experiment.py         # Main orchestration script for experiments
├── pyproject.toml           # Project metadata, dependencies, and Flower App configuration
├── requirements.txt         # Alternative dependency specification
└── README.md               # This documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- CUDA-compatible GPU (optional, but recommended)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/qiuqiu035/dp-federated-learning.git
   cd dp-federated-learning
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   The project uses `pyproject.toml` for dependency management:
   ```bash
   pip install -e .
   ```
   
   Alternatively, install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- **flwr>=1.6.0**: Federated learning framework
- **torch>=1.12.0**: Deep learning framework
- **opacus>=1.4.0,<1.6.0**: Differential privacy for PyTorch
- **torchvision>=0.13.0**: Computer vision datasets and transforms
- **numpy>=1.21.0**: Numerical computing

## Usage

### Running Experiments

The `run_experiment.py` script is the main entry point for all experiments. It reads configurations from `experiment_configs.py` and manages the federated learning process.

#### Single Experiment
Run a specific experiment by name:
```bash
python run_experiment.py <experiment_name>
```

Example:
```bash
python run_experiment.py fedadam_medium_privacy
```

#### List Available Experiments
View all configured experiments:
```bash
python run_experiment.py
```

#### Predefined Experiment Sets
Run multiple related experiments:

**Baseline experiments (No DP):**
```bash
python run_experiment.py all_baseline
```

**All privacy experiments:**
```bash
python run_experiment.py all_privacy
```

**Compare aggregation strategies:**
```bash
python run_experiment.py compare_aggregation
```

### Available Experiments

#### FedAvg Experiments
- `fedavg_baseline`: Standard FedAvg without differential privacy
- `fedavg_low_privacy`: FedAvg with low privacy (ε=0.15)
- `fedavg_medium_privacy`: FedAvg with medium privacy (ε=0.08)
- `fedavg_high_privacy`: FedAvg with high privacy (ε=0.04)

#### FedAdam Experiments
- `fedadam_baseline`: FedAdam without differential privacy
- `fedadam_low_privacy`: FedAdam with low privacy (ε=0.15)
- `fedadam_medium_privacy`: FedAdam with medium privacy (ε=0.08)
- `fedadam_high_privacy`: FedAdam with high privacy (ε=0.04)

#### Krum Experiments
- `krum_baseline`: Krum aggregation without differential privacy
- `krum_medium_privacy`: Krum with medium privacy (ε=0.08)
- `multikrum_medium_privacy`: Multi-Krum with medium privacy (ε=0.08)

### Experiment Configuration

All experiment parameters are defined in `experiment_configs.py`. Each experiment configuration includes:

- **Aggregation method**: `fedavg`, `fedadam`, or `krum`
- **Client optimizer**: `sgd` or `adam`
- **Privacy settings**: `use_dp`, `target_epsilon`, `max_grad_norm`, `delta`
- **Noise mechanism**: `gaussian` (default) or `laplace` for experimental comparison
- **Training parameters**: Number of rounds, local epochs, batch size
- **Strategy-specific parameters**: Krum parameters, FedAdam learning rates

Example configurations:

**Gaussian Noise (Standard):**
```python
"fedavg_medium_privacy": {
    "description": "FedAvg + SGD with Gaussian DP (Medium Privacy, target_epsilon=0.08).",
    "aggregation_method": "fedavg",
    "client_optimizer": "sgd",
    "use_dp": True,
    "target_epsilon": 0.08,
    "max_grad_norm": 1.2,
    "delta": 1e-5,
    "noise_mechanism": "gaussian"
}
```

**Laplace Noise (Experimental):**
```python
"fedavg_medium_privacy_laplace": {
    "description": "FedAvg + SGD with Laplace DP (Medium Privacy, experimental).",
    "aggregation_method": "fedavg",
    "client_optimizer": "sgd",
    "use_dp": True,
    "noise_mechanism": "laplace",
    "epsilon_per_step": 0.01,
    "max_grad_norm": 1.2,
    "delta": 0.0
}
```

### Results and Logging

All experiment results are automatically saved in the `results/` directory, organized by experiment name:

```
results/
├── fedavg_baseline/
│   ├── config.json          # Experiment configuration
│   ├── experiment_log.txt   # Detailed execution logs
│   └── final_results.json   # Final accuracy and metrics
├── fedadam_medium_privacy/
│   └── ...
└── ...
```

## Customization

### Adding New Experiments

1. Open `experiment_configs.py`
2. Add a new entry to the `EXPERIMENT_CONFIGS` dictionary:
   ```python
   "my_custom_experiment": {
       "description": "Custom experiment description",
       "aggregation_method": "fedavg",
       "client_optimizer": "sgd",
       "use_dp": True,
       "target_epsilon": 0.1,
       # ... other parameters
   }
   ```

### Modifying Models and Datasets

The `utils.py` file contains:
- **Model definitions**: `SimpleCNN`, `MLP`
- **Data loading**: `load_datasets()` function
- **Training/testing loops**: `train()`, `test()` functions

To use a different dataset:
1. Modify the `DATASET_NAME` constant in `client.py`
2. Update the `load_datasets()` function in `utils.py`

### Adjusting Federated Learning Parameters

**Server-side parameters** (in `server.py`):
- `fraction_fit`: Fraction of clients participating in each round
- `min_fit_clients`: Minimum number of clients for training
- `min_available_clients`: Minimum clients needed to start

**Client-side parameters** (in `client.py`):
- `LOCAL_EPOCHS`: Number of local training epochs
- `BATCH_SIZE`: Training batch size
- Learning rates and optimization parameters

## Technical Details

### Differential Privacy Implementation

The project implements differential privacy through two distinct noise mechanisms:

#### Gaussian Noise (Standard Implementation)
- **Framework**: Built on Opacus PrivacyEngine
- **Privacy Level**: Sample-level differential privacy
- **Mechanism**: DP-SGD with per-sample gradient computation and clipping
- **Privacy Accounting**: RDP (Rényi Differential Privacy) accountant for tight privacy bounds
- **Noise Distribution**: Gaussian noise added to aggregated per-sample gradients
- **Configuration**: Supports both fixed noise multiplier and target epsilon approaches

#### Laplace Noise (Experimental Implementation)
- **Framework**: Custom implementation extending Opacus
- **Privacy Level**: Batch-level differential privacy
- **Mechanism**: Laplace noise added to batch-averaged gradients
- **Privacy Accounting**: Linear epsilon accumulation (pure ε-differential privacy)
- **Noise Distribution**: Laplace distribution with scale parameter b = sensitivity/epsilon
- **Configuration**: Requires explicit epsilon_per_step parameter

**Additional Features**:
- **Privacy Accountant**: Tracks privacy budget across rounds
- **Gradient Clipping**: L2 norm clipping with configurable `max_grad_norm`
- **Noise Addition**: Noise calibrated to achieve target epsilon values
- **Privacy Analysis**: Automatic computation of privacy guarantees

### Aggregation Strategies

**FedAvg**: Weighted averaging based on client dataset sizes
**FedAdam**: Server-side adaptive optimization with momentum and bias correction
**Krum**: Byzantine-robust aggregation selecting the most representative update
**Multi-Krum**: Extension of Krum averaging multiple selected updates

### Performance Considerations

- **GPU Utilization**: Automatic GPU detection and usage
- **Memory Management**: Efficient cleanup and garbage collection
- **Caching**: Dataset caching to avoid repeated loading
- **Parallel Processing**: Configurable client resources in `pyproject.toml`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or disable GPU in `client.py`
2. **Opacus Compatibility**: Ensure PyTorch version matches Opacus requirements
3. **Privacy Budget Exhaustion**: Reduce target epsilon or number of rounds

### Logging and Debugging

Enable detailed logging by setting environment variables:
```bash
export PYTHONPATH=.
export FL_LOG_LEVEL=DEBUG
python run_experiment.py <experiment_name>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Update documentation as needed
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{dp_federated_learning,
  title={Federated Learning with Differential Privacy and Advanced Aggregation Strategies},
  author={qiuqiu035},
  year={2025},
  url={https://github.com/qiuqiu035/dp-federated-learning}
}
```
