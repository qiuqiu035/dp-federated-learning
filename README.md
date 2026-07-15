# Differentially Private Federated Learning

## Overview

This repository contains code for the unpublished master's thesis *Differentially Private Federated Learning: Study and Implementation of Additive Noise Mechanisms*. It compares server-based and serverless learning under sample-level and client-level differential privacy.

The work was completed for the Master's degree in Data Science at the University of Padova. The manuscript is not public.

## Key comparisons

- Server-based FedAvg and serverless neighbor aggregation
- Sample-level and client-level DP
- Gaussian and Laplace noise
- Full and partial participation
- Two- and four-neighbor serverless topologies

## Quick start

```bash
python -m pip install -r requirements.txt
python run_experiment.py --config fedavg_baseline --seed 2
```

## Installation

Python 3.11 is recommended. Create a virtual environment with `python -m venv .venv`, then activate it with `source .venv/bin/activate` on Linux or `.venv\Scripts\activate` on Windows. Tests used PyTorch 2.5.1, Opacus 1.5.4, Flower 1.22.0, and flwr-datasets 0.5.0.

## Running experiments

```bash
python run_experiment.py --list
python run_experiment.py --config fedavg_privacy --seed 2
python run_serverless_experiment.py --list
python run_serverless_experiment.py --config serverless_fedavg_privacy_ws --seed 2
```

Outputs are saved under `results/`; MNIST is downloaded under `data/`. Both are ignored by Git.

## Available configurations

| Architecture | DP level | Noise | Setting | Configuration |
|---|---|---|---|---|
| Server | None | None | Full participation | `fedavg_baseline` |
| Server | Sample | Gaussian | Full participation | `fedavg_privacy` |
| Server | Sample | Gaussian | 10% participation | `fedavg_privacy_0.1` |
| Server | Sample | Laplace | Full participation | `fedavg_laplace_privacy` |
| Server | Client | Gaussian | Full participation | `fedavg_localdpmod_gaussian` |
| Server | Client | Gaussian | 10% participation | `fedavg_localdpmod_gaussian_0.1` |
| Server | Client | Laplace | Full participation | `fedavg_localdpmod_laplace` |
| Serverless | None | None | 2 neighbors | `serverless_fedavg_baseline_ws` |
| Serverless | None | None | 4 neighbors | `serverless_fedavg_baseline_ws_4` |
| Serverless | Sample | Gaussian | 2 neighbors | `serverless_fedavg_privacy_ws` |
| Serverless | Sample | Gaussian | 4 neighbors | `serverless_fedavg_privacy_ws_4` |
| Serverless | Sample | Laplace | 4 neighbors | `serverless_fedavg_laplace_privacy_ws_4` |
| Serverless | Client | Gaussian | 2 neighbors | `serverless_fedavg_localdpmod_gaussian_ws` |
| Serverless | Client | Gaussian | 4 neighbors | `serverless_fedavg_localdpmod_gaussian_ws_4` |
| Serverless | Client | Laplace | 4 neighbors | `serverless_fedavg_localdpmod_laplace_ws_4` |

## Default experimental settings

| Parameter | Default |
|---|---|
| Dataset / model | MNIST / MLP |
| Clients / rounds | 100 / 500 |
| Local epochs / batch size | 1 / 64 |
| Optimizer | SGD, learning rate 0.001, momentum 0.9, weight decay 0.0001 |
| Dirichlet alpha | 0.2 |
| Target epsilon / Gaussian delta | 30 / 1e-5 |
| Sample-level / client-level clip norm | 1.3 / 0.03 |
| Serverless topology | Watts-Strogatz, p = 0.2 |

## Methodology summary

The server path uses Flower FedAvg. The serverless path performs synchronous local training followed by POST-MIX neighbor averaging on a fixed Watts-Strogatz graph.

Sample-level DP clips per-sample gradients; its Gaussian path uses Opacus. Client-level DP clips complete updates before communication. Gaussian privacy uses Renyi DP accounting; Laplace loss is composed linearly.

## Privacy interpretation

The Laplace experiments retain L2 clipping and apply coordinate-wise Laplace noise to keep the pipeline comparable with the Gaussian setting. This is an engineering comparison, not a canonical Laplace mechanism calibrated from global L1 sensitivity. In high dimensions, formal calibration may require a dimension-dependent sensitivity bound and depends on the adjacency definition.

## Hardware and reproducibility

The code uses CUDA when available and otherwise uses CPU. CPU-only execution is supported. On Slurm, GPU requests and `pyproject.toml` must match the allocation.

Executed and privacy-calibration rounds are separate. Validation may run one round while retaining `privacy_calibration_rounds = 500`. Fixed seeds control all randomized components.

## Repository structure

- `client.py`, `server.py`, `serverless_runner.py`: training and aggregation
- `experiment_configs.py`, `serverless_experiment_configs.py`: configurations
- `opacus_client_dp.py`, `localdp_adapter.py`, `localdpmod/`: privacy mechanisms
- `privacy_accountant.py`, `seed_manager.py`, `utils.py`: accounting and utilities

## Scope and limitations

Under the evaluated MNIST/MLP setup, server-based experiments were generally more accurate and stable than serverless experiments. Gaussian-noise runs were also more stable than Laplace-noise runs.

These observations are specific to the evaluated data, model, privacy settings, participation, and static synchronous topologies. Broader deployments were not evaluated.

## Thesis information

- **Title:** *Differentially Private Federated Learning: Study and Implementation of Additive Noise Mechanisms*
- **Author:** Guo Hongyu
- **Program:** Master's degree in Data Science
- **Institution:** University of Padova
- **Supervisor:** Giovanni Perin
- **Academic year:** 2024-2025
- **Status:** Unpublished; manuscript not publicly available

## License

No open-source license has been assigned. All rights are reserved by the author.
