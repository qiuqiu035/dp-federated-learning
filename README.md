# Differentially Private Federated Learning

This repository contains the implementation developed for the thesis
*Differentially Private Federated Learning: Study and Implementation of Additive Noise Mechanisms*.
It provides a common experimental pipeline for studying how federated-learning architecture,
privacy granularity, and additive-noise distribution interact under heterogeneous client data.

The framework compares:

- server-based FedAvg and synchronous serverless neighbor aggregation;
- sample-level and client-level differential privacy;
- Gaussian and Laplace additive noise;
- full and partial client participation in the server-based setting;
- Watts-Strogatz serverless topologies with two or four neighbors.

## Methodology

Both architectures use the same MNIST model, client training logic, data partitioning, privacy
modules, and random-seed management. The server-based path aggregates client updates through a
Flower strategy. The serverless path performs local training followed by synchronous POST-MIX
neighbor averaging over a fixed Watts-Strogatz graph.

Sample-level DP clips per-sample gradients and injects noise during local optimization. Gaussian
sample-level DP uses Opacus. Client-level DP clips each complete client model update and adds noise
before communication. Gaussian privacy consumption is tracked with Renyi DP accounting; Laplace
privacy loss is composed linearly.

The thesis experiments use 100 clients, 500 communication rounds, one local epoch, batch size 64,
an MLP on MNIST, and SGD with learning rate 0.001, momentum 0.9, and weight decay 0.0001. The main
experiments use a Dirichlet non-IID partition with `alpha = 0.2`, target privacy budget
`epsilon = 30`, and Gaussian `delta = 1e-5`. Sample-level clipping uses `C = 1.3`; client-level
clipping uses `C = 0.03`. These values can be changed in the configuration modules.

## Privacy interpretation

The Laplace experiments deliberately retain L2 clipping and add coordinate-wise Laplace noise.
This keeps clipping, training, and aggregation aligned with the Gaussian and Opacus pipelines so
that the noise distribution is the primary experimental variable.

This path is an engineering comparison. It is not a canonical Laplace mechanism calibrated
directly from global L1 sensitivity. An L2 bound does not by itself justify using `C / epsilon` as
the coordinate-wise Laplace scale for a high-dimensional vector; a formal L1-sensitivity
calibration may require a dimension-dependent factor and can also depend on the adjacency
definition. Results from this path should be interpreted within that experimental scope.

## Installation

Python 3.11 is recommended. The complete regression matrix was verified on Linux with Python
3.11.13, PyTorch 2.5.1, torchvision 0.20.1, Opacus 1.5.4, Flower 1.22.0, and
flwr-datasets 0.5.0.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Running experiments

List the server-based configurations:

```bash
python run_experiment.py --list
```

Run one server-based experiment:

```bash
python run_experiment.py --config fedavg_baseline --seed 2
python run_experiment.py --config fedavg_privacy --seed 2
python run_experiment.py --config fedavg_localdpmod_gaussian --seed 2
```

Use `python run_experiment.py --all` to run every server-based configuration sequentially.

List and run serverless configurations:

```bash
python run_serverless_experiment.py --list
python run_serverless_experiment.py --config serverless_fedavg_baseline_ws --seed 2
python run_serverless_experiment.py --config serverless_fedavg_privacy_ws --seed 2
python run_serverless_experiment.py --config serverless_fedavg_localdpmod_gaussian_ws --seed 2
```

Results are written under `results/`; MNIST is downloaded under `data/`. Both directories are
ignored by Git.

### Available experiment families

| Architecture | Privacy level | Noise | Configurations |
|---|---|---|---|
| Server | None | None | `fedavg_baseline` |
| Server | Sample | Gaussian / Laplace | `fedavg_privacy`, `fedavg_privacy_0.1`, `fedavg_laplace_privacy` |
| Server | Client | Gaussian / Laplace | `fedavg_localdpmod_gaussian`, `fedavg_localdpmod_gaussian_0.1`, `fedavg_localdpmod_laplace` |
| Serverless | None | None | `serverless_fedavg_baseline_ws`, `serverless_fedavg_baseline_ws_4` |
| Serverless | Sample | Gaussian / Laplace | `serverless_fedavg_privacy_ws`, `serverless_fedavg_privacy_ws_4`, `serverless_fedavg_laplace_privacy_ws_4` |
| Serverless | Client | Gaussian / Laplace | `serverless_fedavg_localdpmod_gaussian_ws`, `serverless_fedavg_localdpmod_gaussian_ws_4`, `serverless_fedavg_localdpmod_laplace_ws_4` |

## CPU, GPU, and cluster execution

The code selects CUDA when it is visible to the process and otherwise uses CPU. A CUDA-enabled
PyTorch build does not allocate a GPU by itself: a Slurm job must request GPU resources, and the
Flower client resource settings in `pyproject.toml` must fit the allocation. CPU-only execution is
supported and was used for the compact regression suite.

For privacy comparisons, execution length and privacy-calibration length are separate. Short test
runs may execute one round while retaining `privacy_calibration_rounds = 500`, keeping per-round
calibration consistent with the full thesis configuration.

## Repository structure

- `client.py` - Flower client, local training, and sample-level DP integration.
- `server.py` - server-based strategies, aggregation, evaluation, and accounting.
- `serverless_runner.py` - synchronous decentralized training and POST-MIX aggregation.
- `experiment_configs.py` and `serverless_experiment_configs.py` - controlled experiment definitions.
- `opacus_client_dp.py` - sample-level Gaussian and experimental Laplace paths.
- `localdp_adapter.py` and `localdpmod/` - client-update clipping and noise injection.
- `privacy_accountant.py` - Gaussian RDP and Laplace composition utilities.
- `seed_manager.py` and `utils.py` - reproducibility, data loading, models, and topology helpers.

## Scope of the reported findings

In the thesis experiments, server-based aggregation was more accurate and stable than the tested
serverless topologies, and Gaussian noise was more reliable than Laplace noise across the evaluated
privacy granularities. These are empirical findings for the MNIST/MLP, synchronous-participation,
static-topology setup; they should not be treated as universal claims about all federated systems.

Natural extensions include larger datasets and models, dynamic or asynchronous networks, broader
epsilon sweeps, variable participation, and secure aggregation.
