# server.py

from logging import config
import os
import logging
import random
import copy
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import flwr as fl
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.server import ServerAppComponents
from flwr.common import Parameters, Metrics, Context, ndarrays_to_parameters, parameters_to_ndarrays
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import MLP, get_model_parameters
from experiment_configs import get_experiment_config
from privacy_accountant import ClientGaussianRDPAccountant
from privacy_accountant import PrivacyAccountant
from seed_manager import get_seed_manager, reset_seed_manager, set_deterministic_mode

# Keep Flower logs together with the experiment output.
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)



logger = logging.getLogger(__name__)

# Collect per-round metrics for the final summary.
global_metrics_history = {
    "fit_metrics": [],
    "evaluate_metrics": []
}

def fix_server_seed(seed: int = 2025):
    """
    Fix random seeds for server-side deterministic behavior.
    This ensures client sampling is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Server random seeds fixed to {seed} for reproducible client sampling")

def get_or_create_initial_weights(model_class, num_classes: int = 10, seed: int = 2025,
                                 weights_file: str = "init_weights_mnist_mlp.npz") -> List[np.ndarray]:
    """
    Get fixed initial weights for reproducibility.
    If weights file exists, load from disk. Otherwise, create new ones with fixed seed.
    """
    weights_path = Path(weights_file)

    if weights_path.exists():
        logger.info(f"Loading existing initial weights from {weights_file}")
        data = np.load(weights_file)
        initial_parameters = [data[k] for k in sorted(data.files)]
        return initial_parameters
    else:
        logger.info(f"Creating new initial weights with seed {seed}, saving to {weights_file}")
        # Create model with fixed seed
        torch.manual_seed(seed)
        model = model_class(num_classes=num_classes)
        initial_parameters = get_model_parameters(model)

        # Save to disk for future runs
        np.savez(weights_file, *initial_parameters)
        logger.info(f"Initial weights saved to {weights_file}")
        return initial_parameters

def weighted_average(results: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
    """Compute weighted average of model weights (List[np.ndarray]) based on number of examples."""
    if not results:
        raise ValueError("No client results provided for aggregation")

    num_examples_list = [num_examples for num_examples, _ in results]
    total_examples = float(sum(num_examples_list))
    weights_list = [weights for _, weights in results]  # List[List[np.ndarray]]

    # Initialize output as empty list
    averaged_weights: List[np.ndarray] = []

    for layer_params in zip(*weights_list):  # tuple of np.ndarrays (from different clients, same layer)
        stacked = np.stack(layer_params, axis=0)  # shape: [num_clients, ...]
        coeffs = np.array([n / total_examples for n in num_examples_list], dtype=np.float64)
        coeffs_broadcast = coeffs.reshape((coeffs.shape[0],) + (1,) * (stacked.ndim - 1))
        averaged_layer = (stacked * coeffs_broadcast).sum(axis=0)
        averaged_weights.append(averaged_layer)

    return averaged_weights

def krum_aggregate(results: List[Tuple[int, List[np.ndarray]]], f: int = 1, multi_krum: bool = False) -> List[np.ndarray]:
    """
    Krum aggregation method for Byzantine-robust federated learning.

    Args:
        results: List of (num_examples, weights) tuples from clients
        f: Number of Byzantine clients to tolerate
        multi_krum: If True, use Multi-Krum (select n-f clients and average)

    Returns:
        Aggregated weights using Krum method
    """
    if len(results) <= 2 * f:
        logger.warning(f"Krum: Not enough clients ({len(results)}) for f={f}, using weighted average")
        return weighted_average(results)

    # Extract weights
    weights_list = [weights for _, weights in results]
    n = len(weights_list)

    # Flatten all weights for distance calculation
    flattened_weights = []
    for weights in weights_list:
        flattened = np.concatenate([w.flatten() for w in weights])
        flattened_weights.append(flattened)

    # Calculate distances between all pairs
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
            distances[i, j] = distances[j, i] = dist

    # For each client, sum distances to its k closest neighbors
    k = n - f - 2  # Number of closest neighbors to consider
    if k <= 0:
        k = 1
    scores = []

    for i in range(n):
        # Get distances from client i to all others
        client_distances = distances[i]
        # Sort and take k closest (excluding self)
        closest_distances = np.sort(client_distances)[1:k+2]  # Exclude self (distance 0)
        scores.append(np.sum(closest_distances))

    if multi_krum:
        # Multi-Krum: select top n-f clients (all honest clients) and average their weights
        m = n - f  # Multi-Krum selects all honest clients
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:m]
        logger.info(f"Multi-Krum: Selected {m} clients {selected_indices.tolist()} out of {n} clients (f={f}, selecting n-f={m})")

        # Create results with selected clients for weighted average
        selected_results = [(results[i][0], weights_list[i]) for i in selected_indices]
        return weighted_average(selected_results)
    else:
        # Single Krum: select client with minimum score
        selected_idx = np.argmin(scores)
        logger.info(f"Single Krum: Selected client {selected_idx} out of {n} clients (f={f})")
        return weights_list[selected_idx]

class DPFedAdam(FedAvg):
    """FedAdam with differential privacy support."""

    def __init__(self, privacy_accountant, beta_1: float = 0.9, beta_2: float = 0.999,
                 eta: float = 1e-3, tau: float = 1e-9, **kwargs):
        super().__init__(**kwargs)
        self.privacy_accountant = privacy_accountant
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eta = eta
        self.tau = tau
        self.m_t = None
        self.v_t = None
        self.server_round = 0
        self.current_weights = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:

        if not results:
            return None, {}

        self.server_round = server_round

        # Validate client-level DP updates before aggregation.
        if results and hasattr(self, 'current_weights') and self.current_weights is not None:
            # This diagnostic applies only to client-level DP.
            first_metrics = results[0][1].metrics
            dp_mode = first_metrics.get("dp_mode", "unknown")

            if dp_mode == "client":
                # Calculate L2 statistics for returned client updates.
                update_l2_norms = []
                adapter_noise_l2_list = []

                for client_proxy, fit_res in results:
                    client_weights = parameters_to_ndarrays(fit_res.parameters)
                    update_norm = 0.0
                    for old_w, new_w in zip(self.current_weights, client_weights):
                        delta = new_w - old_w
                        update_norm += np.sum(delta ** 2)
                    update_norm = np.sqrt(update_norm)
                    update_l2_norms.append(update_norm)

                    # Read the noise norm reported by the adapter.
                    if "final_noise_l2" in fit_res.metrics:
                        adapter_noise_l2_list.append(fit_res.metrics["final_noise_l2"])

                update_l2_norms = np.array(update_l2_norms)
                median_l2 = np.median(update_l2_norms)
                mean_l2 = np.mean(update_l2_norms)
                std_l2 = np.std(update_l2_norms)

                C = first_metrics.get("max_grad_norm", 1.0)
                K = len(results)

                logger.info(f"[SERVER DP-CHECK]  Round {server_round} Pre-Aggregation (K={K} clients):")
                logger.info(f"  === CLIENT UPDATE NORMS (Server View) ===")
                logger.info(f"  L2(delta) median: {median_l2:.4e}")
                logger.info(f"  L2(delta) mean: {mean_l2:.4e} +/- {std_l2:.4e}")
                logger.info(f"  Max_grad_norm (C): {C:.4f}")

                # Compare server-observed updates with adapter diagnostics.
                if adapter_noise_l2_list:
                    adapter_noise_median = np.median(adapter_noise_l2_list)
                    adapter_noise_mean = np.mean(adapter_noise_l2_list)
                    logger.info(f"  === ADAPTER REPORTED NOISE ===")
                    logger.info(f"  L2(noise) from adapter median: {adapter_noise_median:.4e}")
                    logger.info(f"  L2(noise) from adapter mean: {adapter_noise_mean:.4e}")

                    # The server and adapter should observe the same returned update.
                    expected_total_from_adapter = first_metrics.get("final_total_l2", 0)
                    if expected_total_from_adapter > 0:
                        ratio = median_l2 / expected_total_from_adapter
                        logger.info(f"  === CONSISTENCY CHECK ===")
                        logger.info(f"  Ratio (server_median / adapter_total): {ratio:.4f}")
                        if abs(ratio - 1.0) > 0.1:
                            logger.error(f"   CRITICAL: Server sees different values than adapter!")
                            logger.error(f"  Server median: {median_l2:.4e}")
                            logger.error(f"  Adapter total: {expected_total_from_adapter:.4e}")
                            raise RuntimeError("DP result lost on the way! Adapter's noisy parameters not reaching server!")

                # Fail fast when the returned perturbation is unexpectedly small.
                failed_clients = []
                for idx, (client_proxy, fit_res) in enumerate(results):
                    delta_l2 = update_l2_norms[idx]
                    if delta_l2 < 2.0 * C:
                        failed_clients.append((idx, delta_l2))

                if failed_clients:
                    logger.error(f" CRITICAL: {len(failed_clients)}/{K} clients have delta_l2 < 2xC={2*C:.4e}")
                    for idx, delta_l2 in failed_clients:
                        logger.error(f"  Client {idx}: delta_l2={delta_l2:.4e} < 2xC={2*C:.4e}")
                    raise RuntimeError(
                        f"DP verification failed! {len(failed_clients)}/{K} clients returned updates with "
                        f"delta_l2 < 2xC={2*C:.4e}. This indicates DP noise was not properly applied. "
                        f"Check LocalDpModAdapter wrapper!"
                    )

                # Flag updates whose norm is inconsistent with the configured noise scale.
                if median_l2 < C * 2:
                    logger.warning(f" SUSPICIOUS: median_l2={median_l2:.4e} < 2xC={2*C:.4e}")
                    logger.warning(f"  Clients may have returned non-noisy or under-noised parameters!")

                    # Check the adapter's explicit DP status flag.
                    dp_not_applied_count = sum(1 for _, fit_res in results if not fit_res.metrics.get("dp_applied", True))
                    if dp_not_applied_count > 0:
                        logger.error(f"   {dp_not_applied_count}/{K} clients reported dp_applied=False!")
                        raise RuntimeError(f"DP not properly applied! {dp_not_applied_count}/{K} clients failed DP check")
                else:
                    logger.info(f" VERIFIED: median_l2={median_l2:.4e} > 2xC={2*C:.4e}")
                    logger.info(f" All clients appear to have properly noised parameters")

        # Calculate client update norms (for adaptive clipping)
        client_update_norms = []
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]

        if self.current_weights is not None:
            for _, client_weights in weights_results:
                update_norm = 0.0
                for old_w, new_w in zip(self.current_weights, client_weights):
                    update_norm += np.sum((new_w - old_w) ** 2)
                update_norm = np.sqrt(update_norm)
                client_update_norms.append(update_norm)

        if self.current_weights is None and hasattr(self, 'initial_parameters'):
            self.current_weights = parameters_to_ndarrays(self.initial_parameters)

        avg_weights = weighted_average(weights_results)

        if self.current_weights is not None:
            pseudo_grad = [avg_w - curr_w for avg_w, curr_w in zip(avg_weights, self.current_weights)]

            if self.m_t is None:
                self.m_t = [np.zeros_like(g) for g in pseudo_grad]
                self.v_t = [np.zeros_like(g) for g in pseudo_grad]

            new_m_t = []
            new_v_t = []
            new_weights = []

            for i, g in enumerate(pseudo_grad):
                m = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * g
                v = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * (g * g)

                m_hat = m / (1 - self.beta_1 ** server_round)
                v_hat = v / (1 - self.beta_2 ** server_round)

                update = self.eta * m_hat / (np.sqrt(v_hat) + self.tau)
                new_weight = self.current_weights[i] + update

                new_m_t.append(m)
                new_v_t.append(v)
                new_weights.append(new_weight)

            self.m_t = new_m_t
            self.v_t = new_v_t
            self.current_weights = new_weights
            aggregated_parameters = ndarrays_to_parameters(new_weights)
        else:
            self.current_weights = avg_weights
            aggregated_parameters = ndarrays_to_parameters(avg_weights)

        # Report post-aggregation update norms.
        first_metrics = results[0][1].metrics if results else {}
        dp_mode = first_metrics.get("dp_mode", "unknown")

        if dp_mode == "client" and len(weights_results) > 0:
            # Compare aggregated parameters with the previous global model.
            old_weights = self.current_weights if hasattr(self, '_old_current_weights') else None
            if old_weights is None and hasattr(self, 'initial_parameters'):
                old_weights = parameters_to_ndarrays(self.initial_parameters)

            if old_weights is not None:
                aggregated_update_norm = 0.0
                for old_w, new_w in zip(old_weights, self.current_weights):
                    delta = new_w - old_w
                    aggregated_update_norm += np.sum(delta ** 2)
                aggregated_update_norm = np.sqrt(aggregated_update_norm)

                K = len(results)
                C = first_metrics.get("max_grad_norm", 1.0)

                # Independent client noise is reduced by approximately sqrt(K).
                logger.info(f"[DP-CHECK] Server round {server_round} post-aggregation statistics:")
                logger.info(f"  L2(aggregated_update) = {aggregated_update_norm:.4e}")
                logger.info(f"  Expected scale: single-client norm / sqrt({K}) = x {1/np.sqrt(K):.3f}")

        aggregated_metrics = {}

        # === Privacy Accounting ===
        privacy_update = _update_privacy_accounting(
            self.privacy_accountant, server_round, results
        )

        # Add the privacy update to our metrics
        aggregated_metrics.update(privacy_update)

        # Copy relevant metrics from client for logging
        first_metrics = results[0][1].metrics
        for k in ["noise_multiplier", "actual_noise_multiplier", "round_target_epsilon", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[k] = first_metrics[k]

        logger.info(f"FedAdam Round {server_round}: Aggregated with beta1={self.beta_1}, beta2={self.beta_2}, eta={self.eta}")

        return aggregated_parameters, aggregated_metrics

class DPKrum(FedAvg):
    """Krum aggregation with differential privacy support."""

    def __init__(self, privacy_accountant, f: int = 1, multi_krum: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.privacy_accountant = privacy_accountant
        self.f = f  # Number of Byzantine clients to tolerate
        self.multi_krum = multi_krum
        self.previous_weights = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:

        if not results:
            return None, {}

        client_update_norms = []
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]

        if hasattr(self, 'previous_weights') and self.previous_weights is not None:
            for _, client_weights in weights_results:
                update_norm = 0.0
                for old_w, new_w in zip(self.previous_weights, client_weights):
                    update_norm += np.sum((new_w - old_w) ** 2)
                update_norm = np.sqrt(update_norm)
                client_update_norms.append(update_norm)

        aggregated_weights = krum_aggregate(weights_results, self.f, self.multi_krum)

        self.previous_weights = copy.deepcopy(aggregated_weights)

        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)

        aggregated_metrics = {}

        # === Privacy Accounting ===
        privacy_update = _update_privacy_accounting(
            self.privacy_accountant, server_round, results
        )

        # Add the privacy update to our metrics
        aggregated_metrics.update(privacy_update)

        # Copy relevant metrics from client for logging
        first_metrics = results[0][1].metrics
        for k in ["noise_multiplier", "actual_noise_multiplier", "round_target_epsilon", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[k] = first_metrics[k]

        krum_type = "Multi" if self.multi_krum else "Single"
        logger.info(f"{krum_type} Krum Round {server_round}: Aggregated {len(results)} clients with f={self.f}")

        return aggregated_parameters, aggregated_metrics

def _update_privacy_accounting(
    accountant: PrivacyAccountant,
    server_round: int,
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
) -> Dict:
    """
    Helper function to update privacy budget using the advanced PrivacyAccountant.
    It inspects client metrics to decide which accounting method to use.
    """
    if not results:
        return {}

    # Collect metrics from all clients
    all_client_metrics = [fit_res.metrics for _, fit_res in results]

    # Calculate client sample rate: participating clients / total registered clients
    num_participating_clients = len(results)
    # Get total clients from context or strategy (default fallback)
    total_clients = accountant.__dict__.get('total_clients', 100)  # fallback to 100
    client_sample_rate = num_participating_clients / total_clients

    # Use metrics from the first client to determine the mode for this round
    # (assuming all clients in a round use the same DP settings)
    first_metrics = all_client_metrics[0]
    mechanism = first_metrics.get("noise_mechanism", "none")
    dp_mode = first_metrics.get("dp_mode", "none")  # Client should report this

    # Baseline clients must not create privacy events. Keep this guard on the
    # server as a second line of defence against malformed client metrics.
    if dp_mode in {"baseline", "none", "unknown"} or mechanism in {None, "none", "unknown"}:
        return {}

    # Add client_sample_rate to metrics for logging
    round_metrics = {
        "client_sample_rate": client_sample_rate,
        "num_participating_clients": num_participating_clients,
        "total_clients": total_clients
    }

    if mechanism == "laplace":
        eps_per_step = first_metrics.get("epsilon_per_step")
        steps = int(first_metrics.get("train_steps_this_round", 1))
        if eps_per_step is not None:
            for _ in range(steps):
                accountant.add_laplace_event(server_round, epsilon=float(eps_per_step))
        else:
            round_epsilon = first_metrics.get("round_target_epsilon")
            if round_epsilon:
                accountant.add_laplace_event(server_round, epsilon=float(round_epsilon))

    elif mechanism == "gaussian":
        if "sample_rate" in first_metrics:  # Sample-level DP (Opacus)
            # Opacus already computed privacy loss for the complete local
            # training. Prefer it over a second approximation from q/sigma.
            reported_epsilon = first_metrics.get("privacy_epsilon")
            if reported_epsilon is not None:
                accountant.add_client_level_gaussian_event(
                    round_num=server_round,
                    epsilon=float(reported_epsilon),
                    delta=float(first_metrics.get("privacy_delta", accountant.target_delta)),
                )
            else:
                noise_multiplier = first_metrics.get("actual_noise_multiplier") or first_metrics.get("noise_multiplier")
                sample_rate = first_metrics.get("sample_rate")
                if noise_multiplier is not None and sample_rate is not None:
                    steps = first_metrics.get("train_steps_this_round", 1)
                    accountant.add_gaussian_event(
                        noise_multiplier=float(noise_multiplier),
                        sampling_probability=float(sample_rate),
                        steps=int(steps),
                    )
        else:  # Client-level DP
            # Use new RDP accounting with client_sample_rate
            noise_multiplier = first_metrics.get("noise_multiplier")

            if noise_multiplier is not None and dp_mode == "client":
                # Use ClientGaussianRDPAccountant for precise RDP calculation
                # Get the strategy instance to access client_rdp
                # We'll store this in the accountant temporarily
                if not hasattr(accountant, '_client_rdp_data'):
                    accountant._client_rdp_data = {
                        'q': client_sample_rate,
                        'sigma': float(noise_multiplier),
                        'delta': first_metrics.get("delta", accountant.target_delta)
                    }
                    logger.info(f"[Server] Client-level Gaussian DP: q={client_sample_rate:.4f}, sigma={noise_multiplier}, delta={accountant._client_rdp_data['delta']}")

                # Add to round_metrics for strategy to use
                round_metrics.update({
                    "dp_mode": dp_mode,
                    "noise_multiplier": float(noise_multiplier),
                    "delta": first_metrics.get("delta", accountant.target_delta)
                })
            else:
                # Fallback: use traditional round_epsilon/delta for one-time accounting
                round_epsilon = first_metrics.get("round_target_epsilon")
                round_delta = first_metrics.get("delta", accountant.target_delta)
                if round_epsilon and round_delta:
                    accountant.add_client_level_gaussian_event(
                        server_round,
                        epsilon=float(round_epsilon),
                        delta=float(round_delta)
                    )

    # Get the latest privacy budget
    total_eps, total_delta = accountant.get_privacy_spent()
    breached = False
    if hasattr(accountant, 'epsilon_limit') and accountant.epsilon_limit is not None and total_eps >= accountant.epsilon_limit:
        breached = True

    # Return a summary dictionary for logging
    result = {
        "cumulative_epsilon": total_eps,
        "cumulative_delta": total_delta,
        "privacy_breached": breached,
    }
    result.update(round_metrics)  # Add client sample rate info
    return result

def fit_config(server_round: int) -> Dict:
    """
    The single source of truth for client configurations.
    It reads the experiment name from the environment and uses the
    central config file directly.
    """
    mode = os.environ.get("EXPERIMENT_MODE")
    if not mode:
        raise ValueError("EXPERIMENT_MODE environment variable not set.")

    # Get the base config for this experiment directly from our clean config file
    try:
        config = get_experiment_config(mode).copy()  # Copy to avoid modifying the original
    except ValueError as e:
        logger.error(f"Could not find configuration for EXPERIMENT_MODE='{mode}'")
        raise e

    if config.get("dp_mode") in ["opacus", "sample", "client"]:
        total_eps = config.get("total_epsilon", None)
        num_rounds = config.get("privacy_calibration_rounds", 500)
        if total_eps and not config.get("round_target_epsilon"):
            config["round_target_epsilon"] = total_eps / num_rounds
            if server_round == 1:
                logger.info(f"Auto-calculated round_target_epsilon: {config['round_target_epsilon']:.6f} (total_epsilon={total_eps} / num_rounds={num_rounds})")

    # Add dynamic, per-round information
    config["server_round"] = server_round

    # Add default values if not present (optional but good practice)
    config.setdefault("learning_rate", 0.001)
    config.setdefault("local_epochs", 1)
    config.setdefault("momentum", 0)
    config.setdefault("weight_decay", 0)

    # === Multi-client aggregation support: compute the number of participating clients ===
    total_clients = int(os.environ.get("NUM_TOTAL_CLIENTS", "100"))
    fraction_fit = config.get("client_fraction", float(os.environ.get("FRACTION_FIT", "0.1")))
    min_fit_clients = int(os.environ.get("MIN_FIT_CLIENTS", "10"))

    expected_clients = max(int(total_clients * fraction_fit), min_fit_clients)
    config["num_clients_in_round"] = expected_clients
    config["total_clients"] = total_clients

    # === Handle client-level DP noise_multiplier configuration first ===
    if config.get("use_dp"):
        dp_mode = config.get("dp_mode", "opacus").lower()
        mech = config.get("noise_mechanism", "gaussian").lower()
        if dp_mode == "client" and mech == "gaussian":
            nm = config.get("noise_multiplier", None)
            if nm is not None:
                config["noise_multiplier"] = float(nm)
                config.pop("round_target_epsilon", None)
                # Fix: Don't override delta if already set in config
                if "delta" not in config:
                    config["delta"] = 1e-5  # Use same default as experiment configs
        elif dp_mode == "client" and mech == "laplace":
            pass

    # === Log configuration (after processing DP parameters) ===
    logger.info(f"Round {server_round}: Sending config for '{mode}' to clients.")
    logger.info(f"  DP enabled: {config.get('use_dp', False)}")
    logger.info(f"  Client fraction: {fraction_fit}")
    logger.info(f"  Clients in round: {expected_clients}")
    if config.get('use_dp'):
        logger.info(f"  dp_mode: {config.get('dp_mode', 'opacus')}")
        logger.info(f"  noise_mechanism: {config.get('noise_mechanism', 'gaussian')}")
        logger.info(f"  max_grad_norm (C): {config.get('max_grad_norm', 'N/A')}")
        logger.info(f"  delta: {config.get('delta', 'N/A')}")

        # Show different parameters based on dp_mode
        if config.get('dp_mode') == "client" and config.get('noise_mechanism') == 'gaussian':
            logger.info(f"  noise_multiplier: {config.get('noise_multiplier', 'N/A')}")
            logger.info(f"  total_epsilon: {config.get('total_epsilon', 'N/A')}")
        else:
            logger.info(f"  round_target_epsilon: {config.get('round_target_epsilon', 'N/A')}")
            logger.info(f"  total_epsilon: {config.get('total_epsilon', 'N/A')}")

        logger.info(f"  batch_size: {config.get('batch_size', 64)}")
        logger.info(f"  local_epochs: {config.get('local_epochs', 1)}")

    # Opacus expects the per-round target under target_epsilon.
    if config.get("dp_mode") in ["opacus", "sample", "client"]:
        if "round_target_epsilon" in config and "target_epsilon" not in config:
            config["target_epsilon"] = config["round_target_epsilon"]

    return config


def fit_metrics_aggregation(results: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate fit metrics from all clients.

    Args:
        results: List of (num_examples, metrics) tuples from clients

    Returns:
        Dict containing aggregated metrics
    """
    if not results:
        logger.warning("No fit results to aggregate")
        return {}

    total_examples = sum(num_examples for num_examples, _ in results)

    aggregated_metrics = {
        "total_fit_examples": total_examples,
        "num_clients_fit": len(results),
    }

    # Type-safe loss aggregation
    if all(("final_test_loss" in metrics) or ("final_train_loss" in metrics) for _, metrics in results):
        def pick_loss(m):
            loss_val = m.get("final_test_loss", m.get("final_train_loss"))
            try:
                return float(loss_val) if loss_val is not None else None
            except (TypeError, ValueError):
                return None

        # Only include valid loss values in aggregation
        valid_loss_results = [(num_examples, pick_loss(metrics))
                              for num_examples, metrics in results
                              if pick_loss(metrics) is not None]

        if valid_loss_results:
            weighted_loss = sum(num_examples * loss for num_examples, loss in valid_loss_results)
            total_valid_examples = sum(num_examples for num_examples, _ in valid_loss_results)
            aggregated_metrics["fit_loss_avg"] = weighted_loss / total_valid_examples

    # Type-safe accuracy aggregation
    if all(("final_test_accuracy" in metrics) or ("final_train_accuracy" in metrics) for _, metrics in results):
        def pick_acc(m):
            acc_val = m.get("final_test_accuracy", m.get("final_train_accuracy"))
            try:
                return float(acc_val) if acc_val is not None else None
            except (TypeError, ValueError):
                return None

        # Only include valid accuracy values in aggregation
        valid_acc_results = [(num_examples, pick_acc(metrics))
                             for num_examples, metrics in results
                             if pick_acc(metrics) is not None]

        if valid_acc_results:
            weighted_accuracy = sum(num_examples * acc for num_examples, acc in valid_acc_results)
            total_valid_acc_examples = sum(num_examples for num_examples, _ in valid_acc_results)
            aggregated_metrics["fit_accuracy_avg"] = weighted_accuracy / total_valid_acc_examples

    first_metrics = results[0][1]
    config_keys = ["learning_rate", "momentum", "weight_decay", "local_epochs", "batch_size"]
    for key in config_keys:
        if key in first_metrics:
            aggregated_metrics[f"config_{key}"] = first_metrics[key]

    privacy_metrics = {}
    # Type-safe privacy epsilon aggregation
    if all("privacy_epsilon" in metrics for _, metrics in results):
        valid_epsilons = []
        valid_epsilon_results = []

        for num_examples, metrics in results:
            eps_val = metrics["privacy_epsilon"]
            try:
                eps_float = float(eps_val)
                valid_epsilons.append(eps_float)
                valid_epsilon_results.append((num_examples, eps_float))
            except (TypeError, ValueError):
                logger.warning(f"Invalid privacy_epsilon value: {eps_val}")
                continue

        if valid_epsilons:
            privacy_metrics["client_max_epsilon"] = max(valid_epsilons)

            weighted_epsilon = sum(num_examples * eps for num_examples, eps in valid_epsilon_results)
            total_valid_epsilon_examples = sum(num_examples for num_examples, _ in valid_epsilon_results)
            privacy_metrics["client_epsilon_avg"] = weighted_epsilon / total_valid_epsilon_examples

    if all("privacy_delta" in metrics for _, metrics in results):
        privacy_metrics["privacy_delta"] = first_metrics["privacy_delta"]

    if all("target_epsilon" in metrics for _, metrics in results):
        privacy_metrics["target_epsilon"] = first_metrics["target_epsilon"]

    # Support new round_target_epsilon key (preferred)
    if all("round_target_epsilon" in metrics for _, metrics in results):
        privacy_metrics["round_target_epsilon"] = first_metrics["round_target_epsilon"]

    if all("noise_multiplier" in metrics for _, metrics in results):
        privacy_metrics["noise_multiplier"] = first_metrics["noise_multiplier"]

    if all("actual_noise_multiplier" in metrics for _, metrics in results):
        privacy_metrics["actual_noise_multiplier"] = first_metrics["actual_noise_multiplier"]

    if all("max_grad_norm" in metrics for _, metrics in results):
        privacy_metrics["max_grad_norm"] = first_metrics["max_grad_norm"]

    if all("sample_rate" in metrics for _, metrics in results):
        avg_sample_rate = sum(metrics["sample_rate"] for _, metrics in results) / len(results)
        privacy_metrics["avg_sample_rate"] = avg_sample_rate

    if all("epsilon_per_step" in metrics for _, metrics in results):
        privacy_metrics["epsilon_per_step"] = first_metrics["epsilon_per_step"]

    if all("scale_parameter" in metrics for _, metrics in results):
        privacy_metrics["scale_parameter"] = first_metrics["scale_parameter"]

    # Enhanced noise mechanism aggregation with statistics
    if any("noise_mechanism" in metrics for _, metrics in results):
        from collections import Counter

        # Collect all noise mechanisms from clients
        mechanisms = [metrics.get("noise_mechanism", "unknown") for _, metrics in results]
        mechanism_counts = Counter(mechanisms)

        # Find the most common mechanism (mode)
        most_common_mechanism = mechanism_counts.most_common(1)[0][0]

        # Store aggregated mechanism info
        privacy_metrics["noise_mechanism"] = most_common_mechanism
        privacy_metrics["noise_mechanism_distribution"] = dict(mechanism_counts)

        # Log mechanism distribution for better visibility
        total_clients = len(results)
        logger.info(f"NOISE MECHANISM DISTRIBUTION ({total_clients} clients):")
        for mechanism, count in mechanism_counts.most_common():
            percentage = (count / total_clients) * 100
            logger.info(f"   - {mechanism}: {count} clients ({percentage:.1f}%)")

        if len(mechanism_counts) > 1:
            logger.warning(f"Mixed mechanisms detected! Most common: {most_common_mechanism}")
        else:
            logger.info(f"Consistent mechanism: {most_common_mechanism}")
    else:
        # No noise mechanism reported - likely baseline or error
        privacy_metrics["noise_mechanism"] = "none"
        logger.info("NOISE MECHANISM: none (baseline or no DP metrics reported)")


    # Add training steps aggregation for privacy accounting verification
    if all("train_steps_this_round" in metrics for _, metrics in results):
        avg_steps = sum(metrics["train_steps_this_round"] for _, metrics in results) / len(results)
        privacy_metrics["avg_train_steps_this_round"] = int(avg_steps)
        # Also keep the first client's value for reference
        privacy_metrics["train_steps_this_round"] = first_metrics["train_steps_this_round"]

    aggregated_metrics.update(privacy_metrics)

    # Aggregate gradient norm diagnostics (for baseline mode)
    # Check for flattened gradient diagnostic keys (grad_median, grad_p95, etc.)
    gradient_diagnostics = {}
    if any("grad_median" in metrics for _, metrics in results):
        clients_with_grad_stats = [(num_examples, metrics) for num_examples, metrics in results if "grad_median" in metrics]

        if clients_with_grad_stats:
            import numpy as np

            # Collect all gradient statistics from clients (using flattened keys)
            all_medians = [metrics["grad_median"] for _, metrics in clients_with_grad_stats]
            all_means = [metrics["grad_mean"] for _, metrics in clients_with_grad_stats]
            all_p95s = [metrics["grad_p95"] for _, metrics in clients_with_grad_stats]
            all_maxes = [metrics["grad_max"] for _, metrics in clients_with_grad_stats]

            # Aggregate statistics across clients
            gradient_diagnostics.update({
                "gradient_median_across_clients": {
                    "min": np.min(all_medians),
                    "max": np.max(all_medians),
                    "mean": np.mean(all_medians),
                    "median": np.median(all_medians)
                },
                "gradient_p95_across_clients": {
                    "min": np.min(all_p95s),
                    "max": np.max(all_p95s),
                    "mean": np.mean(all_p95s),
                    "median": np.median(all_p95s)
                },
                "recommended_max_grad_norm": np.median(all_p95s),  # Median of 95th percentiles
                "num_clients_with_diagnostics": len(clients_with_grad_stats)
            })

            logger.info(f"GRADIENT NORM DIAGNOSTICS AGGREGATED:")
            logger.info(f"  Clients with diagnostics: {gradient_diagnostics['num_clients_with_diagnostics']}")

            # Type-safe logging for gradient diagnostics
            median_val = gradient_diagnostics['gradient_median_across_clients']['median']
            p95_val = gradient_diagnostics['gradient_p95_across_clients']['median']
            rec_val = gradient_diagnostics['recommended_max_grad_norm']

            try:
                median_float = float(median_val)
                logger.info(f"   Median gradient norms across clients: {median_float:.4f}")
            except (TypeError, ValueError):
                logger.info(f"   Median gradient norms across clients: {median_val}")

            try:
                p95_float = float(p95_val)
                logger.info(f"   95th percentile gradient norms across clients: {p95_float:.4f}")
            except (TypeError, ValueError):
                logger.info(f"   95th percentile gradient norms across clients: {p95_val}")

            try:
                rec_float = float(rec_val)
                logger.info(f"  RECOMMENDED max_grad_norm: {rec_float:.4f}")
            except (TypeError, ValueError):
                logger.info(f"  RECOMMENDED max_grad_norm: {rec_val}")

    aggregated_metrics.update(gradient_diagnostics)

    # Aggregate client update L2 norms (from our unified statistics)
    if any("client_update_l2_norm" in metrics for _, metrics in results):
        clients_with_l2_stats = [(num_examples, metrics) for num_examples, metrics in results
                                 if "client_update_l2_norm" in metrics]

        if clients_with_l2_stats:
            import numpy as np

            # Safely extract L2 norms and convert to float
            l2_norms = []
            for _, metrics in clients_with_l2_stats:
                l2_val = metrics["client_update_l2_norm"]
                try:
                    l2_norms.append(float(l2_val))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid client_update_l2_norm value: {l2_val}")
                    continue

            if l2_norms:
                # Compute percentile statistics
                l2_stats = {
                    "client_update_l2_mean": float(np.mean(l2_norms)),
                    "client_update_l2_median": float(np.median(l2_norms)),
                    "client_update_l2_p50": float(np.percentile(l2_norms, 50)),
                    "client_update_l2_p90": float(np.percentile(l2_norms, 90)),
                    "client_update_l2_p95": float(np.percentile(l2_norms, 95)),
                    "client_update_l2_min": float(np.min(l2_norms)),
                    "client_update_l2_max": float(np.max(l2_norms)),
                    "num_clients_with_l2_stats": len(l2_norms)
                }
                aggregated_metrics.update(l2_stats)

                logger.info(f" CLIENT UPDATE L2 NORMS AGGREGATED:")
                logger.info(f"   Clients with L2 stats: {len(l2_norms)}")
                logger.info(f"   Mean: {l2_stats['client_update_l2_mean']:.6f}")
                logger.info(f"   Median (P50): {l2_stats['client_update_l2_median']:.6f}")
                logger.info(f"   P90: {l2_stats['client_update_l2_p90']:.6f}")
                logger.info(f"   P95: {l2_stats['client_update_l2_p95']:.6f}")
                logger.info(f"   Range: [{l2_stats['client_update_l2_min']:.6f}, {l2_stats['client_update_l2_max']:.6f}]")

    if all("train_loss_history" in metrics for _, metrics in results):
        all_loss_histories = [metrics["train_loss_history"] for _, metrics in results]
        if all_loss_histories and all(len(history) > 0 for history in all_loss_histories):
            num_epochs = len(all_loss_histories[0])
            epoch_losses = []

            for epoch in range(num_epochs):
                epoch_weighted_loss = sum(
                    num_examples * all_loss_histories[i][epoch]
                    for i, (num_examples, _) in enumerate(results)
                )
                epoch_losses.append(epoch_weighted_loss / total_examples)

            aggregated_metrics["fit_loss_history"] = epoch_losses
            aggregated_metrics["fit_loss_final"] = epoch_losses[-1] if epoch_losses else 0.0

    logger.info(f"Fit metrics aggregated from {len(results)} clients:")

    # Type-safe logging for loss
    loss_avg = aggregated_metrics.get('fit_loss_avg', None)
    if isinstance(loss_avg, (int, float)):
        logger.info(f"  - Average loss: {loss_avg:.4f}")
    else:
        logger.info(f"  - Average loss: {loss_avg if loss_avg is not None else 'N/A'}")

    # Type-safe logging for accuracy
    acc_avg = aggregated_metrics.get('fit_accuracy_avg', None)
    if isinstance(acc_avg, (int, float)):
        logger.info(f"  - Average accuracy: {acc_avg:.2f}%")
    else:
        logger.info(f"  - Average accuracy: {acc_avg if acc_avg is not None else 'N/A'}")

    if privacy_metrics:
        # Type-safe logging for max client epsilon
        max_eps = privacy_metrics.get("client_max_epsilon", None)
        if isinstance(max_eps, (int, float)):
            logger.info(f"  - Max client epsilon: {max_eps:.4f}")
        else:
            logger.info(f"  - Max client epsilon: {max_eps if max_eps is not None else 'N/A'}")

        # Type-safe logging for average client epsilon
        avg_eps = privacy_metrics.get("client_epsilon_avg", None)
        if isinstance(avg_eps, (int, float)):
            logger.info(f"  - Avg client epsilon: {avg_eps:.4f}")

        # Type-safe logging for privacy delta
        delta = privacy_metrics.get("privacy_delta", None)
        if isinstance(delta, (int, float)):
            logger.info(f"  - Privacy delta: {delta:.1e}")

        # Type-safe logging for target epsilon
        target_eps = privacy_metrics.get("round_target_epsilon", privacy_metrics.get("target_epsilon", None))
        if isinstance(target_eps, (int, float)):
            logger.info(f"  - Target epsilon: {target_eps:.4f}")

        # Type-safe logging for noise multipliers
        noise_mult = privacy_metrics.get("noise_multiplier", None)
        if isinstance(noise_mult, (int, float)):
            logger.info(f"  - Noise multiplier: {noise_mult:.4f}")

        actual_noise_mult = privacy_metrics.get("actual_noise_multiplier", None)
        if isinstance(actual_noise_mult, (int, float)):
            logger.info(f"  - Actual noise multiplier: {actual_noise_mult:.4f}")

        # Type-safe logging for epsilon per step and scale parameter
        eps_per_step = privacy_metrics.get("epsilon_per_step", None)
        if isinstance(eps_per_step, (int, float)):
            logger.info(f"  - Epsilon per step: {eps_per_step:.6f}")

        scale_param = privacy_metrics.get("scale_parameter", None)
        if isinstance(scale_param, (int, float)):
            logger.info(f"  - Scale parameter: {scale_param:.4f}")

        # Type-safe logging for Laplace-specific parameters
        laplace_scale = privacy_metrics.get("laplace_scale", None)
        if isinstance(laplace_scale, (int, float)):
            logger.info(f"  - Laplace scale (b): {laplace_scale:.4f}")

        # Display mechanism distribution summary
        mechanism_dist = privacy_metrics.get("noise_mechanism_distribution", None)
        if mechanism_dist and len(mechanism_dist) > 1:
            logger.info(f"  - Mechanism mix: {mechanism_dist}")

    # Store fit metrics for the final summary.
    global global_metrics_history

    # Infer the current round, with a sequential fallback.
    current_round = len(global_metrics_history["fit_metrics"]) + 1

    fit_history_entry = {
        "round": current_round,
        "loss": aggregated_metrics.get("fit_loss_avg"),
        "accuracy": aggregated_metrics.get("fit_accuracy_avg"),
        "num_clients": aggregated_metrics.get("num_clients_fit"),
        "total_examples": aggregated_metrics.get("total_fit_examples")
    }

    # Include privacy metrics when available.
    if privacy_metrics:
        fit_history_entry.update({
            "max_epsilon": privacy_metrics.get("client_max_epsilon"),
            "avg_epsilon": privacy_metrics.get("client_epsilon_avg"),
            "noise_mechanism": privacy_metrics.get("noise_mechanism"),
            "target_epsilon": privacy_metrics.get("round_target_epsilon") or privacy_metrics.get("target_epsilon")
        })

    global_metrics_history["fit_metrics"].append(fit_history_entry)

    return aggregated_metrics


def aggregate_evaluate_metrics(metrics):
    """
    Aggregate evaluation metrics from all clients using weighted averaging.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients

    Returns:
        Dict containing aggregated evaluation metrics (weighted by num_examples)
    """
    if not metrics:
        return {"accuracy": None, "loss": None}

    # Weight client metrics by sample count.
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        logger.warning("Total validation examples is 0, cannot compute weighted average")
        return {"accuracy": None, "loss": None}

    # Weighted average accuracy
    weighted_accuracy = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
        if "accuracy" in m
    ) / total_examples

    # Weighted average loss
    weighted_loss = sum(
        num_examples * m["loss"]
        for num_examples, m in metrics
        if "loss" in m
    ) / total_examples

    aggregated = {
        "accuracy": float(weighted_accuracy),
        "loss": float(weighted_loss),
    }

    # Log aggregated evaluation metrics.
    logger.info(f"Evaluate metrics aggregated from {len(metrics)} clients (weighted by {total_examples} samples):")
    if aggregated["accuracy"] is not None:
        logger.info(f"  - Weighted average accuracy: {aggregated['accuracy']:.2f}%")
    else:
        logger.info(f"  - Weighted average accuracy: N/A")

    if aggregated["loss"] is not None:
        logger.info(f"  - Weighted average loss: {aggregated['loss']:.4f}")
    else:
        logger.info(f"  - Weighted average loss: N/A")

    # Store evaluation metrics for the final summary.
    global global_metrics_history

    # Infer the current round.
    current_round = len(global_metrics_history["evaluate_metrics"]) + 1

    evaluate_history_entry = {
        "round": current_round,
        "loss": aggregated["loss"],
        "accuracy": aggregated["accuracy"],
        "num_clients": len(metrics),
        "total_validation_samples": total_examples
    }

    global_metrics_history["evaluate_metrics"].append(evaluate_history_entry)

    return aggregated


def print_experiment_summary():
    """
    Print the complete experiment summary.
    """
    global global_metrics_history

    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    fit_metrics = global_metrics_history["fit_metrics"]
    evaluate_metrics = global_metrics_history["evaluate_metrics"]

    if not fit_metrics and not evaluate_metrics:
        logger.info("  No metrics collected during experiment")
        return

    # === FIT METRICS SUMMARY ===
    if fit_metrics:
        logger.info("\nFIT METRICS (Training):")
        logger.info("-" * 60)

        for entry in fit_metrics:
            round_num = entry["round"]
            loss = entry.get("loss")
            accuracy = entry.get("accuracy")
            num_clients = entry.get("num_clients", "N/A")

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"

            logger.info(f"  Round {round_num:2d}: Loss={loss_str:>8}, Accuracy={acc_str:>8}, Clients={num_clients}")

            # Display privacy information when available.
            max_eps = entry.get("max_epsilon")
            if max_eps is not None:
                mechanism = entry.get("noise_mechanism", "unknown")
                logger.info(f"           Privacy: epsilon_max={max_eps:.4f} ({mechanism})")

    # === EVALUATE METRICS SUMMARY ===
    if evaluate_metrics:
        logger.info("\nEVALUATE METRICS (Validation):")
        logger.info("-" * 60)

        for entry in evaluate_metrics:
            round_num = entry["round"]
            loss = entry.get("loss")
            accuracy = entry.get("accuracy")
            num_clients = entry.get("num_clients", "N/A")

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"

            logger.info(f"  Round {round_num:2d}: Loss={loss_str:>8}, Accuracy={acc_str:>8}, Clients={num_clients}")

    # === PERFORMANCE TRENDS ===
    logger.info("\nPERFORMANCE TRENDS:")
    logger.info("-" * 60)

    if len(fit_metrics) >= 2:
        first_fit = fit_metrics[0]
        last_fit = fit_metrics[-1]

        if first_fit.get("loss") is not None and last_fit.get("loss") is not None:
            loss_change = last_fit["loss"] - first_fit["loss"]
            loss_trend = "Improved" if loss_change < 0 else "Worsened" if loss_change > 0 else "Stable"
            logger.info(f"  Training Loss: {first_fit['loss']:.4f} -> {last_fit['loss']:.4f} ({loss_change:+.4f}) {loss_trend}")

        if first_fit.get("accuracy") is not None and last_fit.get("accuracy") is not None:
            acc_change = last_fit["accuracy"] - first_fit["accuracy"]
            acc_trend = "Improved" if acc_change > 0 else "Worsened" if acc_change < 0 else "Stable"
            logger.info(f"  Training Accuracy: {first_fit['accuracy']:.2f}% -> {last_fit['accuracy']:.2f}% ({acc_change:+.2f}%) {acc_trend}")

    if len(evaluate_metrics) >= 2:
        first_eval = evaluate_metrics[0]
        last_eval = evaluate_metrics[-1]

        if first_eval.get("loss") is not None and last_eval.get("loss") is not None:
            loss_change = last_eval["loss"] - first_eval["loss"]
            loss_trend = "Improved" if loss_change < 0 else "Worsened" if loss_change > 0 else "Stable"
            logger.info(f"  Validation Loss: {first_eval['loss']:.4f} -> {last_eval['loss']:.4f} ({loss_change:+.4f}) {loss_trend}")

        if first_eval.get("accuracy") is not None and last_eval.get("accuracy") is not None:
            acc_change = last_eval["accuracy"] - first_eval["accuracy"]
            acc_trend = "Improved" if acc_change > 0 else "Worsened" if acc_change < 0 else "Stable"
            logger.info(f"  Validation Accuracy: {first_eval['accuracy']:.2f}% -> {last_eval['accuracy']:.2f}% ({acc_change:+.2f}%) {acc_trend}")

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)


class DPFedAvg(FedAvg):
    def __init__(self, privacy_accountant: PrivacyAccountant, **kwargs):
        super().__init__(**kwargs)
        self.privacy_accountant = privacy_accountant
        self.total_rounds = None

        # Initialize client-level Gaussian RDP accountant
        self.client_rdp = ClientGaussianRDPAccountant()
        logger.info("[DPFedAvg] Initialized ClientGaussianRDPAccountant")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:

        if failures and os.environ.get("ALLOW_CLIENT_FAILURES", "0") != "1":
            raise RuntimeError(
                f"Round {server_round} failed because {len(failures)} client(s) failed"
            )

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if not results:
            return aggregated_parameters, aggregated_metrics

        # Get first client's metrics for reference
        first_metrics = results[0][1].metrics

        # === Privacy Accounting ===
        privacy_update = _update_privacy_accounting(
            self.privacy_accountant, server_round, results
        )

        # === Client-level Gaussian RDP Accounting ===
        if privacy_update.get("dp_mode") == "client" and first_metrics.get("noise_mechanism") == "gaussian":
            q = privacy_update.get("client_sample_rate")
            sigma = privacy_update.get("noise_multiplier")
            delta = privacy_update.get("delta", 1e-5)

            if q is not None and sigma is not None:
                self.client_rdp.add_round(q=float(q), sigma=float(sigma), steps=1)
                rdp_epsilon = self.client_rdp.get_epsilon(delta=float(delta))

                logger.info(f"[Server Round {server_round}] Client Gaussian RDP: q={q:.4f}, sigma={sigma}, epsilon_cumulative={rdp_epsilon:.6f}")

                # Add RDP epsilon to metrics
                privacy_update["client_gaussian_epsilon_cumulative"] = rdp_epsilon
                privacy_update["cumulative_epsilon"] = rdp_epsilon
                privacy_update["cumulative_delta"] = float(delta)

        # Add the privacy update to our metrics
        aggregated_metrics.update(privacy_update)

        # Copy relevant metrics from client for logging
        for k in ["noise_multiplier", "actual_noise_multiplier", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[k] = first_metrics[k]

        # Print the summary after the final evaluation round.
        if self.total_rounds is not None and server_round >= self.total_rounds:
            # Delay briefly so evaluation metrics are recorded first.
            import threading
            def delayed_summary():
                import time
                time.sleep(1)
                print_experiment_summary()

            threading.Thread(target=delayed_summary, daemon=True).start()

        return aggregated_parameters, aggregated_metrics


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Create server components with privacy budget monitoring and reproducible settings."""

    server_seed = context.run_config.get("seed", 2025)
    seed_mgr = reset_seed_manager(server_seed)
    seed_mgr.set_global_seed()
    seed_mgr.print_seed_summary()

    logger.info(f"[SEED] MASTER_SEED={server_seed}")
    logger.info(f"[INIT] weights_seed={seed_mgr.get_init_weights_seed()}, split_seed={seed_mgr.get_data_split_seed()}")

    # fix_server_seed(server_seed)

    eps_limit = os.environ.get("EPSILON_LIMIT", None)
    eps_limit = float(eps_limit) if eps_limit is not None else None

    # Infer global epsilon limit from experiment configuration if not explicitly set
    if eps_limit is None:
        mode = os.environ.get("EXPERIMENT_MODE", "baseline")
        try:
            config = get_experiment_config(mode)
            # Look for privacy budget information in the config
            if config.get('use_dp', False):
                eps_limit = config.get("total_epsilon")
                logger.info(f"Inferred epsilon limit from config: {eps_limit}")
            else:
                eps_limit = None  # Baseline mode - no privacy limit
        except ValueError:
            logger.warning(f"Could not load config for EXPERIMENT_MODE='{mode}', using default epsilon_limit=None")
            eps_limit = None

    # Get aggregation method configuration and client fraction from config
    mode = os.environ.get("EXPERIMENT_MODE", "baseline")
    try:
        config = get_experiment_config(mode)

        if config.get("dp_mode") in ["opacus", "sample", "client"]:
            total_eps = config.get("total_epsilon", None)
            num_rounds = config.get("privacy_calibration_rounds", 500)
            if total_eps and not config.get("round_target_epsilon"):
                config["round_target_epsilon"] = total_eps / num_rounds
                logger.info(f"Auto-calculated round_target_epsilon: {config['round_target_epsilon']:.6f} (total_epsilon={total_eps} / num_rounds={num_rounds})")

        aggregation_method = config.get("aggregation_method", "fedavg").lower()
        client_fraction = config.get("client_fraction", 1.0)
    except ValueError:
        aggregation_method = os.environ.get("AGGREGATION_METHOD", "fedavg").lower()
        client_fraction = 1.0

    privacy_accountant = PrivacyAccountant(
        target_delta=1e-5
    )
    # Store epsilon_limit separately for the helper function
    privacy_accountant.epsilon_limit = eps_limit

    # Derive availability thresholds from the configured client population.
    total_clients = int(os.environ.get("NUM_TOTAL_CLIENTS", "100"))

    # Store total_clients in accountant for client_sample_rate calculation
    privacy_accountant.total_clients = total_clients

    expected_clients = max(int(total_clients * client_fraction), 1)

    # Allow a small availability gap unless overridden by the environment.
    min_available = int(os.environ.get("MIN_AVAILABLE_CLIENTS", str(max(10, int(expected_clients * 0.95)))))

    # Apply the same threshold to fit and evaluation.
    min_fit = int(os.environ.get("MIN_FIT_CLIENTS", str(max(10, int(expected_clients * 0.9)))))
    min_eval = int(os.environ.get("MIN_EVALUATE_CLIENTS", str(max(10, int(expected_clients * 0.9)))))

    logger.info(f"Dynamic client settings - Expected: {expected_clients}, Min available: {min_available}, Min fit: {min_fit}, Min eval: {min_eval}")

    # Use reproducible initial weights (load from disk if exists, create with fixed seed if not)
    initial_weight_arrays = get_or_create_initial_weights(MLP, num_classes=10, seed=server_seed)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weight_arrays)

    # Select appropriate strategy based on aggregation method
    if aggregation_method == "fedadam" or aggregation_method == "adam":
        beta_1 = float(os.environ.get("ADAM_BETA1", "0.9"))
        beta_2 = float(os.environ.get("ADAM_BETA2", "0.999"))
        eta = float(os.environ.get("ADAM_ETA", "1e-3"))
        tau = float(os.environ.get("ADAM_TAU", "1e-9"))

        strategy = DPFedAdam(
            initial_parameters=initial_parameters,
            privacy_accountant=privacy_accountant,
            beta_1=beta_1,
            beta_2=beta_2,
            eta=eta,
            tau=tau,
            fraction_fit=client_fraction,
            fraction_evaluate=0.1,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=min_available,
            accept_failures=False,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
        logger.info(f"Server initialized with FedAdam strategy (beta1={beta_1}, beta2={beta_2}, eta={eta}, tau={tau}, client_fraction={client_fraction})")

    elif aggregation_method == "krum":
        f = int(os.environ.get("KRUM_F", "1"))
        multi_krum = os.environ.get("KRUM_MULTI", "false").lower() == "true"

        strategy = DPKrum(
            initial_parameters=initial_parameters,
            privacy_accountant=privacy_accountant,
            f=f,
            multi_krum=multi_krum,
            fraction_fit=client_fraction,
            fraction_evaluate=0.1,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=min_available,
            accept_failures=False,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
        krum_type = "Multi" if multi_krum else "Single"
        selected_clients = "n-f" if multi_krum else "1"
        logger.info(f"Server initialized with {krum_type} Krum strategy (f={f}, selects {selected_clients} clients, client_fraction={client_fraction})")

    else:
        strategy = DPFedAvg(
            initial_parameters=initial_parameters,
            privacy_accountant=privacy_accountant,
            fraction_fit=client_fraction,
            fraction_evaluate=0.1,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=min_available,
            accept_failures=False,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
        logger.info(f"Server initialized with FedAvg strategy (client_fraction={client_fraction})")

    # Use fewer rounds for more realistic per-round budget
    # Bound round duration so failed clients cannot block indefinitely.
    rt = int(os.environ.get("ROUND_TIMEOUT", "0"))
    total_rounds = int(os.environ.get("NUM_ROUNDS", "500"))
    server_cfg = fl.server.ServerConfig(
        num_rounds=total_rounds,
        round_timeout=rt if rt > 0 else None,
    )

    # Store the total round count for final-summary detection.
    if hasattr(strategy, 'total_rounds'):
        strategy.total_rounds = total_rounds

    logger.info(f"Server components initialized with privacy monitor (num_rounds={server_cfg.num_rounds}, round_timeout={server_cfg.round_timeout}, epsilon_limit={eps_limit})")
    return ServerAppComponents(strategy=strategy, config=server_cfg)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    logger.info("Flower server ready to start...")
    # fl.server.start_server is deprecated, use fl.simulation.start_simulation or other drivers.
    # The `flwr run` command will handle this.
