# server.py

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import flwr as fl
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Metrics, Context, ndarrays_to_parameters, parameters_to_ndarrays
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils import SimpleCNN, get_model_parameters



logger = logging.getLogger(__name__)

def weighted_average(results: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    """Compute weighted average of arrays based on number of examples."""
    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_weights = [
        num_examples * weights for num_examples, weights in results
    ]
    return np.sum(weighted_weights, axis=0) / total_examples

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
    
    def __init__(self, privacy_monitor, beta_1: float = 0.9, beta_2: float = 0.999, 
                 eta: float = 1e-3, tau: float = 1e-9, **kwargs):
        super().__init__(**kwargs)
        self.privacy_monitor = privacy_monitor
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
        
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]
        
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
        
        aggregated_metrics = {}
        
        round_epsilons = []
        for _, fit_res in results:
            eps = fit_res.metrics.get("privacy_epsilon", None)
            if isinstance(eps, (int, float)) and eps is not None:
                round_epsilons.append(float(eps))

        summary = self.privacy_monitor.update_with_round(round_epsilons)

        aggregated_metrics["round_max_epsilon"] = float(summary["round_max_epsilon"])
        aggregated_metrics["cumulative_epsilon"] = float(summary["cumulative_epsilon"])
        aggregated_metrics["cumulative_delta"] = float(summary["cumulative_delta"])
        aggregated_metrics["total_dp_rounds"] = int(summary["total_dp_rounds"])
        aggregated_metrics["privacy_breached"] = bool(summary["breached"])

        first_metrics = results[0][1].metrics
        for k in ["noise_multiplier", "actual_noise_multiplier", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[f"{k}"] = first_metrics[k]

        logger.info(f"FedAdam Round {server_round}: Aggregated with beta1={self.beta_1}, beta2={self.beta_2}, eta={self.eta}")
        
        if round_epsilons:
            logger.info(
                "[PRIVACY] Round %d: round_max_epsilon=%.4f | cumulative_epsilon=%.4f | δ=%s | dp_rounds=%d%s",
                server_round,
                aggregated_metrics["round_max_epsilon"],
                aggregated_metrics["cumulative_epsilon"],
                f"{self.privacy_monitor.delta:.1e}",
                aggregated_metrics["total_dp_rounds"],
                " | BREACHED!" if aggregated_metrics["privacy_breached"] else "",
            )

        return aggregated_parameters, aggregated_metrics

class DPKrum(FedAvg):
    """Krum aggregation with differential privacy support."""
    
    def __init__(self, privacy_monitor, f: int = 1, multi_krum: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.privacy_monitor = privacy_monitor
        self.f = f  # Number of Byzantine clients to tolerate
        self.multi_krum = multi_krum  
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        
        if not results:
            return None, {}
        
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]
        
        aggregated_weights = krum_aggregate(weights_results, self.f, self.multi_krum)
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        
        aggregated_metrics = {}
        
        round_epsilons = []
        for _, fit_res in results:
            eps = fit_res.metrics.get("privacy_epsilon", None)
            if isinstance(eps, (int, float)) and eps is not None:
                round_epsilons.append(float(eps))

        summary = self.privacy_monitor.update_with_round(round_epsilons)

        aggregated_metrics["round_max_epsilon"] = float(summary["round_max_epsilon"])
        aggregated_metrics["cumulative_epsilon"] = float(summary["cumulative_epsilon"])
        aggregated_metrics["cumulative_delta"] = float(summary["cumulative_delta"])
        aggregated_metrics["total_dp_rounds"] = int(summary["total_dp_rounds"])
        aggregated_metrics["privacy_breached"] = bool(summary["breached"])

        first_metrics = results[0][1].metrics
        for k in ["noise_multiplier", "actual_noise_multiplier", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[f"{k}"] = first_metrics[k]

        krum_type = "Multi" if self.multi_krum else "Single"
        logger.info(f"{krum_type} Krum Round {server_round}: Aggregated {len(results)} clients with f={self.f}")
        
        if round_epsilons:
            logger.info(
                "[PRIVACY] Round %d: round_max_epsilon=%.4f | cumulative_epsilon=%.4f | δ=%s | dp_rounds=%d%s",
                server_round,
                aggregated_metrics["round_max_epsilon"],
                aggregated_metrics["cumulative_epsilon"],
                f"{self.privacy_monitor.delta:.1e}",
                aggregated_metrics["total_dp_rounds"],
                " | BREACHED!" if aggregated_metrics["privacy_breached"] else "",
            )

        return aggregated_parameters, aggregated_metrics

class PrivacyBudgetMonitor:
    def __init__(self, delta: float, epsilon_limit: Optional[float] = None):
        self.delta = float(delta)
        self.epsilon_limit = float(epsilon_limit) if epsilon_limit is not None else None
        self.cumulative_epsilon = 0.0
        self.total_dp_rounds = 0

    def update_with_round(self, round_epsilons: List[float]) -> Dict[str, float]:
        if not round_epsilons:
            return {
                "round_max_epsilon": 0.0,
                "cumulative_epsilon": self.cumulative_epsilon,
                "cumulative_delta": self.delta,
                "total_dp_rounds": self.total_dp_rounds,
                "breached": False,
            }

        round_max = max(round_epsilons)
        self.cumulative_epsilon += float(round_max)
        self.total_dp_rounds += 1

        breached = False
        if self.epsilon_limit is not None and self.cumulative_epsilon >= self.epsilon_limit:
            breached = True
            logger.warning(
                f"[PRIVACY] Cumulative epsilon {self.cumulative_epsilon:.4f} "
                f"has reached/exceeded the limit {self.epsilon_limit:.4f} (δ={self.delta:.1e})."
            )

        return {
            "round_max_epsilon": round_max,
            "cumulative_epsilon": self.cumulative_epsilon,
            "cumulative_delta": self.delta,
            "total_dp_rounds": self.total_dp_rounds,
            "breached": breached,
        }

def fit_config(server_round: int) -> Dict:
    """Return training configuration based on the experiment mode."""
    
    mode = os.environ.get("EXPERIMENT_MODE", "baseline").lower()
    aggregation_method = os.environ.get("AGGREGATION_METHOD", "fedavg").lower()  #     # Added aggregation method configuration
    client_optimizer = os.environ.get("CLIENT_OPTIMIZER", "sgd").lower()  # Added client optimizer configuration

    config = {
        "server_round": server_round,
        "local_epochs": 2, 
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "use_dp": True,     
        "delta": 1e-5,
        "max_grad_norm": 1.2,
        "aggregation_method": aggregation_method,  
        "optimizer": client_optimizer,  
    }
    
    if client_optimizer == "adam":
        config["adam_beta1"] = float(os.environ.get("CLIENT_ADAM_BETA1", "0.9"))
        config["adam_beta2"] = float(os.environ.get("CLIENT_ADAM_BETA2", "0.999"))
        config["adam_eps"] = float(os.environ.get("CLIENT_ADAM_EPS", "1e-8"))

    if mode == 'baseline':
        config["use_dp"] = False
        logger.info(f"Running in BASELINE mode.")
        
    elif mode == 'low_privacy':
        config["target_epsilon"] = 0.15  # target_epsilon=15.0, this value accumulates each round
        logger.info(f"Running in LOW PRIVACY mode (target_epsilon: {config['target_epsilon']}).")
        
    elif mode == 'medium_privacy':
        config["target_epsilon"] = 0.08  # target_epsilon=8.0, this value accumulates each round
        logger.info(f"Running in MEDIUM PRIVACY mode (target_epsilon: {config['target_epsilon']}).")
        
    elif mode == 'high_privacy':
        config["target_epsilon"] = 0.04  # target_epsilon=4.0, this value accumulates each round
        logger.info(f"Running in HIGH PRIVACY mode (target_epsilon: {config['target_epsilon']}).")
        
    elif mode == 'laplace_low_privacy':
        config["noise_mechanism"] = "laplace"
        config["epsilon_per_step"] = 0.004687  # ~15.0 total epsilon for 3200 steps
        config["target_epsilon"] = 15.0  
        logger.info(f"Running in LAPLACE LOW PRIVACY mode (epsilon_per_step: {config['epsilon_per_step']}).")
        
    elif mode == 'laplace_medium_privacy':
        config["noise_mechanism"] = "laplace"
        config["epsilon_per_step"] = 0.0025  # ~8.0 total epsilon for 3200 steps
        config["target_epsilon"] = 8.0  
        logger.info(f"Running in LAPLACE MEDIUM PRIVACY mode (epsilon_per_step: {config['epsilon_per_step']}).")
        
    elif mode == 'laplace_high_privacy':
        config["noise_mechanism"] = "laplace"  
        config["epsilon_per_step"] = 0.00125  # ~4.0 total epsilon for 3200 steps
        config["target_epsilon"] = 4.0  
        logger.info(f"Running in LAPLACE HIGH PRIVACY mode (epsilon_per_step: {config['epsilon_per_step']}).")
        
    else:
        raise ValueError(f"Unknown EXPERIMENT_MODE: {mode}")

    logger.info(f"Round {server_round}: Sending config to clients: {config}")
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
    
    if all("final_train_loss" in metrics for _, metrics in results):
        weighted_loss = sum(
            num_examples * metrics["final_train_loss"] 
            for num_examples, metrics in results
        )
        aggregated_metrics["fit_loss_avg"] = weighted_loss / total_examples
    
    if all("final_train_accuracy" in metrics for _, metrics in results):
        weighted_accuracy = sum(
            num_examples * metrics["final_train_accuracy"] 
            for num_examples, metrics in results
        )
        aggregated_metrics["fit_accuracy_avg"] = weighted_accuracy / total_examples
    
    first_metrics = results[0][1]
    config_keys = ["learning_rate", "momentum", "weight_decay", "local_epochs", "batch_size"]
    for key in config_keys:
        if key in first_metrics:
            aggregated_metrics[f"config_{key}"] = first_metrics[key]
    
    privacy_metrics = {}
    if all("privacy_epsilon" in metrics for _, metrics in results):
        max_epsilon = max(metrics["privacy_epsilon"] for _, metrics in results)
        privacy_metrics["client_max_epsilon"] = max_epsilon
        
        weighted_epsilon = sum(
            num_examples * metrics["privacy_epsilon"] 
            for num_examples, metrics in results
        )
        privacy_metrics["client_epsilon_avg"] = weighted_epsilon / total_examples
    
    if all("privacy_delta" in metrics for _, metrics in results):
        privacy_metrics["privacy_delta"] = first_metrics["privacy_delta"]

    if all("target_epsilon" in metrics for _, metrics in results):
        privacy_metrics["target_epsilon"] = first_metrics["target_epsilon"]
    
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
        
    if all("noise_mechanism" in metrics for _, metrics in results):
        privacy_metrics["noise_mechanism"] = first_metrics["noise_mechanism"]

    aggregated_metrics.update(privacy_metrics)
    
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
    logger.info(f"  - Average train loss: {aggregated_metrics.get('fit_loss_avg', 'N/A'):.4f}")
    logger.info(f"  - Average train accuracy: {aggregated_metrics.get('fit_accuracy_avg', 'N/A'):.2f}%")
    if privacy_metrics:
        logger.info(f"  - Max client epsilon: {privacy_metrics.get('client_max_epsilon', 'N/A'):.4f}")
    
    return aggregated_metrics


class DPFedAvg(FedAvg):
    def __init__(self, privacy_monitor: PrivacyBudgetMonitor, **kwargs):
        super().__init__(**kwargs)
        self.privacy_monitor = privacy_monitor

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if not results:
            return aggregated_parameters, aggregated_metrics

        round_epsilons = []
        for _, fit_res in results:
            eps = fit_res.metrics.get("privacy_epsilon", None)
            if isinstance(eps, (int, float)) and eps is not None:
                round_epsilons.append(float(eps))

        summary = self.privacy_monitor.update_with_round(round_epsilons)

        aggregated_metrics["round_max_epsilon"] = float(summary["round_max_epsilon"])
        aggregated_metrics["cumulative_epsilon"] = float(summary["cumulative_epsilon"])
        aggregated_metrics["cumulative_delta"] = float(summary["cumulative_delta"])
        aggregated_metrics["total_dp_rounds"] = int(summary["total_dp_rounds"])
        aggregated_metrics["privacy_breached"] = bool(summary["breached"])

        first_metrics = results[0][1].metrics
        for k in ["noise_multiplier", "actual_noise_multiplier", "target_epsilon", "max_grad_norm", "sample_rate", "local_epochs", "batch_size"]:
            if k in first_metrics:
                aggregated_metrics[f"{k}"] = first_metrics[k]

        if round_epsilons:
            logger.info(
                "[PRIVACY] Round %d: round_max_epsilon=%.4f | cumulative_epsilon=%.4f | δ=%s | dp_rounds=%d%s",
                server_round,
                aggregated_metrics["round_max_epsilon"],
                aggregated_metrics["cumulative_epsilon"],
                f"{self.privacy_monitor.delta:.1e}",
                aggregated_metrics["total_dp_rounds"],
                " | BREACHED!" if aggregated_metrics["privacy_breached"] else "",
            )
        else:
            logger.info(
                "[PRIVACY] Round %d: non-DP round (no client epsilon reported). "
                "cumulative_epsilon=%.4f | δ=%s | dp_rounds=%d",
                server_round,
                aggregated_metrics["cumulative_epsilon"],
                f"{self.privacy_monitor.delta:.1e}",
                aggregated_metrics["total_dp_rounds"],
            )

        return aggregated_parameters, aggregated_metrics

def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    eps_limit = os.environ.get("EPSILON_LIMIT", None)
    eps_limit = float(eps_limit) if eps_limit is not None else None
    
    # Get aggregation method configuration
    aggregation_method = os.environ.get("AGGREGATION_METHOD", "fedavg").lower()

    privacy_monitor = PrivacyBudgetMonitor(delta=1e-5, epsilon_limit=eps_limit)
    initial_model = SimpleCNN(num_classes=10) 
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_model_parameters(initial_model)
    )
    
    # Select appropriate strategy based on aggregation method
    if aggregation_method == "fedadam" or aggregation_method == "adam":
        beta_1 = float(os.environ.get("ADAM_BETA1", "0.9"))
        beta_2 = float(os.environ.get("ADAM_BETA2", "0.999"))
        eta = float(os.environ.get("ADAM_ETA", "1e-3"))
        tau = float(os.environ.get("ADAM_TAU", "1e-9"))
        
        strategy = DPFedAdam(
            initial_parameters=initial_parameters,
            privacy_monitor=privacy_monitor,
            beta_1=beta_1,
            beta_2=beta_2,
            eta=eta,
            tau=tau,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=100,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )
        logger.info(f"Server initialized with FedAdam strategy (beta1={beta_1}, beta2={beta_2}, eta={eta}, tau={tau})")
        
    elif aggregation_method == "krum":
        f = int(os.environ.get("KRUM_F", "1")) 
        multi_krum = os.environ.get("KRUM_MULTI", "false").lower() == "true"
        
        strategy = DPKrum(
            initial_parameters=initial_parameters,
            privacy_monitor=privacy_monitor,
            f=f,
            multi_krum=multi_krum,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=100,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )
        krum_type = "Multi" if multi_krum else "Single"
        selected_clients = "n-f" if multi_krum else "1"
        logger.info(f"Server initialized with {krum_type} Krum strategy (f={f}, selects {selected_clients} clients)")
        
    else:  
        strategy = DPFedAvg(
            initial_parameters=initial_parameters,
            privacy_monitor=privacy_monitor,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=100,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )
        logger.info("Server initialized with FedAvg strategy")

    config = fl.server.ServerConfig(num_rounds=int(os.environ.get("NUM_ROUNDS", "100")))
    logger.info(f"Server components initialized with display-only privacy monitor (num_rounds={config.num_rounds})")
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    logger.info("Flower server ready to start...")
    # fl.server.start_server is deprecated, use fl.simulation.start_simulation or other drivers.
    # The `flwr run` command will handle this.