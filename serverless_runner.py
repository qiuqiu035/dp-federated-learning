# serverless/unified_pytorch_runner.py
"""
Serverless PyTorch Runner - Uses unified DP implementation from client.py
Ensures identical training logic and DP mechanisms with Server version

Features:
1. Fully reuses FlowerClient and DP Mod components from client.py
2. Supports sample-level (Opacus) and client-level (LocalDpMod) DP
3. Supports Gaussian and Laplace noise mechanisms
4. Adaptive clipping: dynamic adjustment of max_grad_norm (C)
5. Unified DP calibration: auto-calculates noise parameters based on total privacy budget
6. Privacy accounting: automatic tracking of cumulative privacy loss
"""

import sys
import os
import logging
import io
import json
import time
import numpy as np
import torch
from privacy_accountant import ClientGaussianRDPAccountant

def flush_print(*args, **kwargs):
    """Force flush print function"""
    print(*args, **kwargs)
    sys.stdout.flush()

# Collect per-round metrics for the final summary.
serverless_metrics_history = {
    "fit_metrics": [],
    "evaluate_metrics": []
}

def print_serverless_experiment_summary(final_rdp_epsilon=None):
    """
    Print the complete serverless experiment summary.
    """
    global serverless_metrics_history

    flush_print("=" * 80)
    flush_print("SERVERLESS EXPERIMENT SUMMARY")
    flush_print("=" * 80)

    fit_metrics = serverless_metrics_history["fit_metrics"]
    evaluate_metrics = serverless_metrics_history["evaluate_metrics"]

    if not fit_metrics and not evaluate_metrics:
        flush_print("  No metrics collected during serverless experiment")
        return

    # === FIT METRICS SUMMARY ===
    if fit_metrics:
        flush_print("\nFIT METRICS (Decentralized Training):")
        flush_print("-" * 60)

        for entry in fit_metrics:
            round_num = entry["round"]
            success = entry.get("successful_clients", "N/A")
            failed = entry.get("failed_clients", 0)
            total = entry.get("total_clients", "N/A")

            flush_print(f"  Round {round_num:2d}: Successful={success:>3}/{total}, Failed={failed}")

            # Display privacy information when available.
            if entry.get("dp_enabled", False):
                dp_mode = entry.get("dp_mode", "unknown")
                dp_mechanism = entry.get("dp_mechanism", "unknown")
                round_eps = entry.get("round_epsilon")

                flush_print(f"           DP: {dp_mode} mode, {dp_mechanism} mechanism", end="")
                if round_eps is not None:
                    flush_print(f", epsilon_round={round_eps:.4f}")
                else:
                    flush_print("")
            else:
                flush_print("           DP: Disabled (baseline)")

    # === EVALUATE METRICS SUMMARY ===
    if evaluate_metrics:
        flush_print("\nEVALUATE METRICS (POST-MIX Evaluation):")
        flush_print("-" * 60)

        for entry in evaluate_metrics:
            round_num = entry["round"]
            loss = entry.get("loss")
            accuracy = entry.get("accuracy")
            num_clients = entry.get("num_clients", "N/A")

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"

            flush_print(f"  Round {round_num:2d}: Loss={loss_str:>8}, Accuracy={acc_str:>8}, Clients={num_clients}")

    # === PERFORMANCE TRENDS ===
    flush_print("\nPERFORMANCE TRENDS:")
    flush_print("-" * 60)

    if len(evaluate_metrics) >= 2:
        first_eval = evaluate_metrics[0]
        last_eval = evaluate_metrics[-1]

        if first_eval.get("loss") is not None and last_eval.get("loss") is not None:
            loss_change = last_eval["loss"] - first_eval["loss"]
            loss_trend = "Improved" if loss_change < 0 else "Worsened" if loss_change > 0 else "Stable"
            flush_print(f"  Validation Loss: {first_eval['loss']:.4f} -> {last_eval['loss']:.4f} ({loss_change:+.4f}) {loss_trend}")

        if first_eval.get("accuracy") is not None and last_eval.get("accuracy") is not None:
            acc_change = last_eval["accuracy"] - first_eval["accuracy"]
            acc_trend = "Improved" if acc_change > 0 else "Worsened" if acc_change < 0 else "Stable"
            flush_print(f"  Validation Accuracy: {first_eval['accuracy']:.2f}% -> {last_eval['accuracy']:.2f}% ({acc_change:+.2f}%) {acc_trend}")

    # === TRAINING SUCCESS RATE ===
    if fit_metrics:
        flush_print("\nTRAINING SUCCESS RATE:")
        flush_print("-" * 60)

        total_attempts = sum(entry.get("total_clients", 0) for entry in fit_metrics)
        total_successes = sum(entry.get("successful_clients", 0) for entry in fit_metrics)

        if total_attempts > 0:
            success_rate = (total_successes / total_attempts) * 100
            flush_print(f"  Overall Success Rate: {total_successes}/{total_attempts} ({success_rate:.1f}%)")

        # Display the final-round success rate.
        if fit_metrics:
            last_fit = fit_metrics[-1]
            last_success = last_fit.get("successful_clients", 0)
            last_total = last_fit.get("total_clients", 0)
            if last_total > 0:
                last_rate = (last_success / last_total) * 100
                flush_print(f"  Final Round Success Rate: {last_success}/{last_total} ({last_rate:.1f}%)")

    flush_print("\n" + "=" * 80)

    # === Final Privacy Summary ===
    if final_rdp_epsilon is not None:
        flush_print("FINAL PRIVACY CONSUMPTION:")
        flush_print(f"  Client-level Gaussian RDP: epsilon_cumulative = {final_rdp_epsilon:.6f}")
        flush_print("=" * 80)

    flush_print("SERVERLESS EXPERIMENT COMPLETED")
    flush_print("=" * 80)

os.environ['PYTHONUNBUFFERED'] = '1'
# The current POST-MIX coordinator executes clients sequentially and does not
# use the legacy Ray remote function below. Keep it disabled so importing this
# module cannot auto-start a Ray runtime and exhaust a login/check job.
RAY_AVAILABLE = False
ray = None


# Keep runner logs together with experiment output.
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stdout
)
os.environ["RAY_DISABLE_LOGGING"] = "1"
VERBOSE = False

# === Device selection (added for serverless safety) ===
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ServerlessRunner] Using device: {DEVICE}")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from privacy_accountant import PrivacyAccountant
from pathlib import Path

from client import (
    FlowerClient,
    fix_client_seed,
    load_model,
)
from localdp_adapter import create_localdp_adapter
from flwr.common import FitIns, parameters_to_ndarrays, ndarrays_to_parameters
from utils import load_datasets, get_model_parameters, set_model_parameters, MLP
from utils import ServerlessNeighborAverager  # Import topology class

from utils import prepare_mnist_partitions, prepare_cifar10_partitions

# Import get_or_create_initial_weights from server.py for consistent initialization
import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
from server import get_or_create_initial_weights

# Import seed manager for unified randomness control
from seed_manager import get_seed_manager, reset_seed_manager, set_deterministic_mode

# The legacy flwr_serverless package initializes Ray during import and this
# optional alias is not used by the current in-memory POST-MIX flow.
SharedFolderLike = None

class LocalFolder:
    """Local filesystem version of SharedFolder"""
    def __init__(self, root: str = "./_serverless_share"):
        import os
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def put(self, key: str, nds: List[np.ndarray], meta: Dict):
        path = f"{self.root}/{key}.npz"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with io.BytesIO() as buf:
            np.savez_compressed(buf, *nds)
            data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        with open(f"{self.root}/{key}.json", "w") as f:
            json.dump(meta, f)

    def get(self, key: str) -> Tuple[List[np.ndarray], Dict]:
        """Get data based on specific key"""
        path = f"{self.root}/{key}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Key {key} not found")

        with np.load(path, allow_pickle=False) as d:
            nds = [d[k] for k in d.files]
        with open(f"{self.root}/{key}.json", "r") as f:
            meta = json.load(f)
        return nds, meta

    def get_latest(self, prefix: str) -> Tuple[List[np.ndarray], Dict]:
        """Get the latest data matching the prefix"""
        import glob, os
        files = sorted(glob.glob(f"{self.root}/{prefix}*.npz"))
        if not files:
            raise FileNotFoundError(f"No files found with prefix {prefix}")
        path = files[-1]
        key = os.path.splitext(os.path.basename(path))[0]
        return self.get(key)

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys matching the prefix"""
        import glob, os
        files = glob.glob(f"{self.root}/{prefix}*.npz")
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

# -----------------------------
# Aggregator: FedAvg
# -----------------------------
def fedavg_weights(list_of_nds: List[List[np.ndarray]],
                   example_counts: List[int]) -> List[np.ndarray]:
    if not list_of_nds:
        raise ValueError("Empty client list for aggregation")
    if len(list_of_nds) != len(example_counts):
        raise ValueError("Lengths mismatch between client weights and example counts")

    total = float(sum(example_counts))
    w = np.array([c / total for c in example_counts], dtype=np.float64)  # shape [K]

    out = []
    for params in zip(*list_of_nds):          # params: list of arrays from K clients (same shape)
        stacked = np.stack(params, axis=0)    # [K, ...]
        w_broadcast = w.reshape((w.shape[0],) + (1,) * (stacked.ndim - 1))
        out.append((stacked * w_broadcast).sum(axis=0))
    return out

def average_params(list_of_params: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Simple unweighted averaging of parameters for neighbor aggregation

    Args:
        list_of_params: List of parameter lists, each containing np.ndarrays

    Returns:
        Averaged parameters as list of np.ndarrays
    """
    if not list_of_params:
        raise ValueError("Empty parameter list for averaging")

    num_params = len(list_of_params[0])
    averaged = []

    for i in range(num_params):
        # Stack parameters from all clients for this layer
        layer_params = [params[i] for params in list_of_params]
        stacked = np.stack(layer_params, axis=0)  # [num_clients, ...]
        # Simple average
        averaged.append(np.mean(stacked, axis=0))

    return averaged

# -----------------------------
# Unified DP Configuration
# -----------------------------
@dataclass
class UnifiedDPConfig:
    """Fully consistent DP configuration with client.py (simplified version, based on direct epsilon definition)"""
    enable_dp: bool = False          # Whether to enable DP (regardless of sample/client)
    mode: str = "client"             # "client" (client-level) / "sample" (sample-level Opacus)
    mechanism: str = "gaussian"      # "gaussian" / "laplace" / "none"
    C: float = 1.0
    total_epsilon: float = 1.0
    sensitivity: Optional[float] = None
    noise_multiplier: Optional[float] = None
    round_epsilon: Optional[float] = None
    delta: float = 1e-5

# -----------------------------
# Unified Serverless Client
# -----------------------------
class UnifiedServerlessClient:
    """
    Serverless client using FlowerClient and all Mods from client.py
    Ensures identical training logic and DP implementation with Server version
    """
    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        num_clients: int,
        dp_config: UnifiedDPConfig,
        shared_folder,
        seed: int = 0,
        batch_size: int = 64,
        local_epochs: int = 1,
        experiment_config: dict = None,  # complete experiment configuration
        use_dynamic_neighbors: bool = True,  # whether to use dynamic neighbors
        non_iid: bool = True,  # Non-IID data partitioning flag
        alpha: float = 0.2,  # Dirichlet concentration parameter
    ):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.dp_config = dp_config
        self.shared = shared_folder
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.experiment_config = experiment_config or {}  # Save experiment configuration
        self.use_dynamic_neighbors = use_dynamic_neighbors  # Save dynamic neighbor configuration
        self.non_iid = non_iid  # Save Non-IID configuration
        self.alpha = alpha  # Save alpha configuration

        # Set client seed (consistent with server version)
        fix_client_seed(seed, client_id)

        # Log Non-IID configuration for this client
        data_mode = f"Non-IID (alpha={alpha})" if non_iid else "IID"
        print(f"[ServerlessClient {client_id}]  Initializing with data_mode={data_mode} (non_iid={non_iid}, alpha={alpha})")

        # Create FlowerClient instance (reuse complete logic, explicitly pass total partition count)
        self.flower_client = FlowerClient(partition_id=client_id, num_partitions=num_clients, non_iid=non_iid, alpha=alpha)

        # Apply DP Mod and create wrapped client
        self._apply_dp_mods()

        # Initialize privacy accountant (disable auto-save)
        self.privacy_accountant = None
        if dp_config.enable_dp:
            # Disable file saving: pass None to avoid generating numerous JSON files
            self.privacy_accountant = PrivacyAccountant(save_path=None)
            print(f"[ServerlessClient {client_id}] Privacy accountant initialized (file saving disabled)")

        # Initialize neighbor topology (Ring Topology)
        self.neighbors = self._init_neighbors(client_id, num_clients)
        if VERBOSE:
            print(f"[ServerlessClient {client_id}] Neighbors: {self.neighbors}, Mode: DECENTRALIZED")
            print(f"[ServerlessClient {client_id}] Initialized with DP: {dp_config.enable_dp}, mechanism: {dp_config.mechanism}, mode: {dp_config.mode}")

    def _init_neighbors(self, client_id: int, num_clients: int) -> List[int]:
        """
        Initialize neighbor topology (Ring Topology)

        Args:
            client_id: Current client ID
            num_clients: Total number of clients

        Returns:
            List of neighbor client IDs
        """
        # Ring topology: each client connects to previous and next
        if num_clients == 1:
            return []  # No neighbors when only one client
        elif num_clients == 2:
            return [1 - client_id]  # Two clients are neighbors to each other
        else:
            prev_neighbor = (client_id - 1) % num_clients
            next_neighbor = (client_id + 1) % num_clients
            return [prev_neighbor, next_neighbor]

    def set_dynamic_neighbors(self, neighbor_ids: List[int]):
        """Set neighbor ID list passed by scheduler each round"""
        self.neighbors = neighbor_ids
        if VERBOSE:
            print(f"[Client {self.client_id}] Dynamic neighbors set: {self.neighbors}")

    def save_model_to_shared_dir(self, model, round_num: int):
        """
        [DEPRECATED] Save local model to shared directory for neighbor reading
        New synchronous neighbor averaging flow no longer uses file sharing, changed to in-memory parameter passing
        """
        # This method is no longer used in the new synchronous neighbor averaging approach
        pass

    def load_models_from_neighbors(self, round_num: int) -> Dict[int, Dict]:
        """
        [DEPRECATED] Load model state dict from neighbors (supports multi-layer fallback to round -1)
        New synchronous neighbor averaging flow no longer uses file sharing, changed to in-memory parameter passing
        """
        # This method is no longer used in the new synchronous neighbor averaging approach
        return {}

    def _extract_params_aligned(self, neighbor_state_dict: Dict) -> List[np.ndarray]:
        """
        [DEPRECATED] Safely extract parameters from neighbor state_dict
        New synchronous neighbor averaging flow no longer uses file sharing, changed to in-memory parameter passing
        """
        # This method is no longer used in the new synchronous neighbor averaging approach
        return []

    def aggregate_neighbors(self, local_model, neighbor_models: Dict[int, Dict]) -> None:
        """
        [DEPRECATED] Aggregate neighbor models using Metropolis weights
        New synchronous neighbor averaging flow changed to neighbor averaging before round start
        """
        # This method is no longer used in the new synchronous neighbor averaging approach
        pass

    def _calculate_round_epsilon(self) -> float:
        """Calculate privacy consumption epsilon per round"""
        if not self.dp_config.enable_dp:
            return 0.0

        if self.dp_config.mechanism.lower() == "laplace":
            if hasattr(self.dp_config, 'epsilon'):
                return self.dp_config.epsilon
            elif hasattr(self.dp_config, 'total_epsilon') and 'num_rounds' in self.experiment_config:
                total_rounds = self.experiment_config.get('privacy_calibration_rounds', 500)
                return self.dp_config.total_epsilon / total_rounds
            else:
                # Default value
                return 0.1
        else:
            # For Gaussian mechanism, need to calculate from noise multiplier and other parameters
            # Simplified handling here, can be improved as needed
            if hasattr(self.dp_config, 'epsilon'):
                return self.dp_config.epsilon
            else:
                return 0.1  # Default value

    def _apply_dp_mods(self):
        """Apply client.py DP logic directly in serverless mode (using only self-developed implementation)"""
        base = self.flower_client.to_client()

        # First check if DP is enabled, return directly if not
        if not self.dp_config.enable_dp:
            self.client_impl = base
            return

        # Map UnifiedDPConfig to config format expected by client.py
        # Use original dp_mode from experiment config for client routing logic
        original_dp_mode = self.experiment_config.get("dp_mode", "opacus")
        config = {
            "dp_mode": original_dp_mode,  # Keep original mode ("opacus" or "client") for client.py logic
            "use_dp": self.dp_config.enable_dp,
        }

        # --- Configure downstream distribution by modexmechanism ---
        # Note: client.py uses config["dp_mode"] to decide between Opacus and LocalDpMod paths
        config["dp_mechanism"] = self.dp_config.mechanism.lower()       # "gaussian" / "laplace"
        config["max_grad_norm"] = float(self.dp_config.C)

        if config["dp_mechanism"] == "gaussian":
            # Use self.dp_config.mode (converted mode) to determine parameter handling
            if self.dp_config.mode == "client":
                # Client-level Gaussian: only send multiplier, not round_epsilon
                if self.dp_config.noise_multiplier is None:
                    raise ValueError("client-level Gaussian requires noise_multiplier")
                config["noise_multiplier"] = float(self.dp_config.noise_multiplier)
                config.pop("round_target_epsilon", None)
                config["delta"] = float(self.dp_config.delta)           # As accounting reference, can be kept
            else:
                # Sample-level Gaussian: only send round_epsilon (or total/num_rounds), not multiplier
                if self.dp_config.round_epsilon is not None:
                    config["round_target_epsilon"] = float(self.dp_config.round_epsilon)
                elif self.dp_config.total_epsilon is not None and self.experiment_config.get("num_rounds"):
                    config["round_target_epsilon"] = float(self.dp_config.total_epsilon) / float(self.experiment_config["num_rounds"])
                else:
                    raise ValueError("sample-level Gaussian requires round_epsilon or total_epsilon/num_rounds")
                config["delta"] = float(self.dp_config.delta)
                config.pop("noise_multiplier", None)

        elif config["dp_mechanism"] == "laplace":
            # Laplace mechanism handling (both sample-level and client-level use epsilon)
            if self.dp_config.round_epsilon is not None:
                config["round_target_epsilon"] = float(self.dp_config.round_epsilon)
            elif self.dp_config.total_epsilon is not None and self.experiment_config.get("num_rounds"):
                config["round_target_epsilon"] = float(self.dp_config.total_epsilon) / float(self.experiment_config["num_rounds"])
            else:
                raise ValueError("laplace requires round_epsilon or total_epsilon/num_rounds")
            config.pop("noise_multiplier", None)
            config["delta"] = 0.0

        # Add necessary parameters (from experiment configuration)
        config.update({
            "total_epsilon": self.experiment_config.get("total_epsilon", None),
            "client_level_dp": True,
            "total_clients": self.experiment_config.get("num_clients", 10),  # Add total client count
        })

        partition_id = 0   # serverless has no partition concept, use 0 as placeholder

        dp_mode = config.get("dp_mode", "opacus")
        use_dp = config.get("use_dp", False)

        # Only use LocalDpMod adapter for client-level DP
        # For sample-level DP (opacus), use FlowerClient's Opacus implementation directly
        if not use_dp or dp_mode == "opacus":
            self.client_impl = base
            return

        max_grad_norm = config.get("max_grad_norm", 1.0)
        noise_mechanism = config.get("dp_mechanism", "gaussian")
        noise_multiplier = config.get("noise_multiplier", None)

        # Use new LocalDpMod adapter (compatible with original LocalDpFixedMod functionality)
        # Note: scale_mode related parameters removed to ensure client-level DP consistency

        # Set auto_calculate_noise based on mode
        if config["dp_mode"] == "client" and config["dp_mechanism"] == "gaussian":
            # Client-level Gaussian: auto_calculate_noise=False if noise_multiplier provided
            epsilon = 1.0  # Placeholder value, not used for noise calculation
            auto_calculate_noise = noise_multiplier is None
        else:
            # Sample-level Gaussian / Laplace: auto_calculate_noise=True (need round_epsilon to take effect)
            epsilon = config.get("round_target_epsilon", 1.0)
            auto_calculate_noise = True

        delta = config.get("delta", 1e-5)
        if delta is None:
            # Use delta=0 for Laplace mechanism, default value for Gaussian mechanism
            delta = 0.0 if noise_mechanism == "laplace" else 1e-5

        client_mod = create_localdp_adapter(
            mechanism=noise_mechanism,
            max_grad_norm=max_grad_norm,
            epsilon=epsilon,
            delta=delta,
            partition_id=self.client_id,
            noise_list=None,  # Use automatic calculation
            auto_calculate_noise=auto_calculate_noise,  # Dynamically set based on mode
            noise_multiplier=noise_multiplier,
        )

        self.client_impl = client_mod.wrap(base)

        # Improved logging: show noise_multiplier in client+gaussian to avoid confusion
        if config["dp_mode"] == "client" and config["dp_mechanism"] == "gaussian":
            nm_info = f"noise_multiplier={config.get('noise_multiplier', 'N/A')}"
            print(f"[ServerlessClient {self.client_id}] Using LocalDpModAdapter (mode={config['dp_mode']}, mechanism={noise_mechanism}, {nm_info})")
        else:
            print(f"[ServerlessClient {self.client_id}] Using LocalDpModAdapter (mode={config['dp_mode']}, mechanism={noise_mechanism}, epsilon={epsilon:.4f})")

    def _sample_level_dp_training(self, model, weights_before, config):
        """Use Opacus for sample-level DP training"""
        from opacus_client_dp import OpacusClientDP
        from client import flatten_weights, unflatten_to_state_dict, compute_client_update

        # Apply pre-training weights
        set_model_parameters(model, weights_before)

        # Read optimizer parameters from config (aligned with client.py)
        learning_rate = config.get("learning_rate", 0.001)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 0.0001)
        optimizer_type = config.get("optimizer", "sgd").lower()

        # Create optimizer based on config
        if optimizer_type == "adam":
            beta1 = config.get("adam_beta1", 0.9)
            beta2 = config.get("adam_beta2", 0.999)
            eps = config.get("adam_eps", 1e-8)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
            print(f"[ServerlessClient {self.client_id}] Sample-level DP: Using Adam optimizer (lr={learning_rate}, beta1={beta1}, beta2={beta2}, weight_decay={weight_decay})")
        else:  # sgd
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
            print(f"[ServerlessClient {self.client_id}] Sample-level DP: Using SGD optimizer (lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay})")

        # Calculate training steps and sampling rate
        num_examples = len(self.flower_client.train_loader.dataset)
        batch_size = config["batch_size"]
        local_epochs = config["local_epochs"]
        steps_per_epoch = len(self.flower_client.train_loader)
        total_steps = steps_per_epoch * local_epochs
        # Sampling rates are probabilities and must not exceed one.
        sample_rate = min(1.0, float(batch_size) / max(1, num_examples))


        # Set Opacus DP engine
        if self.dp_config.mechanism == "gaussian":
            dp_engine = OpacusClientDP(
                max_grad_norm=self.dp_config.C,
                delta=self.dp_config.delta,
                epochs=local_epochs,
                sample_rate=sample_rate,
                strict_mode=False,
                experimental_mode=True,
            )
            # Opacus wrapping - Gaussian branch with target_epsilon
            private_model, private_optimizer, private_data_loader = dp_engine.attach(
                model=model,
                optimizer=optimizer,
                data_loader=self.flower_client.train_loader,
                target_epsilon=self.dp_config.round_epsilon  # Directly pass epsilon
            )
        else:  # laplace
            # Total steps per round
            steps_per_epoch = len(self.flower_client.train_loader)
            total_steps = steps_per_epoch * local_epochs

            # Guard the per-step epsilon calculation against division by zero.
            total_steps = max(total_steps, 1)
            epsilon_per_step = max(self.dp_config.round_epsilon, 1e-12) / float(total_steps)

            print(f"[Sample DP] L2-clipping + coordinate-wise Laplace per-step eps={epsilon_per_step:.6f}")

            dp_engine = OpacusClientDP(
                max_grad_norm=self.dp_config.C,
                sample_rate=sample_rate,
                noise_mechanism="laplace",
                epsilon_per_step=epsilon_per_step,  # Laplace mechanism uses epsilon_per_step
                epochs=local_epochs,
                strict_mode=False,
                delta=0.0,  # Laplace is pure epsilon-DP
            )

            # Opacus wrapping - Laplace branch
            private_model, private_optimizer, private_data_loader = dp_engine.attach(
                model=model,
                optimizer=optimizer,
                data_loader=self.flower_client.train_loader
            )

        # Execute training
        private_model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(private_data_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)

                private_optimizer.zero_grad()
                output = private_model(data)
                loss = criterion(output, target)
                loss.backward()
                private_optimizer.step()

        weights_after = get_model_parameters(private_model)

        set_model_parameters(model, weights_after)

        spent = dp_engine.get_privacy_spent()
        # Ensure consumed_epsilon has a reasonable value, add stronger protection
        round_epsilon = self.dp_config.round_epsilon
        if round_epsilon is None:
            round_epsilon = 0.1  # Safe default value
        epsilon_from_spent = spent.get("epsilon")
        if epsilon_from_spent is None:
            consumed_epsilon = round_epsilon
        else:
            consumed_epsilon = float(epsilon_from_spent)


        if self.dp_config.mechanism == "gaussian":
            metrics = {
                "consumed_epsilon": consumed_epsilon,
                "noise_mechanism": self.dp_config.mechanism,
                "clip_norm": self.dp_config.C,
                "epsilon_per_round": self.dp_config.round_epsilon,
                "delta": self.dp_config.delta,
                "train_steps_this_round": total_steps,
                "dp_mode": "sample",
                "sample_rate": spent.get("sample_rate"),
                "actual_noise_multiplier": spent.get("actual_noise_multiplier") or spent.get("noise_multiplier"),
            }
        else:  # laplace
            # Calculate noise scale for recording (although actual calculation is inside OpacusClientDP)
            noise_scale = self.dp_config.C / epsilon_per_step
            metrics = {
                "consumed_epsilon": self.dp_config.round_epsilon,  # Pure epsilon-DP accumulation can use round_epsilon
                "noise_mechanism": "laplace",
                "clip_norm": self.dp_config.C,
                "epsilon_per_round": self.dp_config.round_epsilon,
                "epsilon_per_step": epsilon_per_step,
                "delta": 0.0,
                "train_steps_this_round": total_steps,
                "dp_mode": "sample",
                "noise_scale": noise_scale,  # L2-clipping + coordinate-wise Laplace scale
            }

        # === Detach DP engine to allow reattachment in next round ===
        dp_engine.detach()
        print(f"[Sample DP] DP engine detached successfully")

        return weights_after, num_examples, metrics

    def fit_one_round_with_init_params(self, round_idx: int, init_params: List[np.ndarray],
                                     num_clients_in_round: int = 1) -> List[np.ndarray]:
        """
        New simplified training interface: start training local epochs from given initial parameters

        Args:
            round_idx: Current round index
            init_params: Initial parameters (from neighbor averaging)
            num_clients_in_round: Number of clients participating in this round

        Returns:
            List of parameters after training
        """
        # 1. Create model and load initial parameters
        from client import load_model, DATASET_NAME
        model = load_model(DATASET_NAME).to(DEVICE)
        set_model_parameters(model, init_params)

        # 2. Configure training parameters
        config = {
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "server_round": round_idx,
            "num_clients_in_round": num_clients_in_round,
            "total_clients": self.num_clients,
        }

        # 3. Add DP configuration (if enabled)
        if self.dp_config.enable_dp:
            current_c = self.dp_config.C

            config.update({
                "use_dp": True,
                "dp_mode": self.dp_config.mode,
                "max_grad_norm": current_c,
                "noise_mechanism": self.dp_config.mechanism,
                "delta": self.dp_config.delta,
            })

            # Only add noise_multiplier when client+gaussian to avoid confusion
            if (self.dp_config.mode == "client" and
                self.dp_config.mechanism == "gaussian" and
                self.dp_config.noise_multiplier is not None):
                config["noise_multiplier"] = self.dp_config.noise_multiplier

            if self.dp_config.sensitivity is not None:
                config["sensitivity"] = self.dp_config.sensitivity
            else:
                config["sensitivity"] = current_c

        # 4. Execute training
        if self.dp_config.enable_dp and self.dp_config.mode == "sample":
            # Sample-level DP
            weights_before = get_model_parameters(model)
            weights_after, num_examples, metrics = self._sample_level_dp_training(
                model, weights_before, config
            )
        elif self.dp_config.enable_dp and self.dp_config.mode == "client":
            # Client-level DP
            weights_before = get_model_parameters(model)
            client = self.client_impl
            res = client.fit(FitIns(parameters=ndarrays_to_parameters(weights_before), config=config))
            weights_after = parameters_to_ndarrays(res.parameters)
            num_examples = res.num_examples
            metrics = res.metrics
        else:
            # No DP
            weights_before = get_model_parameters(model)
            client = self.client_impl
            res = client.fit(FitIns(parameters=ndarrays_to_parameters(weights_before), config=config))
            weights_after = parameters_to_ndarrays(res.parameters)
            num_examples = res.num_examples
            metrics = res.metrics

        # 5. Privacy accounting (if DP enabled)
        if self.privacy_accountant and self.dp_config.enable_dp:
            consumed_eps = None
            if isinstance(metrics, dict):
                consumed_eps = metrics.get("consumed_epsilon", None)

            if self.dp_config.mechanism.lower() == "laplace":
                round_eps = float(metrics.get("consumed_epsilon", self.dp_config.round_epsilon))
                self.privacy_accountant.add_laplace_event(round_idx, epsilon=round_eps, client_id=self.client_id)
            elif self.dp_config.mechanism.lower() == "gaussian":
                if self.dp_config.mode.lower() == "sample":
                    if consumed_eps is not None:
                        self.privacy_accountant.add_client_level_gaussian_event(
                            round_num=round_idx,
                            epsilon=float(consumed_eps),
                            delta=float(metrics.get("delta", self.dp_config.delta)),
                            client_id=self.client_id,
                        )
                        noise_multiplier = None
                    else:
                        noise_multiplier = metrics.get("actual_noise_multiplier") or metrics.get("noise_multiplier")
                    sample_rate = metrics.get("sample_rate")
                    steps = int(metrics.get("train_steps_this_round", 1))

                    if noise_multiplier is not None and sample_rate is not None:
                        for _ in range(steps):
                            self.privacy_accountant.add_gaussian_event(
                                round_num=round_idx,
                                noise_multiplier=float(noise_multiplier),
                                sampling_probability=float(sample_rate),
                                client_id=self.client_id
                            )
                elif self.dp_config.mode.lower() == "client":
                    round_epsilon = self.dp_config.round_epsilon
                    round_delta = self.dp_config.delta

                    if round_epsilon is not None and round_delta is not None:
                        self.privacy_accountant.add_client_level_gaussian_event(
                            round_num=round_idx,
                            epsilon=float(round_epsilon),
                            delta=float(round_delta),
                            client_id=self.client_id
                        )

        # Report local training metrics.
        if isinstance(metrics, dict):
            flush_print(f"[Client {self.client_id}] Round {round_idx + 1} training completed:")

            # Report training loss.
            train_loss = metrics.get("final_train_loss") or metrics.get("final_test_loss")
            if train_loss is not None:
                flush_print(f"  - Training loss: {train_loss:.4f}")

            # Report training accuracy.
            train_acc = metrics.get("final_train_accuracy") or metrics.get("final_test_accuracy")
            if train_acc is not None:
                flush_print(f"  - Training accuracy: {train_acc:.2f}%")

            # Report privacy consumption.
            if self.dp_config.enable_dp:
                consumed_eps = metrics.get("consumed_epsilon") or metrics.get("privacy_epsilon")
                if consumed_eps is not None:
                    flush_print(f"  - Privacy consumed (epsilon): {consumed_eps:.6f}")

                noise_mechanism = metrics.get("noise_mechanism", "unknown")
                flush_print(f"  - DP mechanism: {noise_mechanism}")

        return weights_after, num_examples

    def one_round(self, round_idx: int, num_clients_in_round: int = 1) -> Dict:
        """
        [DEPRECATED] Execute one round of training, supporting pure decentralized mode
        New synchronous neighbor averaging flow uses fit_one_round_with_init_params method
        This method is retained for compatibility but not used in new architecture
        """
        print(f"[WARNING] Client {self.client_id}: Using deprecated one_round method - new flow uses fit_one_round_with_init_params")

        if self.client_id == 0 or VERBOSE:
            print(f"[Client {self.client_id}] Starting round {round_idx} (Decentralized mode)")

        # 1. Create model
        from client import load_model, DATASET_NAME
        model = load_model(DATASET_NAME).to(DEVICE)

        # 2. Load own previous model as starting point
        if round_idx > 0:
            try:
                prev_model_path = f"shared_models/round_{round_idx-1}/client_{self.client_id}.pt"
                if os.path.exists(prev_model_path):
                    model.load_state_dict(torch.load(prev_model_path, map_location=DEVICE))
                    if VERBOSE:
                        print(f"[Client {self.client_id}] Loaded own previous model from round {round_idx-1}")
                else:
                    # If the current client has never participated before, try global initialization or random initialization
                    if round_idx == 0:
                        init_weights, _ = self.shared.get("global_round_0")
                        set_model_parameters(model, init_weights)
                        print(f"[Client {self.client_id}] Warning: Previous model not found, starting from initial weights.")
                    else:
                        # Try to inherit from the most recent round's model (stronger global stability)
                        found = False
                        for r in range(round_idx-1, -1, -1):
                            alt_path = f"shared_models/round_{r}/client_{self.client_id}.pt"
                            if os.path.exists(alt_path):
                                model.load_state_dict(torch.load(alt_path, map_location=DEVICE))
                                print(f"[Client {self.client_id}] Loaded historical model from round {r}")
                                found = True
                                break
                        if not found:
                            init_weights, _ = self.shared.get("global_round_0")
                            set_model_parameters(model, init_weights)
                            print(f"[Client {self.client_id}] No history found, starting from initial weights.")

                # Load models from neighbors and aggregate
                neighbor_models = self.load_models_from_neighbors(round_idx - 1)
                self.aggregate_neighbors(model, neighbor_models)

            except Exception as e:
                print(f"[Client {self.client_id}] Warning: Failed during decentralized aggregation start: {e}. Starting from initial weights.")
                init_weights, _ = self.shared.get("global_round_0")
                set_model_parameters(model, init_weights)
        else: # round_idx == 0
            # Round 0: All clients start from the same initial weights
            init_weights, _ = self.shared.get("global_round_0")
            set_model_parameters(model, init_weights)
            print(f"[Client {self.client_id}] Round 0: Starting from initial global weights.")

        # 3. Execute local training
        config = {
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "server_round": round_idx,
            "num_clients_in_round": num_clients_in_round,  # Add support for multi-client aggregation
            "total_clients": self.num_clients,  # Add total client count for client_sample_rate calculation
        }

        # === Add current round's DP configuration ===
        if self.dp_config.enable_dp:
            current_c = self.dp_config.C  # Get current C value (may have been adaptively adjusted)

            config.update({
                "use_dp": True,
                "dp_mode": self.dp_config.mode,
                "max_grad_norm": current_c,  # Use current latest C value (clip threshold)
                "noise_mechanism": self.dp_config.mechanism,
                "delta": self.dp_config.delta,
            })

            # Only add noise_multiplier for client+gaussian to avoid confusion
            if (self.dp_config.mode == "client" and
                self.dp_config.mechanism == "gaussian" and
                self.dp_config.noise_multiplier is not None):
                config["noise_multiplier"] = self.dp_config.noise_multiplier

            # Self-check and log verification
            print(f"[ServerlessConfig] dp_mode={self.dp_config.mode}, mechanism={self.dp_config.mechanism}, C={current_c}, noise_multiplier={self.dp_config.noise_multiplier}")

            # Add sensitivity parameter (if specified in config)
            if self.dp_config.sensitivity is not None:
                config["sensitivity"] = self.dp_config.sensitivity
                print(f"[Client {self.client_id}] DP Config: C={current_c:.6f}, sensitivity={self.dp_config.sensitivity:.6f} (custom)")
            else:
                # Default sensitivity = C (for backward compatibility)
                config["sensitivity"] = current_c
                print(f"[Client {self.client_id}] DP Config: C={current_c:.6f}, sensitivity={current_c:.6f} (default=C)")

        if self.dp_config.enable_dp and self.dp_config.mode == "sample":
            # Sample-level DP: Use Opacus to wrap training
            if VERBOSE:
                print(f"[Client {self.client_id}] Using SAMPLE-LEVEL DP training")
            weights_before = get_model_parameters(model)
            weights_after, num_examples, metrics = self._sample_level_dp_training(
                model, weights_before, config
            )
        elif self.dp_config.enable_dp and self.dp_config.mode == "client":
            # Client-level DP: Use Flower pipeline (will trigger LocalDpMod / LaplaceLocalDpMod)
            if VERBOSE:
                print(f"[Client {self.client_id}] Using CLIENT-LEVEL DP training")
            weights_before = get_model_parameters(model)
            client = self.client_impl
            res = client.fit(FitIns(parameters=ndarrays_to_parameters(weights_before), config=config))
            weights_after = parameters_to_ndarrays(res.parameters)
            num_examples = res.num_examples
            metrics = res.metrics
        else:
            if VERBOSE:
                print(f"[Client {self.client_id}] Using NO DP training")
            weights_before = get_model_parameters(model)
            client = self.client_impl
            res = client.fit(FitIns(parameters=ndarrays_to_parameters(weights_before), config=config))
            weights_after = parameters_to_ndarrays(res.parameters)
            num_examples = res.num_examples
            metrics = res.metrics

        set_model_parameters(model, weights_after)

        self.save_model_to_shared_dir(model, round_idx)
        if VERBOSE:
            print(f"[Client {self.client_id}] Round {round_idx} completed, saved model for neighbors.")

        # Record privacy cost with the accountant for the selected mechanism.
        if self.privacy_accountant and self.dp_config.enable_dp:
            consumed_eps = None
            if isinstance(metrics, dict):
                consumed_eps = metrics.get("consumed_epsilon", None)

            if self.dp_config.mechanism.lower() == "laplace":
                round_eps = float(metrics.get("consumed_epsilon", self.dp_config.round_epsilon))
                self.privacy_accountant.add_laplace_event(round_idx, epsilon=round_eps, client_id=self.client_id)
            elif self.dp_config.mechanism.lower() == "gaussian":
                if self.dp_config.mode.lower() == "sample":
                    # Sample-level DP: Use noise_multiplier and sample_rate from Opacus metrics
                    noise_multiplier = metrics.get("actual_noise_multiplier") or metrics.get("noise_multiplier")
                    sample_rate = metrics.get("sample_rate")
                    steps = int(metrics.get("train_steps_this_round", 1))

                    if noise_multiplier is not None and sample_rate is not None:
                        # Account for each step within the round
                        for _ in range(steps):
                            self.privacy_accountant.add_gaussian_event(
                                round_num=round_idx,
                                noise_multiplier=float(noise_multiplier),
                                sampling_probability=float(sample_rate),
                                client_id=self.client_id
                            )
                    else:
                        print(f"[Accountant Warning] Client {self.client_id}: Missing noise_multiplier or sample_rate for sample-level Gaussian accounting.")

                elif self.dp_config.mode.lower() == "client":
                    # Client-level Gaussian: use RDP with noise_multiplier + client_sample_rate
                    nm = metrics.get("actual_noise_multiplier") or metrics.get("noise_multiplier")
                    q  = metrics.get("client_sample_rate", 1.0)
                    if nm is not None:
                        self.privacy_accountant.add_gaussian_event(
                            round_num=round_idx,
                            noise_multiplier=float(nm),
                            sampling_probability=float(q),
                            client_id=self.client_id
                        )
                    else:
                        # fallback: only if metrics missing (shouldn't happen in client-level Gaussian)
                        round_epsilon = self.dp_config.round_epsilon
                        round_delta = self.dp_config.delta
                        if round_epsilon is not None and round_delta is not None:
                            self.privacy_accountant.add_client_level_gaussian_event(
                                round_num=round_idx,
                                epsilon=float(round_epsilon),
                                delta=float(round_delta),
                                client_id=self.client_id
                            )
                        else:
                            print(f"[Accountant Warning] Client {self.client_id}: Missing noise_multiplier for client-level Gaussian RDP accounting.")

            self.privacy_accountant.print_status(round_idx)


        return {
            "client_id": self.client_id,
            "round": round_idx,
            "num_examples": num_examples,
            "metrics": metrics,
        }

# -----------------------------
# Ray parallel training function
# -----------------------------
if RAY_AVAILABLE:
    @ray.remote
    def run_one_client_remote(
        client_id: int,
        round_idx: int,
        neighbor_ids: List[int],
        dataset_name: str,
        num_clients: int,
        dp_config_dict: dict,
        shared_folder_root: str,
        seed: int,
        batch_size: int,
        local_epochs: int,
        experiment_config: dict,
        num_clients_in_round: int,
    ):
        """
        Ray remote function to run one client for one round of training
        1. Dynamically load necessary modules to avoid Ray subprocess import issues
        """
        import sys
        import os
        import torch
        import numpy as np
        import json
        from dataclasses import dataclass
        from typing import Dict, List, Optional

        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            from client import (
                FlowerClient,
                fix_client_seed,
                load_model,
                DEVICE,
            )
            from localdp_adapter import create_localdp_adapter
            from flwr.common import FitIns, parameters_to_ndarrays, ndarrays_to_parameters
            from utils import get_model_parameters, set_model_parameters
            from privacy_accountant import PrivacyAccountant
        except ImportError as e:
            return {
                "client_id": client_id,
                "round": round_idx,
                "error": f"Import failed: {e}",
                "num_examples": 0,
                "metrics": {},
            }

        @dataclass
        class UnifiedDPConfig:
            enable_dp: bool = False
            mode: str = "client"
            mechanism: str = "gaussian"
            C: float = 1.0
            total_epsilon: float = 1.0
            sensitivity: Optional[float] = None
            noise_multiplier: Optional[float] = None
            round_epsilon: Optional[float] = None
            delta: float = 1e-5

        dp_config = UnifiedDPConfig(**dp_config_dict)

        class LocalFolder:
            def __init__(self, root: str):
                self.root = root
                os.makedirs(self.root, exist_ok=True)

            def get(self, key: str):
                path = f"{self.root}/{key}.npz"
                with np.load(path, allow_pickle=False) as d:
                    nds = [d[k] for k in d.files]
                with open(f"{self.root}/{key}.json", "r") as f:
                    meta = json.load(f)
                return nds, meta

        shared_folder = LocalFolder(shared_folder_root)

        client = UnifiedServerlessClient(
            client_id=client_id,
            dataset_name=dataset_name,
            num_clients=num_clients,
            dp_config=dp_config,
            shared_folder=shared_folder,
            seed=seed,
            batch_size=batch_size,
            local_epochs=local_epochs,
            experiment_config=experiment_config,
            use_dynamic_neighbors=True,
        )

        client.set_dynamic_neighbors(neighbor_ids)

        result = client.one_round(round_idx, num_clients_in_round)

        return result
else:
    def run_one_client_remote(*args, **kwargs):
        raise RuntimeError("Ray is not available, cannot use parallel training")

# -----------------------------
# Aggregator and coordination logic
# -----------------------------
# Main function to run unified serverless federated learning
# -----------------------------
def run_unified_serverless_fl(
    dataset_name: str = "mnist",
    num_clients: int = 100,
    num_rounds: int = 3,
    local_epochs: int = 1,
    batch_size: int = 64,
    client_fraction: float = 1.0,
    dp_config: UnifiedDPConfig = None,
    seed: int = 0,
    use_inmemory: bool = True,
    experiment_config: dict = None,
    use_ray_parallel: bool = False,
    # Topology configuration parameters
    serverless_topology: str = "ring",  # "ring" or "ws"
    ws_k: int = 2,  # Watts-Strogatz k parameter (degree)
    ws_p: float = 0.2,  # Watts-Strogatz p parameter (rewiring probability)
    non_iid: bool = True,  # Non-IID data partitioning flag
    alpha: float = 0.1,  # Dirichlet concentration parameter
    results_dir: str = "./results",
):
    """
    Run pure neighbor aggregation-based Serverless Federated Learning.

    NEW: Fixed topology support (ring and Watts-Strogatz)
    - serverless_topology: "ring" or "ws"
    - ws_k: {2, 3, 4} for WS topology
    - ws_p: rewiring probability (default 0.2)
    - Q=1: single gossip step per round
    """
    # ==================== SEED MANAGER INITIALIZATION ====================
    # Initialize unified seed manager for reproducibility
    master_seed = seed if seed != 0 else 2025
    seed_mgr = reset_seed_manager(master_seed)
    seed_mgr.set_global_seed()
    set_deterministic_mode()
    seed_mgr.print_seed_summary()
    flush_print(f"[SEED] Serverless MASTER_SEED={master_seed}")
    # ====================================================================

    # ==================== NON-IID CONFIGURATION ====================
    # Log Non-IID data distribution configuration
    data_mode = f"Non-IID (DirichletPartitioner, alpha={alpha})" if non_iid else "IID (Uniform Random)"
    flush_print(f"[Serverless]  Data Distribution Mode: {data_mode}")
    flush_print(f"[Serverless]  non_iid parameter: {non_iid}, alpha: {alpha}")
    # ================================================================

    # Reset metrics from any previous run in this process.
    global serverless_metrics_history
    serverless_metrics_history = {
        "fit_metrics": [],
        "evaluate_metrics": []
    }

    # Initialize client-level Gaussian RDP accountant
    client_rdp = ClientGaussianRDPAccountant()
    flush_print("[Serverless] Initialized ClientGaussianRDPAccountant")

    # Pre-cache dataset partitions to avoid Ray subprocess conflicts
    print("Pre-checking dataset cache...")
    if dataset_name.lower() == "mnist":
        prepare_mnist_partitions(num_partitions=num_clients, data_dir="./data")
        print("MNIST cache ready.")
    elif dataset_name.lower() == "cifar10":
        prepare_cifar10_partitions(num_partitions=num_clients, data_dir="./data")
        print("CIFAR10 cache ready.")
    else:
        print(f"Dataset {dataset_name} may not have pre-caching support.")



    if dp_config is None:
        dp_config = UnifiedDPConfig()

    # === DP Calibration: Get total privacy budget from dp_config ===
    # Note: total_epsilon should already be set to dp_config by caller
    eps_total = getattr(dp_config, 'total_epsilon', 1.0)  # Get total privacy budget from dp_config
    delta_total = getattr(dp_config, 'delta', 1e-5)
    # Keep privacy calibration tied to the full experiment design. Smoke tests
    # may execute fewer rounds without spending the entire budget in one round.
    calibration_rounds = int(
        (experiment_config or {}).get("privacy_calibration_rounds", num_rounds)
    )
    R = max(1, calibration_rounds)

    # Fixed sensitivity (simplified version no longer uses complex sensitivity parameters)
    dp_config.sensitivity = 1.0

    if dp_config.enable_dp:
        if dp_config.mode == "client":
            # Changed to noise_multiplier priority, no longer evenly distributing epsilon
            if dp_config.noise_multiplier is not None:
                # Use specified noise_multiplier, don't calculate round_epsilon
                dp_config.round_epsilon = None
                dp_config.total_epsilon = eps_total  # Retain but not used for calculation
            else:
                # Fallback: simplified version calculates epsilon budget per round
                if eps_total is not None and R > 0:
                    round_epsilon = eps_total / R
                    dp_config.round_epsilon = round_epsilon
                else:
                    dp_config.round_epsilon = None
            dp_config.delta = delta_total if dp_config.mechanism == "gaussian" else None

            if dp_config.mechanism == "gaussian":
                noise_info = f"noise_multiplier={dp_config.noise_multiplier}" if dp_config.noise_multiplier else f"epsilon_round={dp_config.round_epsilon:.6f}"
                print(f"[DP-Gaussian] epsilon_total={eps_total}, delta_total={delta_total}, R={R}, "
                      f"{noise_info}, sensitivity={dp_config.sensitivity}")
            else:
                noise_info = f"noise_multiplier={dp_config.noise_multiplier}" if dp_config.noise_multiplier else f"epsilon_round={dp_config.round_epsilon:.6f}"
                print(f"[DP-Laplace] epsilon_total={eps_total}, R={R}, "
                      f"{noise_info}, sensitivity={dp_config.sensitivity}")
        elif dp_config.mode == "sample":
            # Sample-level DP: also need to set round_epsilon for Laplace mechanism
            if eps_total is not None and R > 0:
                round_epsilon = eps_total / R
                dp_config.round_epsilon = round_epsilon
            else:
                dp_config.round_epsilon = None
            dp_config.delta = delta_total if dp_config.mechanism == "gaussian" else None

            print(f"[Sample-level DP] mechanism={dp_config.mechanism}, epsilon_total={eps_total}, "
                  f"epsilon_round={dp_config.round_epsilon}, rounds={R}")

    print("Starting POST-MIX NEIGHBOR AVERAGING Serverless Federated Learning")
    print("Architecture: Full participation + local DP training -> POST-MIX neighbor averaging")
    print("Flow: Previous aggregated weights -> Local fit with DP -> Neighbor averaging -> Evaluation")
    print(f"Dataset: {dataset_name}, Clients: {num_clients}, Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}, Batch size: {batch_size}")
    print(f"DP Config: use_dp={dp_config.enable_dp}, mechanism={dp_config.mechanism}")

    # Topology configuration
    print(f"\n TOPOLOGY CONFIGURATION:")
    print(f"  Type: {serverless_topology}")
    if serverless_topology == "ws":
        print(f"  WS k: {ws_k}")
        print(f"  WS p: {ws_p}")
    print(f"  Seed: {seed}")
    print(f"  Gossip steps per round (Q): 1")
    print(f"  Fixed topology: True (initialized once, never changes)")

    # 1. Create shared directory and clients (same as before)
    shared_folder = LocalFolder("./_unified_serverless_share")
    clients = [
        UnifiedServerlessClient(
            client_id=cid,
            dataset_name=dataset_name,
            num_clients=num_clients,
            dp_config=dp_config,
            shared_folder=shared_folder,
            seed=seed,
            batch_size=batch_size,
            local_epochs=local_epochs,
            experiment_config=experiment_config,
            use_dynamic_neighbors=True,
            non_iid=non_iid,  # Pass Non-IID configuration
            alpha=alpha,  # Pass Dirichlet alpha parameter
        )
        for cid in range(num_clients)
    ]

    # Reuse the server path's initialization for controlled comparisons.
    # Get or create initial weights with the same seed and mechanism as server
    from client import DATASET_NAME

    # Use the same seed as server (from seed_manager) and same weights file
    # This ensures serverless and server experiments start from identical initial weights
    init_weights_seed = seed_mgr.get_init_weights_seed()
    flush_print(f"[Serverless] Loading initial weights using get_or_create_initial_weights (seed={init_weights_seed})")
    global_init = get_or_create_initial_weights(
        model_class=MLP,
        num_classes=10,
        seed=init_weights_seed,
        weights_file="init_weights_mnist_mlp.npz"  # Same file as server
    )
    flush_print(f"[Serverless] Initial weights loaded/created with {len(global_init)} layers")

    shared_folder.put(key="global_round_0", nds=global_init, meta={"round": 0})

    # In decentralized mode, this initial weight also needs to be saved as each client's round -1 model
    # The persistence helper expects a model instance.
    temp_model = load_model(DATASET_NAME).to(DEVICE)
    set_model_parameters(temp_model, global_init)
    for client in clients:
        client.save_model_to_shared_dir(temp_model, -1)  # Save as round -1 for round 0 to load

    # 3. Evaluation will be done using client's local validation splits
    # No centralized global test dataset needed - each client evaluates on their own local val set

    # 4. Initialize fixed topology (ONLY ONCE before training starts)
    flush_print(f"\n{'='*60}")
    flush_print(" INITIALIZING FIXED TOPOLOGY")
    flush_print(f"{'='*60}")

    topology_averager = ServerlessNeighborAverager(
        num_clients=num_clients,
        topo_kind=serverless_topology,
        k=ws_k,
        p=ws_p,
        seed=seed
    )

    # Initialize topology (builds graph and caches neighbors)
    topology_averager.initialize_topology()

    # Print sanity check: first 3 nodes' neighbors
    flush_print(f"\n Topology Sanity Check (first 3 nodes):")
    for node_id in range(min(3, num_clients)):
        neighbors = topology_averager.neighbors.get(node_id, [])
        flush_print(f"  Node {node_id} -> Neighbors: {neighbors}")

    flush_print(f"\n Topology initialized and cached in memory")
    flush_print(f"{'='*60}\n")

    history = []
    # 5. Execute federated learning rounds - new synchronous neighbor averaging + local training process
    import random

    # Initialization: maintain all client parameters from the previous round
    current_params_by_client = {}
    for cid in range(num_clients):
        current_params_by_client[cid] = [param.copy() for param in global_init]  # Deep copy initial parameters

    # Use topology from topology_averager (already initialized)
    neighbor_graph = topology_averager.neighbors  # Get cached neighbor dict

    flush_print(f"[Coordinator] Using fixed topology: {serverless_topology}")
    if serverless_topology == "ring":
        flush_print(f"[Coordinator] Ring topology (k=2): each node connects to previous and next")
    else:
        flush_print(f"[Coordinator] Watts-Strogatz topology (k={ws_k}, p={ws_p}): random rewiring")
    flush_print(f"[Coordinator] Neighbor graph (first 3 nodes): {dict(list(neighbor_graph.items())[:3])}")

    for round_idx in range(num_rounds):
        flush_print(f"\n{'='*50}\nRound {round_idx + 1}/{num_rounds}\n{'='*50}")

        # === Full participation: all clients participate in each training round ===
        selected_client_ids = list(range(num_clients))   # Full participation

        flush_print(f"[Coordinator] Round {round_idx + 1}: Full participation - all {num_clients} clients")

        # === A) Local training: use "previous round aggregated weights" as starting point ===
        local_noised_params_by_client = {}
        failed_clients = []
        client_sample_counts = {}  # Store sample counts for each client

        flush_print(f"[Coordinator] Phase A: Local training with DP (using previous round aggregated weights)...")
        for cid in range(num_clients):
            try:
                client = clients[cid]
                init_params = current_params_by_client[cid]

                updated_local, num_examples = client.fit_one_round_with_init_params(
                    round_idx=round_idx,
                    init_params=init_params,
                    num_clients_in_round=num_clients
                )
                local_noised_params_by_client[cid] = updated_local
                client_sample_counts[cid] = num_examples

                if (cid + 1) % 20 == 0 or cid == num_clients - 1:
                    flush_print(f"[PhaseA] Completed local training {cid + 1}/{num_clients} clients")
            except Exception as e:
                flush_print(f"[PhaseA] Warning: Client {cid} training failed: {e}. Using previous weights.")
                failed_clients.append(cid)
                # Fall back to the previous round's weights.
                local_noised_params_by_client[cid] = current_params_by_client[cid]
                # Use default sample count for failed clients (could also use cached value)
                client_sample_counts[cid] = client.flower_client.train_dataset_len

        if failed_clients:
            if not (experiment_config or {}).get("allow_client_failures", False):
                raise RuntimeError(
                    f"Round {round_idx + 1} failed because {len(failed_clients)} client(s) failed"
                )
            flush_print(f"[Coordinator] Round {round_idx + 1}: {len(failed_clients)} clients failed, continuing with previous weights")

        # Store fit metrics for the final summary.
        successful_clients = num_clients - len(failed_clients)

        # Store aggregate metrics because training is decentralized.
        # Calculate client sample rate (in serverless mode, typically 100% participation)
        client_sample_rate = num_clients / num_clients  # Always 1.0 for full participation

        fit_history_entry = {
            "round": round_idx + 1,
            "successful_clients": successful_clients,
            "failed_clients": len(failed_clients),
            "total_clients": num_clients,
            "client_sample_rate": client_sample_rate,  # Add client sample rate
            "training_method": "decentralized_local_training"
        }

        # Include privacy metrics when DP is enabled.
        if dp_config.enable_dp:
            fit_history_entry.update({
                "dp_enabled": True,
                "dp_mode": dp_config.mode,
                "dp_mechanism": dp_config.mechanism,
                "round_epsilon": getattr(dp_config, 'round_epsilon', None),
                "total_epsilon_target": getattr(dp_config, 'total_epsilon', None)
            })

            # === Client-level Gaussian RDP Accounting ===
            if dp_config.mode == "client" and dp_config.mechanism == "gaussian":
                q = client_sample_rate
                sigma = getattr(dp_config, 'noise_multiplier', None)
                delta = getattr(dp_config, 'delta', 1e-5)

                if sigma is not None:
                    client_rdp.add_round(q=float(q), sigma=float(sigma), steps=1)
                    rdp_epsilon = client_rdp.get_epsilon(delta=float(delta))

                    flush_print(f"[Serverless Round {round_idx + 1}] Client Gaussian RDP: q={q:.4f}, sigma={sigma}, epsilon_cumulative={rdp_epsilon:.6f}")

                    # Add RDP info to fit_history_entry
                    fit_history_entry.update({
                        "noise_multiplier": float(sigma),
                        "client_gaussian_epsilon_cumulative": rdp_epsilon
                    })
        else:
            fit_history_entry["dp_enabled"] = False

        serverless_metrics_history["fit_metrics"].append(fit_history_entry)

        # === B) Synchronous neighbor aggregation: use topology_averager.single_gossip (Q=1) ===
        flush_print(f"[Coordinator] Phase B: Synchronous neighbor aggregation (POST-MIX, Q=1)...")

        # Prepare client_params in order (list of lists of ndarrays)
        client_params_ordered = [local_noised_params_by_client[cid] for cid in range(num_clients)]

        # Prepare sample counts in order (list of ints)
        sample_counts_ordered = [client_sample_counts[cid] for cid in range(num_clients)]

        # Call single_gossip ONCE per round (Q=1) with weighted averaging by sample counts
        aggregated_params_list = topology_averager.single_gossip(client_params_ordered, sample_counts_ordered)

        # Convert back to dict
        aggregated_params_by_client = {cid: aggregated_params_list[cid] for cid in range(num_clients)}

        if VERBOSE:
            flush_print(f"[PhaseB] Q=1 gossip completed using {serverless_topology} topology")
            # Show first 3 clients' neighbor info for verification
            for cid in range(min(3, num_clients)):
                neighbors = neighbor_graph[cid]
                flush_print(f"  Client {cid} averaged with neighbors: {neighbors}")

        flush_print(f"[Coordinator] Round {round_idx + 1} completed - POST-MIX aggregation finished")


        # === C) Evaluation: use "aggregated POST-MIX weights" for evaluation ===
        # Evaluate every post-mix model on the same full test set.
        flush_print(f"[Evaluator] Distributing POST-MIX models to all {num_clients} clients for evaluation...")

        client_eval_results = []  # Store (num_examples, accuracy, loss) from each client

        for client_id in range(num_clients):
            try:
                client = clients[client_id]
                client_params = aggregated_params_by_client[client_id]

                # Load client's model and set aggregated parameters
                eval_model = load_model(DATASET_NAME).to(DEVICE)
                set_model_parameters(eval_model, client_params)
                eval_model.eval()

                # Use client's FULL TEST SET (not partitioned, same for all clients)
                # Note: UnifiedServerlessClient wraps FlowerClient, access via flower_client
                # All clients share the same full test set, so weighted average = simple average
                correct, total, total_loss = 0, 0, 0.0
                with torch.no_grad():
                    for data, target in client.flower_client.test_loader:  # Full test set (not partitioned)
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = eval_model(data)
                        total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)

                client_accuracy = correct / total if total > 0 else 0.0
                client_loss = total_loss / total if total > 0 else 0.0

                # Store (num_examples, accuracy, loss) for weighted averaging
                # Note: Since all clients use the full test set, total is the same for all clients
                client_eval_results.append((total, client_accuracy, client_loss))

            except Exception as e:
                flush_print(f"Error evaluating client {client_id}: {e}")

        # Weight client metrics by sample count.
        # Since all clients use the full test set (same num_examples), weighted avg = simple avg
        if client_eval_results:
            total_examples = sum(num_ex for num_ex, _, _ in client_eval_results)

            if total_examples > 0:
                # Weighted average accuracy
                avg_accuracy = sum(num_ex * acc for num_ex, acc, _ in client_eval_results) / total_examples
                # Weighted average loss
                avg_loss = sum(num_ex * loss for num_ex, _, loss in client_eval_results) / total_examples
            else:
                avg_accuracy = 0.0
                avg_loss = 0.0
        else:
            avg_accuracy = 0.0
            avg_loss = 0.0

        history.append({
            "round": round_idx + 1,
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "accuracy_active": avg_accuracy,
            "loss_active": avg_loss,
        })

        flush_print(f"\n[Round {round_idx + 1} Summary] (POST-MIX Evaluation - Federated Evaluation)")
        flush_print(f"  Weighted Average Accuracy (All {len(client_eval_results)}/{num_clients} clients): {avg_accuracy:.4f}")
        flush_print(f"  Weighted Average Loss (All {len(client_eval_results)}/{num_clients} clients): {avg_loss:.4f}")
        unique_validation_samples = client_eval_results[0][0] if client_eval_results else 0
        flush_print(f"  Validation samples per client: {unique_validation_samples}")
        flush_print(f"  Total evaluation records: {total_examples if client_eval_results else 0}")
        flush_print(f"  Note: All clients use full test set, so weighted avg = simple avg")

        # Store evaluation metrics for the final summary.
        evaluate_history_entry = {
            "round": round_idx + 1,
            "loss": avg_loss,
            "accuracy": avg_accuracy * 100,
            "num_clients": len(client_eval_results),
            "evaluated_clients": num_clients,
            "total_validation_samples": total_examples if client_eval_results else 0
        }

        serverless_metrics_history["evaluate_metrics"].append(evaluate_history_entry)

        # === D) Enter next round: use "aggregated POST-MIX weights" as initial weights ===
        current_params_by_client = aggregated_params_by_client

    flush_print("\n POST-MIX NEIGHBOR AVERAGING Serverless FL completed!")
    flush_print("Architecture: Full participation + local DP training -> POST-MIX neighbor averaging")
    flush_print("Evaluation: All metrics based on POST-MIX aggregated weights")

    # Save topology to file for reproducibility
    import os
    os.makedirs(results_dir, exist_ok=True)

    topology_file = topology_averager.save_topology_to_file(results_dir)
    flush_print(f"\n Topology saved to: {topology_file}")
    flush_print(f"   Use this file to reproduce experiments with identical neighbor structure")

    final_accuracy = history[-1]["accuracy"] if history else 0.0
    final_loss = history[-1]["loss"] if history else 0.0

    if history:
        print(f"\n[SUMMARY]")
        print(f"Run finished {num_rounds} round(s)")

        print(f"\tHistory (loss, distributed):")
        for h in history:
            print(f"\t\tround {h['round']}: {h['loss']}")

        flush_print(f"\tHistory (accuracy, distributed):")
        for h in history:
            flush_print(f"\t\tround {h['round']}: {h['accuracy']}")

        flush_print(f"\tHistory (accuracy, active):")
        for h in history:
            flush_print(f"\t\tround {h['round']}: {h['accuracy_active']}")

        flush_print(f"\tHistory (metrics, distributed, fit):")
        print(f"\t{{'fit_accuracy_avg': [", end="")
        for i, h in enumerate(history):
            if i == 0:
                print(f"({h['round']}, {h['accuracy']*100:.3f})", end="")
            else:
                print(f",\n\t                      ({h['round']}, {h['accuracy']*100:.3f})", end="")
        print("],")

        print(f"\t 'fit_accuracy_active': [", end="")
        for i, h in enumerate(history):
            if i == 0:
                print(f"({h['round']}, {h['accuracy_active']*100:.3f})", end="")
            else:
                print(f",\n\t                        ({h['round']}, {h['accuracy_active']*100:.3f})", end="")
        print("],")

        print(f"\t 'fit_loss_avg': [", end="")
        for i, h in enumerate(history):
            if i == 0:
                print(f"({h['round']}, {h['loss']:.6f})", end="")
            else:
                print(f",\n\t                 ({h['round']}, {h['loss']:.6f})", end="")
        print("],")

        print(f"\t 'fit_loss_active': [", end="")
        for i, h in enumerate(history):
            if i == 0:
                print(f"({h['round']}, {h['loss_active']:.6f})", end="")
            else:
                print(f",\n\t                   ({h['round']}, {h['loss_active']:.6f})", end="")
        print("]}")

    flush_print(f"Final Global Model Evaluation:")
    flush_print(f"   Average Accuracy (All clients): {final_accuracy:.4f}")
    flush_print(f"   Average Loss (All clients): {final_loss:.4f}")
    if history:
        final_accuracy_active = history[-1]["accuracy_active"]
        final_loss_active = history[-1]["loss_active"]
        flush_print(f"   Average Accuracy (Active clients): {final_accuracy_active:.4f}")
        flush_print(f"   Average Loss (Active clients): {final_loss_active:.4f}")

    # === Privacy Budget Summary ===
    if dp_config.enable_dp and clients:
        flush_print(f"\nFinal Privacy Budget Summary:")
        try:
            # Sample client for privacy accountant access
            sample_client = clients[0]
            if hasattr(sample_client, 'privacy_accountant') and sample_client.privacy_accountant:
                total_eps, total_delta = sample_client.privacy_accountant.get_privacy_spent()
                if dp_config.mode == "client" and dp_config.mechanism == "gaussian":
                    total_delta = getattr(dp_config, "delta", 1e-5)
                    total_eps = client_rdp.get_epsilon(delta=total_delta)

                flush_print(f"   Total Privacy Consumption:")
                flush_print(f"     epsilon (epsilon): {total_eps:.6f}")
                if dp_config.mechanism == "gaussian" and total_delta is not None:
                    flush_print(f"     delta (delta): {total_delta:.2e}")

                # Show target vs actual comparison
                target_epsilon = getattr(dp_config, 'total_epsilon', None)
                if target_epsilon is not None:
                    ratio = total_eps / target_epsilon if target_epsilon > 0 else float('inf')
                    flush_print(f"   Target vs Actual:")
                    flush_print(f"     Target epsilon: {target_epsilon:.6f}")
                    flush_print(f"     Consumption ratio: {ratio:.2%}")
                    if ratio > 1.05:  # 5% tolerance
                        flush_print(f"       Privacy budget exceeded!")
                    elif ratio < 0.95:
                        flush_print(f"      Privacy budget under-utilized")
                    else:
                        flush_print(f"      Privacy budget well-utilized")

                # Detailed mechanism-specific parameters (matching server output style)
                flush_print(f"   DP Configuration:")
                flush_print(f"     Mechanism: {dp_config.mechanism}")
                flush_print(f"     Clipping threshold (C): {dp_config.C:.6f}")
                flush_print(f"     Rounds completed: {num_rounds}")
                flush_print(f"     Total clients: {num_clients}")

                # Per-round budget information
                round_epsilon = getattr(dp_config, 'round_epsilon', None)
                if round_epsilon is not None:
                    flush_print(f"     Target epsilon per round: {round_epsilon:.6f}")
                    flush_print(f"     Avg epsilon per round (actual): {total_eps / num_rounds:.6f}")

                # Mechanism-specific parameters
                if dp_config.mechanism == "gaussian":
                    if hasattr(dp_config, 'noise_multiplier') and dp_config.noise_multiplier:
                        flush_print(f"     Noise multiplier: {dp_config.noise_multiplier:.4f}")
                    if hasattr(dp_config, 'delta') and dp_config.delta:
                        flush_print(f"     Target delta: {dp_config.delta:.2e}")

                    # Sample rate and batch information
                    sample_rate = getattr(dp_config, 'q', None) or getattr(dp_config, 'sample_rate', None)
                    if sample_rate:
                        flush_print(f"     Sample rate (q): {sample_rate:.6f}")

                elif dp_config.mechanism == "laplace":
                    # Laplace scale parameter
                    if round_epsilon:
                        laplace_scale = dp_config.C / round_epsilon
                        flush_print(f"     Laplace scale (b): {laplace_scale:.6f}")
                        flush_print(f"     Epsilon per step: {round_epsilon:.6f}")

                # Client-level statistics (if available)
                if hasattr(sample_client.privacy_accountant, 'get_client_privacy_spent'):
                    try:
                        client_0_eps, client_0_delta = sample_client.privacy_accountant.get_client_privacy_spent(0)
                        flush_print(f"   Client-level Summary (Sample - Client 0):")
                        flush_print(f"     Client 0 epsilon consumption: {client_0_eps:.6f}")
                        if client_0_delta is not None:
                            flush_print(f"     Client 0 delta consumption: {client_0_delta:.2e}")
                    except Exception:
                        pass  # Skip if client-level stats unavailable

            else:
                flush_print(f"   Privacy accountant not available for detailed summary")
                flush_print(f"   DP Configuration:")
                flush_print(f"     Mechanism: {dp_config.mechanism}")
                flush_print(f"     Clipping threshold (C): {dp_config.C:.6f}")
                if hasattr(dp_config, 'total_epsilon'):
                    flush_print(f"     Target epsilon: {dp_config.total_epsilon:.6f}")
                if hasattr(dp_config, 'noise_multiplier') and dp_config.noise_multiplier:
                    flush_print(f"     Noise multiplier: {dp_config.noise_multiplier:.4f}")

        except Exception as e:
            flush_print(f"   Error computing privacy summary: {e}")
            flush_print(f"   DP was enabled with mechanism: {dp_config.mechanism}")
    else:
        if dp_config.enable_dp:
            flush_print(f"\nPrivacy Configuration:")
            flush_print(f"   DP enabled but no client privacy accountant available")
            flush_print(f"   Mechanism: {dp_config.mechanism}")
            flush_print(f"   Clipping threshold (C): {dp_config.C:.6f}")

    result = {
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "final_accuracy_active": history[-1]["accuracy_active"] if history else 0.0,
        "final_loss_active": history[-1]["loss_active"] if history else 0.0,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "history": history,
    }

    # Add privacy budget information to result (matching server framework output)
    if dp_config.enable_dp and clients:
        try:
            sample_client = clients[0]
            if hasattr(sample_client, 'privacy_accountant') and sample_client.privacy_accountant:
                total_eps, total_delta = sample_client.privacy_accountant.get_privacy_spent()
                if dp_config.mode == "client" and dp_config.mechanism == "gaussian":
                    total_delta = getattr(dp_config, "delta", 1e-5)
                    total_eps = client_rdp.get_epsilon(delta=total_delta)

                # Core privacy metrics
                privacy_result = {
                    "privacy_epsilon": float(total_eps),
                    "privacy_delta": float(total_delta) if total_delta is not None else None,
                    "cumulative_epsilon": float(total_eps),  # Alias for compatibility
                    "cumulative_delta": float(total_delta) if total_delta is not None else None,
                    "privacy_breached": total_eps > getattr(dp_config, 'total_epsilon', float('inf')) if hasattr(dp_config, 'total_epsilon') else False,
                }

                # DP configuration parameters
                privacy_result.update({
                    "dp_mechanism": dp_config.mechanism,
                    "noise_mechanism": dp_config.mechanism,  # Alias for compatibility
                    "clipping_threshold": float(dp_config.C),
                    "max_grad_norm": float(dp_config.C),  # Alias for compatibility
                    "target_epsilon": getattr(dp_config, 'total_epsilon', None),
                    "total_rounds": num_rounds,
                    "total_clients": num_clients,
                })

                # Per-round information
                round_epsilon = getattr(dp_config, 'round_epsilon', None)
                if round_epsilon is not None:
                    privacy_result.update({
                        "round_target_epsilon": float(round_epsilon),
                        "epsilon_per_round": float(round_epsilon),  # Alias
                        "avg_epsilon_per_round_actual": float(total_eps / num_rounds) if num_rounds > 0 else 0.0,
                    })

                # Mechanism-specific parameters
                if dp_config.mechanism == "gaussian":
                    if hasattr(dp_config, 'noise_multiplier') and dp_config.noise_multiplier:
                        privacy_result["noise_multiplier"] = float(dp_config.noise_multiplier)
                        privacy_result["actual_noise_multiplier"] = float(dp_config.noise_multiplier)  # Alias

                    if hasattr(dp_config, 'delta') and dp_config.delta:
                        privacy_result["target_delta"] = float(dp_config.delta)

                    # Sample rate information
                    sample_rate = getattr(dp_config, 'q', None) or getattr(dp_config, 'sample_rate', None)
                    if sample_rate:
                        privacy_result["sample_rate"] = float(sample_rate)

                elif dp_config.mechanism == "laplace":
                    if round_epsilon:
                        laplace_scale = dp_config.C / round_epsilon
                        privacy_result.update({
                            "laplace_scale": float(laplace_scale),
                            "scale_parameter": float(laplace_scale),  # Alias
                            "epsilon_per_step": float(round_epsilon),
                        })

                # Client-level statistics (if available)
                if hasattr(sample_client.privacy_accountant, 'get_client_privacy_spent'):
                    try:
                        client_0_eps, client_0_delta = sample_client.privacy_accountant.get_client_privacy_spent(0)
                        privacy_result.update({
                            "client_0_epsilon": float(client_0_eps),
                            "client_0_delta": float(client_0_delta) if client_0_delta is not None else None,
                        })
                    except Exception:
                        pass  # Skip if client-level stats unavailable

                result.update(privacy_result)

        except Exception as e:
            result.update({
                "privacy_error": str(e),
                "dp_mechanism": dp_config.mechanism,
                "dp_enabled": True,
            })
    else:
        # Add basic DP info even when disabled
        result.update({
            "dp_enabled": dp_config.enable_dp,
            "dp_mechanism": dp_config.mechanism if dp_config.enable_dp else None,
        })

    # Print the final experiment summary.
    # Get final RDP epsilon if applicable
    final_rdp_epsilon = None
    if dp_config.enable_dp and dp_config.mode == "client" and dp_config.mechanism == "gaussian":
        final_rdp_epsilon = client_rdp.get_epsilon(delta=getattr(dp_config, 'delta', 1e-5))

    print_serverless_experiment_summary(final_rdp_epsilon=final_rdp_epsilon)

    if use_ray_parallel and RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
        flush_print("Ray shutdown completed")

    return result


if __name__ == "__main__":
    baseline_config = UnifiedDPConfig(enable_dp=False)

    flush_print("Testing Baseline (No DP)...")
    run_unified_serverless_fl(
        num_rounds=2,
        dp_config=baseline_config,
        seed=42
    )
