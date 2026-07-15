"""
Federated Learning Client with Differential Privacy (Using LocalDpMod Adapter)
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.client import NumPyClient, ClientApp, Client
from flwr.common import Context, FitIns, FitRes, EvaluateIns, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from typing import Dict, List, Tuple
import logging
import gc
import traceback
# Removed: from opacus.accountants import RDPAccountant
# All Opacus imports are now lazy-loaded in opacus_client_dp.py to avoid module-lock deadlock

from utils import (
    SimpleCNN,
    MLP,
    load_datasets,
    get_model_parameters,
    set_model_parameters,
    train,
    test,
)
from opacus_client_dp import OpacusClientDP
from seed_manager import get_seed_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# Weight handling utility functions - ensure that the server and serverless versions use the same weight layout
# ======================================================================

def flatten_weights(state_dict):
    """
    Flatten the weight tensors into a 1D vector (maintaining the same weight layout between Server and Serverless)

    Args:
        state_dict: PyTorch model's state_dict

    Returns:
        tuple: (flat_tensor, shapes, keys)
            - flat_tensor: Flattened 1D tensor
            - shapes: List of original shapes of each layer's weights
            - keys: List of keys for the weight parameters
    """
    flats, shapes, keys = [], [], []
    for k, v in state_dict.items():
        flats.append(v.view(-1))
        shapes.append(v.shape)
        keys.append(k)
    return torch.cat(flats), shapes, keys


def unflatten_to_state_dict(flat, shapes, keys, ref_state_dict):
    """
    Restore a flattened vector back to state_dict format

    Args:
        flat: Flattened 1D tensor
        shapes: List of original shapes of each layer's weights
        keys: List of keys for the weight parameters
        ref_state_dict: Reference state_dict (used to determine dtype and device)

    Returns:
        dict: Restored state_dict
    """
    out = {}
    offset = 0
    for k, sh in zip(keys, shapes):
        n = int(torch.prod(torch.tensor(sh)))
        out[k] = flat[offset:offset+n].view(sh).to(ref_state_dict[k].dtype)
        offset += n
    return out


@torch.no_grad()
def compute_client_update(model, trainloader, device, epochs, optimizer, criterion):
    """
    Perform one round of local training to generate client updates (weight deltas) and the original gradient norms

    Args:
        model: PyTorch model
        trainloader: Training data loader
        device: Computing device
        epochs: Number of local training epochs
        optimizer: Optimizer
        criterion: Loss function

    Returns:
        tuple: (delta, norm_raw, metadata)
            - delta: Weight delta dictionary {key: tensor_diff}
            - norm_raw: Original update's L2 norm
            - metadata: (shapes, keys, theta_before) for further processing
    """
    theta_before = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    theta_after = model.state_dict()
    delta = {}
    for k in theta_after:
        delta[k] = theta_after[k] - theta_before[k]

    flat_delta, shapes, keys = flatten_weights(delta)
    norm_raw = torch.linalg.vector_norm(flat_delta, ord=2).item()

    return delta, norm_raw, (shapes, keys, theta_before)


def clip_and_noise_delta(delta, C, mechanism, noise_multiplier=None, round_epsilon=None, delta_param=1e-5, rng=None):
    """
    Client-level clipping and noise addition (Gaussian / L2-Laplace), using unified L2 clipping

    Args:
        delta: Weight delta dictionary
        C: Clipping threshold (using unified L2 clipping)
        mechanism: Noise mechanism "gaussian" | "laplace" | "none"
        noise_multiplier: Gaussian noise multiplier (sigma = C * noise_multiplier)
        round_epsilon: Per-round privacy budget
        delta_param: Privacy parameter delta (for Gaussian mechanism)
        rng: Random number generator (optional)

    Returns:
        tuple: (noisy_delta, norm_clipped, mechanism_used, extra_info)
            - noisy_delta: Dictionary of weight updates after noise addition
            - norm_clipped: Clipped norm
            - mechanism_used: Actual mechanism used
            - extra_info: Additional information dictionary (e.g., sigma, b, etc.)
    """
    import math
    device = next(iter(delta.values())).device

    flat, shapes, keys = flatten_weights(delta)

    norm = torch.linalg.vector_norm(flat, ord=2)

    scale = min(1.0, C / (norm + 1e-12))
    flat = flat * scale
    norm_clipped = (norm * scale).item()

    if mechanism == "gaussian":
        assert noise_multiplier is not None or round_epsilon is not None
        if noise_multiplier is not None:
            sigma = C * noise_multiplier
        else:
            sigma = C * math.sqrt(2 * math.log(1.25 / max(delta_param, 1e-12))) / (round_epsilon + 1e-12)

        noise = torch.randn_like(flat) * sigma
        flat_noisy = flat + noise
        mech = "gaussian"
        extra = {"sigma": float(sigma)}

    elif mechanism == "laplace":
        assert round_epsilon is not None

        b = C / (round_epsilon + 1e-12)
        noise = torch.distributions.Laplace(loc=0.0, scale=b).sample(flat.shape).to(device)
        flat_noisy = flat + noise

        mech = "laplace"
        extra = {"b": float(b)}

    else:
        flat_noisy = flat
        mech = "none"
        extra = {}

    noisy_delta = unflatten_to_state_dict(flat_noisy, shapes, keys, delta)
    return noisy_delta, norm_clipped, mech, extra


# ======================================================================
# TensorFlow version of the weight handling utility functions (for the Serverless version)
# ======================================================================

def tf_flatten_weights(weights_list):
    """
    TensorFlow version: Flatten weight list into 1D array

    Args:
        weights_list: TensorFlow Weight list (list of numpy arrays)

    Returns:
        tuple: (flat_array, shapes)
            - flat_array: Flattened 1D numpy array
            - shapes: Original shapes of each layer's weights
    """
    import numpy as np
    flats, shapes = [], []
    for w in weights_list:
        shapes.append(w.shape)
        flats.append(w.reshape(-1))
    flat = np.concatenate(flats, axis=0).astype(np.float32, copy=False)
    return flat, shapes


def tf_unflatten_weights(flat, shapes):
    """
    TensorFlow version: Unflatten array back to weight list

    Args:
        flat: Flattened 1D array
        shapes: Original shapes of each layer's weights

    Returns:
        list: Unflattened weight list (list of numpy arrays)
    """
    import numpy as np
    out, off = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(flat[off:off+n].reshape(sh))
        off += n
    return out


def tf_sample_coordinate_l1_laplace(dim, b, rng):
    """
    TensorFlow version: Sample coordinate-wise L1-Laplace noise

    Args:
        dim: Dimension
        b: Laplace parameter
        rng: numpy random number generator

    Returns:
        numpy.ndarray: Coordinate-wise L1-Laplace noise vector
    """
    import numpy as np
    return rng.laplace(loc=0.0, scale=b, size=(dim,)).astype(np.float32)


def tf_sample_spherical_l2_gaussian(dim, sigma, rng):
    """
    TensorFlow version: Sample spherical L2-Gaussian noise

    Args:
        dim: Dimension
        sigma: Gaussian standard deviation
        rng: numpy random number generator

    Returns:
        numpy.ndarray: Spherical L2-Gaussian noise vector
    """
    import numpy as np
    return rng.normal(loc=0.0, scale=float(sigma), size=(dim,)).astype(np.float32)


def tf_compute_sigma_from_epsilon_delta(epsilon_round, delta, C=1.0):
    """
    TensorFlow version: Compute sigma for Gaussian noise from (epsilon,delta)

    Note: The returned coefficient does not include C; it should be multiplied by C at the call site (see tf_clip_and_noise_weights_update)

    Args:
        epsilon_round: Per-round privacy budget epsilon
        delta: Privacy parameter delta
        C: Clipping threshold (not used here, kept for interface compatibility)

    Returns:
        float: Noise coefficient (needs to be multiplied by C to get actual sigma)
    """
    import math
    if epsilon_round <= 0:
        raise ValueError("epsilon_round must be > 0")
    return math.sqrt(2 * math.log(1.25 / max(delta, 1e-12))) / epsilon_round


def tf_clip_and_noise_weights_update(weights_before, weights_after, C, mechanism,
                                   noise_multiplier=None, round_epsilon=None,
                                   delta=1e-5, rng=None):
    """
    TensorFlow version: Compute weight update with clipping and noise

    Args:
        weights_before: List of weights before training
        weights_after: List of weights after training
        C: Clipping threshold
        mechanism: Noise mechanism "gaussian" | "laplace" | "none"
        noise_multiplier: Gaussian noise multiplier
        round_epsilon: Per-round privacy budget
        delta: Privacy parameter delta (only for Gaussian)
        rng: numpy random number generator

    Returns:
        tuple: (noisy_weights, norm_raw, norm_clipped, mechanism_used, extra_info)
            - noisy_weights: Noisy weight list
            - norm_raw: Original update norm
            - norm_clipped: Clipped norm
            - mechanism_used: Actual mechanism used
            - extra_info: Extra information dictionary
    """
    import numpy as np
    import math

    if rng is None:
        rng = np.random.default_rng()

    delta_weights = [after - before for after, before in zip(weights_after, weights_before)]

    flat_delta, shapes = tf_flatten_weights(delta_weights)
    norm_raw = float(np.linalg.norm(flat_delta))

    # Unified L2 clipping for both mechanisms (consistent with the rest of codebase)
    norm_for_clipping = norm_raw

    scale = min(1.0, C / (norm_for_clipping + 1e-12))
    flat_delta_clipped = flat_delta * scale
    norm_clipped = norm_for_clipping * scale

    if mechanism == "gaussian":
        if noise_multiplier is not None:
            sigma = C * noise_multiplier
        elif round_epsilon is not None:
            coefficient = tf_compute_sigma_from_epsilon_delta(round_epsilon, delta)
            sigma = C * coefficient
        else:
            raise ValueError("Either noise_multiplier or round_epsilon must be provided for Gaussian mechanism")

        noise = tf_sample_spherical_l2_gaussian(len(flat_delta_clipped), sigma, rng)
        flat_noisy = flat_delta_clipped + noise
        mech = "gaussian"
        extra = {"sigma": float(sigma)}

    elif mechanism == "laplace":
        if round_epsilon is None:
            raise ValueError("round_epsilon must be provided for Laplace mechanism")

        b = C / (round_epsilon + 1e-12)
        noise = tf_sample_coordinate_l1_laplace(len(flat_delta_clipped), b, rng)
        flat_noisy = flat_delta_clipped + noise
        mech = "laplace"
        extra = {"b": float(b)}

    else:
        flat_noisy = flat_delta_clipped
        mech = "none"
        extra = {}

    delta_noisy_list = tf_unflatten_weights(flat_noisy, shapes)
    noisy_weights = [before + delta_noisy for before, delta_noisy in zip(weights_before, delta_noisy_list)]

    return noisy_weights, norm_raw, norm_clipped, mech, extra


def diagnose_gpu_setup(partition_id: int = None):
    """
    Diagnose GPU visibility and common CUDA configuration problems.

    Args:
        partition_id: Client identifier used in log messages.

    Returns:
        dict: GPU diagnostic information.
    """
    import sys

    prefix = f"[Client {partition_id}] " if partition_id is not None else ""

    # Check CUDA support reported by PyTorch.
    is_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "not set")

    logger.info(f"{prefix}[CUDA] is_available={is_available}, count={device_count}, visible={cuda_visible}")
    logger.info(f"{prefix}[PYTORCH] version={torch.__version__}, cuda_version={torch.version.cuda}")

    # Record the active Python environment.
    logger.info(f"{prefix}[PYTHON] executable={sys.executable}")

    # Report actionable diagnostics.
    diagnosis = {
        "is_available": is_available,
        "device_count": device_count,
        "cuda_visible": cuda_visible,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_executable": sys.executable,
        "recommended_device": "cpu",
    }

    if not is_available:
        logger.warning(f"{prefix} GPU is unavailable. Possible causes:")
        logger.warning(f"{prefix}  1. A CPU-only PyTorch build is installed (torch.version.cuda={torch.version.cuda})")
        logger.warning(f"{prefix}  2. The CUDA driver is missing or incompatible")
        logger.warning(f"{prefix}  Install a CUDA-compatible PyTorch build if a GPU was allocated")
    elif device_count == 0:
        logger.warning(f"{prefix} CUDA is enabled, but no GPU devices are visible")
        logger.warning(f"{prefix} Check CUDA_VISIBLE_DEVICES={cuda_visible}")
    else:
        # Use the currently visible CUDA device.
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"{prefix} GPU available: {device_name} (device {current_device})")
        diagnosis["recommended_device"] = f"cuda:{current_device}"
        diagnosis["device_name"] = device_name

        # Report memory when the runtime exposes it.
        if hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(current_device)
            logger.info(f"{prefix} GPU memory: {free_mem/1e9:.2f}GB free / {total_mem/1e9:.2f}GB total")
            diagnosis["gpu_memory_free_gb"] = free_mem / 1e9
            diagnosis["gpu_memory_total_gb"] = total_mem / 1e9

    return diagnosis


def get_optimal_device(partition_id: int = None, force_diagnose: bool = False):
    """
    Select the best device when the client starts.

    Args:
        partition_id: Client identifier.
        force_diagnose: Run the full diagnostic even if it has already run.

    Returns:
        torch.device: Selected training device.
    """
    # Run full diagnostics once unless explicitly requested again.
    if force_diagnose or not hasattr(get_optimal_device, '_diagnosed'):
        diagnosis = diagnose_gpu_setup(partition_id)
        get_optimal_device._diagnosed = True
        get_optimal_device._diagnosis = diagnosis
    else:
        diagnosis = getattr(get_optimal_device, '_diagnosis', {"recommended_device": "cpu"})

    recommended = diagnosis.get("recommended_device", "cpu")
    device = torch.device(recommended)

    prefix = f"[Client {partition_id}] " if partition_id is not None else ""
    logger.debug(f"{prefix}Using device: {device}")

    return device


def fix_client_seed(seed: int, partition_id: int):
    """
    Fix random seeds for client-side deterministic behavior.
    Uses base seed + partition_id to ensure different clients have different but reproducible seeds.
    """
    client_seed = seed + partition_id
    os.environ["PYTHONHASHSEED"] = str(client_seed)
    random.seed(client_seed)
    np.random.seed(client_seed)
    torch.manual_seed(client_seed)
    torch.cuda.manual_seed_all(client_seed)
    logger.info(f"[Client {partition_id}] Random seeds fixed to {client_seed} for reproducible training")

# --- Configuration ---
# Device selection is delayed until client initialization because Flower may set
# CUDA_VISIBLE_DEVICES after importing this module.
DATASET_NAME = "mnist"
NUM_CLIENTS = 100
BATCH_SIZE = 64
NUM_WORKERS = 4


def _global_l2_between_param_lists(params_after, params_before) -> float:
    """
    Compute the global L2 norm difference between two parameter lists (avoiding large concatenation and memory issues).

    Args:
        params_after: List of parameters after training
        params_before: List of parameters before training

    Returns:
        float: Global L2 norm ||params_after - params_before||_2
    """
    sqsum = 0.0
    for a, b in zip(params_after, params_before):
        d = (a.astype(np.float64) - b.astype(np.float64))
        sqsum += float((d * d).sum())
    return max(sqsum ** 0.5, 1e-12)


def load_model(dataset_name: str) -> nn.Module:
    """Load a fresh model instance based on dataset name."""
    if dataset_name.lower() == "cifar10":
        return SimpleCNN(num_classes=10)
    elif dataset_name.lower() == "mnist":
        return MLP(num_classes=10)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class FlowerClient(NumPyClient):
    """
    A Flower client that loads its own data in the constructor.
    This is the recommended approach to avoid state-related issues.
    """

    def __init__(self, partition_id: int, num_partitions: int = NUM_CLIENTS, non_iid: bool = True, alpha: float = 0.2):
        """
        Initializes the client by loading data for its partition.

        Args:
            partition_id: The partition ID for this client (0 to num_partitions-1)
            num_partitions: Total number of data partitions (defaults to NUM_CLIENTS)
            non_iid: Whether to use Non-IID data partitioning (defaults to True)
            alpha: Dirichlet concentration parameter (smaller = more heterogeneous, default 0.1)
        """
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.non_iid = non_iid
        self.alpha = alpha

        # Select the device after Flower has configured process visibility.
        self.device = get_optimal_device(partition_id=partition_id, force_diagnose=True)

        # Read the master seed through the shared seed manager.
        seed_mgr = get_seed_manager()
        master_seed = seed_mgr.get_master_seed()
        fix_client_seed(seed=master_seed, partition_id=partition_id)
        logger.info(f"[Client {partition_id}]  Initialized with MASTER_SEED={master_seed} (from seed_manager)")

        # Log Non-IID configuration
        data_distribution = f"Non-IID (DirichletPartitioner, alpha={alpha})" if self.non_iid else "IID (Uniform Random)"
        logger.info(f"[Client {partition_id}]  Data Distribution: {data_distribution}")

        logger.info(f"[Client {partition_id}] Initializing and loading data from partition {partition_id}/{num_partitions}...")
        use_pin_memory = self.device.type == 'cuda'

        # Persistent workers reduce loader startup overhead across epochs.
        num_workers = NUM_WORKERS if self.device.type == 'cuda' else 0
        persistent_workers = num_workers > 0

        try:
            train_dataset, test_dataset = load_datasets(
                name=DATASET_NAME,
                num_partitions=self.num_partitions,
                partition_id=self.partition_id,
                non_iid=self.non_iid,
                alpha=self.alpha
            )

            self.train_dataset_len = len(train_dataset)
            self.test_dataset_len = len(test_dataset)

            # Log dataset loading confirmation with Non-IID status
            logger.info(f"[Client {partition_id}]  Loaded datasets with non_iid={self.non_iid}, alpha={self.alpha}: "
                       f"train={self.train_dataset_len}, test={self.test_dataset_len}")

            # Derive independent, reproducible seeds for both loaders.
            train_seed = seed_mgr.get_train_loader_seed(partition_id)
            eval_seed = seed_mgr.get_eval_loader_seed(partition_id)

            # DataLoader generators remain on CPU even during GPU training.
            if self.device.type == 'cuda':
                train_generator = torch.Generator(device='cpu')
            else:
                train_generator = torch.Generator()
            train_generator.manual_seed(train_seed)

            if self.device.type == 'cuda':
                eval_generator = torch.Generator(device='cpu')
            else:
                eval_generator = torch.Generator()
            eval_generator.manual_seed(eval_seed)

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                drop_last=True,
                generator=train_generator,
                persistent_workers=persistent_workers,
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                generator=eval_generator,
                persistent_workers=persistent_workers,
            )

            logger.info(f"[Client {partition_id}] Initialization complete. Train: {self.train_dataset_len}, Test: {self.test_dataset_len}")
            logger.info(f"[Client {partition_id}] DataLoader: num_workers={num_workers}, pin_memory={use_pin_memory}, persistent_workers={persistent_workers}")

            # Store dataset for recreating loader with different seeds per round
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

        except Exception as e:
            logger.error(f"[Client {partition_id}] FAILED to initialize: {e}")
            logger.error(traceback.format_exc())
            raise

    def _recreate_train_loader(self, round_num: int):
        """
        Recreate the training DataLoader with a round-specific seed.

        Args:
            round_num: Zero-based training round.

        Note:
            Each round receives a different batch order while remaining reproducible.
        """
        seed_mgr = get_seed_manager()
        train_seed = seed_mgr.get_train_loader_seed(self.partition_id, round_num)

        # DataLoader requires a CPU generator.
        if self.device.type == 'cuda':
            train_generator = torch.Generator(device='cpu')
        else:
            train_generator = torch.Generator()
        train_generator.manual_seed(train_seed)

        use_pin_memory = self.device.type == 'cuda'
        num_workers = NUM_WORKERS if self.device.type == 'cuda' else 0
        persistent_workers = num_workers > 0

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
            generator=train_generator,
            persistent_workers=persistent_workers,
        )
        logger.debug(f"[Client {self.partition_id}] Train loader recreated for round {round_num} with seed {train_seed}")

    def _train_with_gradient_diagnostics(
        self, net: nn.Module, trainloader: DataLoader, epochs: int,
        device: torch.device, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Performs training and returns a flat dictionary of serializable diagnostic metrics.
        """
        import numpy as np

        net.train()
        all_grad_norms = []
        total_loss = 0
        num_batches = 0

        logger.info(f"[Client {self.partition_id}]  Starting gradient norm diagnostics...")

        for epoch in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Measure gradient norm
                total_norm = 0.0
                for param in net.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5
                all_grad_norms.append(total_norm)

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # Aggregate all metrics into a flat, serializable dictionary
        diagnostic_metrics = {}
        if all_grad_norms:
            diagnostic_metrics.update({
                'grad_median': float(np.median(all_grad_norms)),
                'grad_mean': float(np.mean(all_grad_norms)),
                'grad_std': float(np.std(all_grad_norms)),
                'grad_max': float(np.max(all_grad_norms)),
                'grad_min': float(np.min(all_grad_norms)),
                'grad_p75': float(np.percentile(all_grad_norms, 75)),
                'grad_p90': float(np.percentile(all_grad_norms, 90)),
                'grad_p95': float(np.percentile(all_grad_norms, 95)),
                'grad_p99': float(np.percentile(all_grad_norms, 99)),
            })
            logger.info(f"[Client {self.partition_id}]  GRADIENT DIAGNOSTICS SUMMARY:")
            logger.info(f"  - 95th percentile: {diagnostic_metrics['grad_p95']:.4f} (recommended max_grad_norm)")
            logger.info(f"  - Total batches: {len(all_grad_norms)}")
        else:
            # If no gradient norms were collected, add default values
            logger.warning(f"[Client {self.partition_id}] No gradient norms collected!")
            diagnostic_metrics.update({
                'grad_median': 0.0,
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'grad_max': 0.0,
                'grad_min': 0.0,
                'grad_p75': 0.0,
                'grad_p90': 0.0,
                'grad_p95': 0.0,
                'grad_p99': 0.0,
            })

        if num_batches > 0:
            diagnostic_metrics['diagnostic_avg_loss'] = float(total_loss / num_batches)
        else:
            diagnostic_metrics['diagnostic_avg_loss'] = 0.0

        return diagnostic_metrics

    def _sanitize_metrics(self, metrics: Dict) -> Dict:
        """
        Ensure all metrics values are of valid types for Flower framework.
        Flower requires: Union[int, float, str, bytes, bool, list[int], list[float], list[str], list[bytes], list[bool]]
        """
        sanitized = {}
        for key, value in metrics.items():
            if value is None:
                logger.warning(f"[Client {self.partition_id}] Skipping None value for metric '{key}'")
                continue
            elif isinstance(value, (int, float, str, bytes, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                if all(isinstance(item, (int, float, str, bytes, bool)) for item in value):
                    sanitized[key] = list(value)
                else:
                    logger.warning(f"[Client {self.partition_id}] Skipping invalid list metric '{key}' with mixed types")
            else:
                try:
                    sanitized[key] = str(value)
                    logger.warning(f"[Client {self.partition_id}] Converted metric '{key}' from {type(value)} to string")
                except Exception:
                    logger.warning(f"[Client {self.partition_id}] Skipping invalid metric '{key}' of type {type(value)}")
        return sanitized

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on the client's local data.
        """
        try:
            server_round = config.get('server_round', 0)
            logger.info(f"[Client {self.partition_id}] Starting fit for round {server_round}")

            # Rebuild the loader with the current round seed.
            self._recreate_train_loader(server_round)

            if not self.train_loader or self.train_dataset_len == 0:
                raise RuntimeError("Train data loader not initialized properly.")

            model = load_model(DATASET_NAME).to(self.device)
            set_model_parameters(model, parameters)

            theta_before = [param.copy() for param in get_model_parameters(model)]

            learning_rate = config.get("learning_rate", 0.001)
            momentum = config.get("momentum", 0.0)
            weight_decay = config.get("weight_decay", 0.0)
            optimizer_type = config.get("optimizer", "sgd").lower()  # New optimizer type configuration

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
                logger.info(f"[Client {self.partition_id}] Using Adam optimizer (lr={learning_rate}, beta1={beta1}, beta2={beta2})")
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
                logger.info(f"[Client {self.partition_id}] Using SGD optimizer (lr={learning_rate}, momentum={momentum})")

            use_dp = config.get("use_dp", False)
            dp_mode = config.get("dp_mode", "opacus")
            if use_dp and dp_mode in ["sample", "opacus"]:
                criterion = nn.CrossEntropyLoss(reduction="sum")
                logger.info(f"[Client {self.partition_id}] Using CrossEntropyLoss with reduction='sum' for sample-level DP")
            else:
                criterion = nn.CrossEntropyLoss()
                logger.info(f"[Client {self.partition_id}] Using CrossEntropyLoss with reduction='mean' (default)")

            metrics = {}

            if use_dp and dp_mode == "opacus":
                logger.info(f"[Client {self.partition_id}] Using Opacus DP training.")

                target_epsilon = config.get("target_epsilon")
                if target_epsilon is None:
                    raise ValueError("target_epsilon must be provided for Opacus DP training.")


                max_grad_norm = config.get("max_grad_norm")
                if max_grad_norm is None:
                    raise ValueError("max_grad_norm must be provided for Opacus DP training.")

                delta = config.get("delta", 1e-5)
                sample_rate = min(1.0, float(BATCH_SIZE) / max(1, self.train_dataset_len))


                local_epochs = config["local_epochs"]
                logger.info(f"[Client {self.partition_id}] Using {local_epochs} local epochs for DP calculations")


                noise_mechanism = config.get("noise_mechanism", "gaussian")
                if noise_mechanism == "laplace":
                    epsilon_per_step = config.get("epsilon_per_step")
                    if epsilon_per_step is None:
                        estimated_steps = len(self.train_loader) * local_epochs
                        epsilon_per_step = target_epsilon / estimated_steps
                        logger.info(f"[Client {self.partition_id}] Estimated epsilon_per_step: {epsilon_per_step:.6f} (based on round_target_epsilon: {target_epsilon}, {estimated_steps} steps)")

                    dp_engine = OpacusClientDP(
                        max_grad_norm=max_grad_norm,
                        sample_rate=sample_rate,
                        noise_mechanism="laplace",
                        epsilon_per_step=epsilon_per_step,
                        strict_mode=False,
                    )

                    private_model, private_optimizer, private_data_loader = dp_engine.attach(
                        model=model, optimizer=optimizer, data_loader=self.train_loader
                    )
                    logger.info(f"[Client {self.partition_id}] Using Laplace noise with epsilon_per_step: {epsilon_per_step}")
                else:
                    dp_engine = OpacusClientDP(
                        noise_multiplier=1,
                        delta=delta,
                        epochs=local_epochs,
                        sample_rate=sample_rate,
                        strict_mode=False,
                        max_grad_norm=max_grad_norm,
                        experimental_mode=True
                    )

                    private_model, private_optimizer, private_data_loader = dp_engine.attach(
                        model=model, optimizer=optimizer, data_loader=self.train_loader, target_epsilon=target_epsilon
                    )
                    logger.info(f"[Client {self.partition_id}] Using Gaussian noise with round_target_epsilon: {target_epsilon} for {local_epochs} local epochs")

                train_metrics = train(
                    net=private_model, trainloader=private_data_loader, epochs=local_epochs,
                    device=self.device, optimizer=private_optimizer, criterion=criterion
                )

                privacy_metrics = dp_engine.get_privacy_spent()
                final_loss, final_accuracy = test(private_model, self.test_loader, self.device)

                estimated_steps = len(self.train_loader) * local_epochs

                metrics.update({
                    "final_test_loss": float(final_loss), "final_test_accuracy": float(final_accuracy),
                    "privacy_epsilon": privacy_metrics["epsilon"], "privacy_delta": privacy_metrics["delta"],
                    "target_epsilon": target_epsilon,
                    "max_grad_norm": max_grad_norm, "sample_rate": sample_rate,
                    "noise_mechanism": noise_mechanism,
                    "train_steps_this_round": int(estimated_steps),
                    "dp_mode": "opacus",
                })


                if noise_mechanism == "laplace":
                    if "epsilon_per_step" in privacy_metrics:
                        metrics["epsilon_per_step"] = privacy_metrics["epsilon_per_step"]
                    if "scale_parameter" in privacy_metrics:
                        metrics["scale_parameter"] = privacy_metrics["scale_parameter"]
                    logger.info(f"[Client {self.partition_id}] Laplace: epsilon_per_step={privacy_metrics.get('epsilon_per_step', 'N/A')}, scale={privacy_metrics.get('scale_parameter', 'N/A')}")
                else:
                    if privacy_metrics.get("actual_noise_multiplier") is not None:
                        metrics["actual_noise_multiplier"] = privacy_metrics["actual_noise_multiplier"]
                        logger.info(f"[Client {self.partition_id}] Gaussian: actual_noise_multiplier={metrics['actual_noise_multiplier']:.4f}")
                    elif privacy_metrics.get("noise_multiplier") is not None:
                        metrics["noise_multiplier"] = privacy_metrics["noise_multiplier"]

                trained_parameters = get_model_parameters(private_model)

                # === Detach DP engine to allow reattachment in next round ===
                dp_engine.detach()
                logger.info(f"[Client {self.partition_id}] DP engine detached successfully")

            elif use_dp and dp_mode == "client":
                logger.info(f"[Client {self.partition_id}] Using LocalDpMod DP training (handled by client wrapper).")

                # Give the adapter a reproducible, client-specific noise seed.
                if "partition_id" not in config:
                    config["partition_id"] = self.partition_id
                if "server_round" not in config:
                    config["server_round"] = server_round

                local_epochs = config.get("local_epochs", 1)
                train(
                    net=model, trainloader=self.train_loader, epochs=local_epochs,
                    device=self.device, optimizer=optimizer, criterion=criterion
                )

                estimated_steps = len(self.train_loader) * local_epochs
                final_loss, final_accuracy = test(model, self.test_loader, self.device)

                metrics.update({
                    "final_test_loss": float(final_loss),
                    "final_test_accuracy": float(final_accuracy),
                    "train_steps_this_round": int(estimated_steps),
                    "max_grad_norm": config.get("max_grad_norm", 1.0),
                    "noise_mechanism": config.get("noise_mechanism", "gaussian"),
                    "dp_mode": "client",
                })

                if "noise_multiplier" in config:
                    metrics["noise_multiplier"] = config["noise_multiplier"]
                if "global_target_epsilon" in config:
                    metrics["global_target_epsilon"] = config["global_target_epsilon"]

                trained_parameters = get_model_parameters(model)

            else:
                logger.info(f"[Client {self.partition_id}] Using standard training.")

                enable_grad_diagnostics = config.get("enable_gradient_diagnostics", False)

                if enable_grad_diagnostics:
                    diagnostic_metrics = self._train_with_gradient_diagnostics(
                        net=model, trainloader=self.train_loader, epochs=config["local_epochs"],
                        device=self.device, optimizer=optimizer, criterion=criterion
                    )
                    metrics.update(diagnostic_metrics)

                    if 'grad_p95' in diagnostic_metrics:
                        logger.info(f"[Client {self.partition_id}]  Recommended max_grad_norm: {diagnostic_metrics['grad_p95']:.4f}")
                else:
                    train(
                        net=model, trainloader=self.train_loader, epochs=config["local_epochs"],
                        device=self.device, optimizer=optimizer, criterion=criterion
                    )

                estimated_steps = len(self.train_loader) * config["local_epochs"]
                final_loss, final_accuracy = test(model, self.test_loader, self.device)
                metrics.update({
                    "final_test_loss": float(final_loss),
                    "final_test_accuracy": float(final_accuracy),
                    "train_steps_this_round": int(estimated_steps),
                    "dp_mode": "baseline",
                })
                trained_parameters = get_model_parameters(model)

            metrics.update({
                "learning_rate": learning_rate, "momentum": momentum, "weight_decay": weight_decay,
                "local_epochs": config["local_epochs"], "batch_size": BATCH_SIZE,
            })

            theta_after = trained_parameters
            client_update_l2 = _global_l2_between_param_lists(theta_after, theta_before)
            metrics["client_update_l2_norm"] = float(client_update_l2)

            # Client-level verification runs after the adapter adds noise.

            metrics = self._sanitize_metrics(metrics)

            # Do not overwrite a mode already reported by the privacy path.
            # Only report privacy fields for an actual DP run. Fabricating
            # Laplace/epsilon defaults here charged baseline runs a budget even
            # though no clipping or noise was applied.
            metrics.setdefault("dp_mode", "unknown")
            is_dp_run = bool(config.get("use_dp", False)) and metrics.get("dp_mode") != "baseline"
            if is_dp_run:
                if "noise_mechanism" in config:
                    metrics.setdefault("noise_mechanism", config["noise_mechanism"])
                if "round_target_epsilon" in config:
                    metrics.setdefault("round_target_epsilon", float(config["round_target_epsilon"]))
                if "max_grad_norm" in config:
                    metrics.setdefault("max_grad_norm", float(config["max_grad_norm"]))
                if metrics.get("noise_mechanism") != "gaussian":
                    metrics.pop("delta", None)
            else:
                metrics["dp_mode"] = "baseline"
                for privacy_key in (
                    "noise_mechanism", "round_target_epsilon", "target_epsilon",
                    "epsilon_per_step", "noise_multiplier", "actual_noise_multiplier",
                    "sample_rate", "delta", "max_grad_norm",
                ):
                    metrics.pop(privacy_key, None)

            logger.info(f"[Client {self.partition_id}] Fit completed. Final test loss: {metrics.get('final_test_loss', 'N/A'):.4f}, Test Acc: {metrics.get('final_test_accuracy', 'N/A'):.2f}%, Steps: {metrics.get('train_steps_this_round', 'N/A')}, Client Update L2: {client_update_l2:.6f}")
            return trained_parameters, self.train_dataset_len, metrics

        except Exception as e:
            logger.error(f"[Client {self.partition_id}] FIT FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            gc.collect()

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on the client's local test data.
        """
        model = None
        try:
            if not self.test_loader or self.test_dataset_len == 0:
                raise RuntimeError("Test data loader not initialized properly.")

            model = load_model(DATASET_NAME).to(self.device)
            set_model_parameters(model, parameters)

            loss, accuracy = test(model, self.test_loader, self.device)

            eval_metrics = {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "server_round": config.get("server_round", -1),
            }

            logger.info(f"[Client {self.partition_id}] Evaluate completed. Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            return float(loss), self.test_dataset_len, eval_metrics

        except Exception as e:
            logger.error(f"[Client {self.partition_id}] EVALUATE FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Explicitly trigger garbage collection
            gc.collect()


def client_fn(context: Context) -> Client:
    """Flower client factory based on context for unified DP configuration"""
    from flwr.common.context import Context
    from flwr.common.logger import log
    from logging import INFO

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    use_dp = context.run_config.get("use_dp")
    dp_mode = context.run_config.get("dp_mode")
    non_iid = context.run_config.get("non_iid", False)  # Default to True for Non-IID

    if use_dp is None or dp_mode is None:
        import os
        from experiment_configs import get_experiment_config
        mode = os.environ.get("EXPERIMENT_MODE", "fedavg_baseline")
        try:
            cfg = get_experiment_config(mode)
            use_dp = cfg.get("use_dp", False) if use_dp is None else use_dp
            dp_mode = cfg.get("dp_mode", "opacus") if dp_mode is None else dp_mode
            log(INFO, f"[Client {partition_id}] Fallback config loaded from EXPERIMENT_MODE='{mode}': use_dp={use_dp}, dp_mode={dp_mode}")
        except Exception as e:
            log(INFO, f"[Client {partition_id}] Failed to load fallback config: {e}, using defaults")
            use_dp = False if use_dp is None else use_dp
            dp_mode = "opacus" if dp_mode is None else dp_mode

    client = FlowerClient(partition_id=partition_id, num_partitions=num_partitions, non_iid=non_iid)

    if use_dp and dp_mode == "client":
        mechanism = context.run_config.get("noise_mechanism")
        max_grad_norm = context.run_config.get("max_grad_norm")
        noise_multiplier = context.run_config.get("noise_multiplier")

        if mechanism is None or max_grad_norm is None:
            import os
            from experiment_configs import get_experiment_config
            mode = os.environ.get("EXPERIMENT_MODE", "fedavg_baseline")
            try:
                cfg = get_experiment_config(mode)
                mechanism = cfg.get("noise_mechanism", "gaussian") if mechanism is None else mechanism
                max_grad_norm = cfg.get("max_grad_norm", 1.0) if max_grad_norm is None else max_grad_norm
                noise_multiplier = cfg.get("noise_multiplier") if noise_multiplier is None else noise_multiplier
                log(INFO, f"[Client {partition_id}] Fallback DP params from config: mechanism={mechanism}, max_grad_norm={max_grad_norm}, noise_multiplier={noise_multiplier}")
            except Exception as e:
                log(INFO, f"[Client {partition_id}] Failed to load fallback DP params: {e}")
                mechanism = "gaussian" if mechanism is None else mechanism
                max_grad_norm = 1.0 if max_grad_norm is None else max_grad_norm

        dp_mech = mechanism
        C = max_grad_norm
        noise_multiplier = context.run_config.get("noise_multiplier")

        if noise_multiplier is None and mechanism is not None:
            import os
            from experiment_configs import get_experiment_config
            mode = os.environ.get("EXPERIMENT_MODE", "fedavg_baseline")
            try:
                cfg = get_experiment_config(mode)
                noise_multiplier = cfg.get("noise_multiplier")
            except:
                pass

        if "round_target_epsilon" in context.run_config:
            eps_round = context.run_config["round_target_epsilon"]
        elif "total_epsilon" in context.run_config:
            total_eps = context.run_config["total_epsilon"]
            num_rounds = int(
                context.run_config.get(
                    "privacy_calibration_rounds",
                    os.environ.get("PRIVACY_CALIBRATION_ROUNDS", "500"),
                )
            )
            eps_round = total_eps / num_rounds
        else:
            import os
            from experiment_configs import get_experiment_config
            mode = os.environ.get("EXPERIMENT_MODE", "fedavg_baseline")
            try:
                cfg = get_experiment_config(mode)
                if "round_target_epsilon" in cfg:
                    eps_round = cfg["round_target_epsilon"]
                elif "total_epsilon" in cfg:
                    total_eps = cfg["total_epsilon"]
                    num_rounds = int(cfg.get("privacy_calibration_rounds", 500))
                    eps_round = total_eps / num_rounds
                else:
                    eps_round = 1.0
            except:
                eps_round = 1.0

        if dp_mech == "gaussian":
            if noise_multiplier is not None:
                epsilon = 1.0
                auto_calculate_noise = False
                log(INFO, f"[Client {partition_id}] Using LocalDpFixedMod (mechanism={dp_mech}, noise_multiplier={noise_multiplier:.4f}, client-level DP)")
            else:
                epsilon = eps_round
                auto_calculate_noise = True
                log(INFO, f"[Client {partition_id}] Using LocalDpFixedMod (mechanism={dp_mech}, epsilon={epsilon:.4f}, client-level DP)")
        else:
            epsilon = eps_round
            auto_calculate_noise = True
            log(INFO, f"[Client {partition_id}] Using LocalDpFixedMod (mechanism={dp_mech}, epsilon={epsilon:.4f}, client-level DP)")

        delta = context.run_config.get("delta", 1e-5)

        import sys
        sys.path.append('.')
        from localdp_adapter import create_localdp_adapter

        localdp_adapter = create_localdp_adapter(
            mechanism=dp_mech,
            max_grad_norm=C,
            epsilon=epsilon,
            delta=delta if dp_mech == "gaussian" else None,
            partition_id=partition_id,
            auto_calculate_noise=auto_calculate_noise,
            noise_multiplier=(
                noise_multiplier if (dp_mech == "gaussian" and noise_multiplier is not None) else None
            ),
        )

        base_client = client.to_client()
        dp_client = localdp_adapter.wrap(base_client)
        log(INFO, f"[Client {partition_id}]  Successfully wrapped client with LocalDpModAdapter (mechanism={dp_mech}, C={C})")
        return dp_client

    else:
        if use_dp:
            log(INFO, f"[Client {partition_id}] Using Opacus for DP")
        else:
            log(INFO, f"[Client {partition_id}] No DP enabled")
        return client.to_client()



# Flower ClientApp for simulation engine
app = ClientApp(client_fn=client_fn)
