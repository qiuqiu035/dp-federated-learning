"""
Utility functions for the federated learning system
Includes model definitions, training/testing functions, and data loading utilities with robust caching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, List
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from seed_manager import get_seed_manager
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    A simple CNN model for CIFAR-10 classification with robust, dynamic feature calculation.
    This model outputs raw logits, suitable for use with nn.CrossEntropyLoss.
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.features(dummy_input)
            self._flattened_features = dummy_output.view(dummy_output.size(0), -1).shape[1]

        logger.info(f"SimpleCNN: Automatically calculated flattened features = {self._flattened_features}")

        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    """
    A simple MLP model for MNIST classification.
    This model outputs raw logits, suitable for use with nn.CrossEntropyLoss.
    """

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- Training and Testing Functions ---

def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Generic PyTorch training function.

    Args:
        net (nn.Module): The model to train.
        trainloader (DataLoader): The DataLoader for the training set.
        epochs (int): Number of training epochs.
        device (torch.device): The device (CPU or GPU) to run training on.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        verbose (bool): Whether to print detailed logs.

    Returns:
        A dictionary containing training metrics with simple float values.
    """
    net.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        running_loss = 0.0
        num_epoch_batches = len(trainloader)

        for i, (images, labels) in enumerate(trainloader):
            # Use asynchronous transfers when pinned memory is enabled.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1

            # Log frequency control - only log every 100 batches if verbose
            if verbose and (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{num_epoch_batches}], Loss: {loss.item():.4f}')

        avg_epoch_loss = running_loss / num_epoch_batches
        if verbose:
            logger.info(f"Epoch [{epoch+1}/{epochs}] finished with avg loss: {avg_epoch_loss:.4f}")

    # Return simple metrics that are serializable
    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"train_loss": float(avg_train_loss)}


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Standard PyTorch testing function.

    Args:
        net (nn.Module): The model to evaluate.
        testloader (DataLoader): The DataLoader for the test set.
        device (torch.device): The device (CPU or GPU) to run evaluation on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    criterion = nn.CrossEntropyLoss(reduction='sum')
    net.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in testloader:
            # Use asynchronous transfers when pinned memory is enabled.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    return avg_loss, accuracy



def load_datasets(name: str, num_partitions: int, partition_id: int, data_dir: str = "./data", non_iid: bool = True, alpha: float = 0.2) -> Tuple[Subset, datasets.VisionDataset]:
    """
    Main data loading function that handles both CIFAR-10 and MNIST datasets.

    Args:
        name (str): Dataset name ('cifar10' or 'mnist').
        num_partitions (int): The number of client partitions.
        partition_id (int): The ID of the partition to return (0-indexed).
        data_dir (str): The directory to store data and cache.
        non_iid (bool): If True, use DirichletPartitioner for non-IID data. Default True.
        alpha (float): Dirichlet concentration parameter for non-IID partitioning.
                       Smaller values = more heterogeneous. Default 0.1.

    Returns:
        Tuple of (train_dataset, test_dataset) for the specified partition.
    """
    if name.lower() == 'cifar10':
        partitions = prepare_cifar10_partitions(num_partitions, data_dir, non_iid=non_iid, alpha=alpha)
    elif name.lower() == 'mnist':
        partitions = prepare_mnist_partitions(num_partitions, data_dir, non_iid=non_iid, alpha=alpha)
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: 'cifar10', 'mnist'")

    if partition_id >= len(partitions):
        raise ValueError(f"Partition ID {partition_id} out of range. Available partitions: 0-{len(partitions)-1}")

    if non_iid:
        logger.info(f"Loaded non-IID partitions (DirichletPartitioner with alpha={alpha})")
    else:
        logger.info(f"Loaded IID partitions")

    return partitions[partition_id]


def prepare_cifar10_partitions(num_partitions: int, data_dir: str = "./data", non_iid: bool = True, alpha: float = 0.2) -> List[Tuple[Subset, datasets.CIFAR10]]:
    """
    Downloads, partitions, and caches the CIFAR-10 dataset for federated learning.
    This function ensures the process is run only once and loads from cache on subsequent calls.

    Args:
        num_partitions (int): The number of client partitions to create.
        data_dir (str): The directory to store data and cache.
        non_iid (bool): If True, use DirichletPartitioner for non-IID data. Default True.
        alpha (float): Dirichlet concentration parameter. Smaller values = more heterogeneous. Default 0.1.

    Returns:
        A list of tuples, where each tuple contains (train_subset, full_test_set).
    """
    cache_dir = Path(data_dir) / "cache"
    cache_filename = f"cifar10_partitions_dirichlet_alpha{alpha}_{num_partitions}.pkl" if non_iid else f"cifar10_partitions_{num_partitions}.pkl"
    cache_path = cache_dir / cache_filename

    if cache_path.exists():
        logger.info(f"Loading cached CIFAR-10 partitions from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-preparing dataset.")

    partition_type = f"non-IID (Dirichlet alpha={alpha})" if non_iid else "IID"
    logger.info(f"Preparing and caching CIFAR-10 {partition_type} partitions for {num_partitions} clients...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    full_testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    if non_iid:
        # Non-IID: Use Flower's DirichletPartitioner
        logger.info(f"Using DirichletPartitioner with alpha={alpha} (smaller alpha = more heterogeneous)")

        # Create DirichletPartitioner
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,  # Concentration parameter: smaller = more heterogeneous
            seed=42  # Fixed seed for reproducibility
        )

        # Load dataset using flwr_datasets with the partitioner
        fds = FederatedDataset(
            dataset="cifar10",
            partitioners={"train": partitioner}
        )

        # Create client partitions
        client_partitions = []
        for i in range(num_partitions):
            partition_dataset = fds.load_partition(i, "train")
            # Convert to PyTorch dataset with transforms
            train_subset = _convert_to_torch_dataset(partition_dataset, transform_train, is_cifar=True)
            client_partitions.append((train_subset, full_testset))

    else:
        # IID: Original random uniform partitioning
        # Shuffle the complete training set immediately after loading it.
        indices = torch.randperm(len(full_trainset), generator=torch.Generator().manual_seed(42))
        full_trainset = torch.utils.data.Subset(full_trainset, indices)

        # Split training data into partitions
        partition_size = len(full_trainset) // num_partitions
        lengths = [partition_size] * num_partitions
        # Distribute the remainder
        for i in range(len(full_trainset) % num_partitions):
            lengths[i] += 1

        # Use seed manager for reproducible data splitting
        seed_mgr = get_seed_manager()
        split_seed = seed_mgr.get_data_split_seed()
        train_subsets = random_split(full_trainset, lengths, generator=torch.Generator().manual_seed(split_seed))

        # Each client gets its own training subset and access to the full test set
        client_partitions = [(subset, full_testset) for subset in train_subsets]

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(client_partitions, f)
        logger.info(f"CIFAR-10 partitions cached at {cache_path}")
    except Exception as e:
        logger.warning(f"Could not cache partitions: {e}")

    return client_partitions


def _prepare_mnist_partitions_unlocked(num_partitions: int, data_dir: str = "./data", non_iid: bool = True, alpha: float = 0.2) -> List[Tuple[Subset, datasets.MNIST]]:
    """
    Downloads, partitions, and caches the MNIST dataset.

    Args:
        num_partitions (int): The number of client partitions to create.
        data_dir (str): The directory to store data and cache.
        non_iid (bool): If True, use DirichletPartitioner for non-IID data. Default True.
        alpha (float): Dirichlet concentration parameter. Smaller values = more heterogeneous. Default 0.1.

    Returns:
        A list of tuples, where each tuple contains (train_subset, full_test_set).
    """
    cache_dir = Path(data_dir) / "cache"
    cache_filename = f"mnist_partitions_dirichlet_alpha{alpha}_{num_partitions}.pkl" if non_iid else f"mnist_partitions_{num_partitions}.pkl"
    cache_path = cache_dir / cache_filename

    if cache_path.exists():
        logger.info(f"Loading cached MNIST partitions from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-preparing dataset.")

    partition_type = f"non-IID (Dirichlet alpha={alpha})" if non_iid else "IID"
    logger.info(f"Preparing and caching MNIST {partition_type} partitions for {num_partitions} clients...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    full_trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if non_iid:
        # Non-IID: Use Flower's DirichletPartitioner
        logger.info(f"Using DirichletPartitioner with alpha={alpha} (smaller alpha = more heterogeneous)")

        # Create DirichletPartitioner
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,  # Concentration parameter: smaller = more heterogeneous
            min_partition_size=1,
            seed=42  # Fixed seed for reproducibility
        )

        # Load dataset using flwr_datasets with the partitioner
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner}
        )

        # Create client partitions
        client_partitions = []
        for i in range(num_partitions):
            partition_dataset = fds.load_partition(i, "train")
            # Convert to PyTorch dataset with transforms
            train_subset = _convert_to_torch_dataset(partition_dataset, transform, is_cifar=False)
            client_partitions.append((train_subset, full_testset))

    else:
        # IID: Original random uniform partitioning
        # Shuffle the complete training set immediately after loading it.
        indices = torch.randperm(len(full_trainset), generator=torch.Generator().manual_seed(42))
        full_trainset = torch.utils.data.Subset(full_trainset, indices)

        partition_size = len(full_trainset) // num_partitions
        lengths = [partition_size] * num_partitions
        for i in range(len(full_trainset) % num_partitions):
            lengths[i] += 1

        # Use seed manager for reproducible data splitting
        seed_mgr = get_seed_manager()
        split_seed = seed_mgr.get_data_split_seed()
        train_subsets = random_split(full_trainset, lengths, generator=torch.Generator().manual_seed(split_seed))
        client_partitions = [(subset, full_testset) for subset in train_subsets]

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(client_partitions, f)
        logger.info(f"MNIST partitions cached at {cache_path}")
    except Exception as e:
        logger.warning(f"Could not cache partitions: {e}")

    return client_partitions


def prepare_mnist_partitions(num_partitions: int, data_dir: str = "./data", non_iid: bool = True, alpha: float = 0.2) -> List[Tuple[Subset, datasets.MNIST]]:
    """Prepare MNIST once at a time across concurrent client processes."""
    lock_dir = Path(data_dir) / "cache"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "mnist_prepare.lock"

    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0)
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            return _prepare_mnist_partitions_unlocked(
                num_partitions=num_partitions,
                data_dir=data_dir,
                non_iid=non_iid,
                alpha=alpha,
            )
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)



def _convert_to_torch_dataset(partition_dataset, transform, is_cifar: bool = True):
    """
    Helper function to convert Flower dataset partition to PyTorch dataset.

    Args:
        partition_dataset: Flower dataset partition (datasets.Dataset)
        transform: PyTorch transforms to apply
        is_cifar: True for CIFAR-10, False for MNIST

    Returns:
        PyTorch Dataset compatible with DataLoader
    """
    # Flower datasets return HuggingFace Dataset objects
    # We need to convert them to PyTorch format

    class FlowerTorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform, is_cifar):
            self.hf_dataset = hf_dataset
            self.transform = transform
            self.is_cifar = is_cifar

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            sample = self.hf_dataset[idx]

            # Get image and label
            # Flower datasets use 'img' or 'image' for images, 'label' for labels
            if 'img' in sample:
                img = sample['img']
            elif 'image' in sample:
                img = sample['image']
            else:
                raise KeyError(f"Image key not found in sample: {sample.keys()}")

            label = sample['label']

            # Convert to PIL Image if needed
            if not isinstance(img, Image.Image):
                if self.is_cifar:
                    # CIFAR-10: RGB image
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    else:
                        img = Image.fromarray(np.array(img))
                else:
                    # MNIST: Grayscale image
                    if isinstance(img, np.ndarray):
                        if len(img.shape) == 3:
                            img = img.squeeze()
                        img = Image.fromarray(img, mode='L')
                    else:
                        img_array = np.array(img)
                        if len(img_array.shape) == 3:
                            img_array = img_array.squeeze()
                        img = Image.fromarray(img_array, mode='L')

            # Apply transform
            if self.transform:
                img = self.transform(img)

            return img, label

    return FlowerTorchDataset(partition_dataset, transform, is_cifar)



def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extracts model parameters as a list of NumPy arrays."""
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Sets model parameters from a list of NumPy arrays."""
    params_dict = zip(model.parameters(), parameters)
    for param, new_param_np in params_dict:
        new_param = torch.from_numpy(new_param_np).to(device=param.device, dtype=param.dtype)
        param.data.copy_(new_param)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing Utility Functions ---")

    # Test device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("\n[1] Testing SimpleCNN model instantiation...")
    try:
        cnn_model = SimpleCNN(num_classes=10).to(device)
        logger.info("SimpleCNN instantiated successfully.")
        dummy_cnn_input = torch.randn(2, 3, 32, 32).to(device)
        output = cnn_model(dummy_cnn_input)
        logger.info(f"Forward pass successful! Output shape: {output.shape}")
        assert output.shape == (2, 10)
    except Exception as e:
        logger.error(f"SimpleCNN test failed: {e}", exc_info=True)

    logger.info("\n[2] Testing MLP model instantiation...")
    try:
        mlp_model = MLP(num_classes=10).to(device)
        logger.info("MLP instantiated successfully.")
        dummy_mlp_input = torch.randn(2, 1, 28, 28).to(device)
        output = mlp_model(dummy_mlp_input)
        logger.info(f"Forward pass successful! Output shape: {output.shape}")
        assert output.shape == (2, 10)
    except Exception as e:
        logger.error(f"MLP test failed: {e}", exc_info=True)

    logger.info("\n[3] Testing load_datasets function...")
    try:
        train_dataset, test_dataset = load_datasets("cifar10", num_partitions=2, partition_id=0, data_dir="./test_data")
        logger.info(f"CIFAR-10 - Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        assert len(train_dataset) > 0

        train_dataset, test_dataset = load_datasets("mnist", num_partitions=2, partition_id=1, data_dir="./test_data")
        logger.info(f"MNIST - Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        assert len(train_dataset) > 0
    except Exception as e:
        logger.error(f"load_datasets test failed: {e}", exc_info=True)

    logger.info("\n[4] Testing train and test functions...")
    try:
        # Create a small dataset for testing
        train_dataset, test_dataset = load_datasets("mnist", num_partitions=10, partition_id=0, data_dir="./test_data")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Test training with new signature
        model = MLP(num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        logger.info("Starting training test...")
        metrics = train(model, train_loader, epochs=2, device=device, optimizer=optimizer, criterion=criterion)
        logger.info(f"Training metrics: {metrics}")

        # Test evaluation
        logger.info("Starting testing...")
        avg_loss, accuracy = test(model, test_loader, device=device)
        logger.info(f"Test results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    except Exception as e:
        logger.error(f"Train/test function test failed: {e}", exc_info=True)

    logger.info("\n--- All tests completed ---")


# ============================================================================
# Topology Functions for Serverless Communication
# ============================================================================

import networkx as nx
import random
import json


def build_topology(kind: str, num_nodes: int, k: int = 2, p: float = 0.2, seed: int = 42) -> nx.Graph:
    """
    Build a fixed topology for serverless communication.

    Args:
        kind: Topology type, "ring" or "ws" (Watts-Strogatz)
        num_nodes: Number of nodes (clients)
        k: Number of nearest neighbors (for WS). Must be even for some networkx versions.
        p: Rewiring probability (for WS)
        seed: Random seed for reproducibility

    Returns:
        A NetworkX graph representing the topology
    """
    random.seed(seed)
    np.random.seed(seed)

    if kind == "ring":
        # Ring topology: each node has degree 2 (left and right neighbors)
        g = nx.cycle_graph(num_nodes)
        logger.info(f"Built ring topology: {num_nodes} nodes, degree=2")
        return g

    if kind == "ws":
        # Watts-Strogatz: small-world network
        try:
            g = nx.connected_watts_strogatz_graph(n=num_nodes, k=k, p=p, tries=100, seed=seed)
            logger.info(f"Built WS topology: {num_nodes} nodes, k={k}, p={p}")
            return g
        except Exception as e:
            # Fallback for odd k (some networkx versions require k to be even)
            logger.warning(f"WS graph generation failed with k={k}: {e}")
            logger.warning(f"Attempting fallback with k_even={k+1 if k % 2 == 1 else k}")

            k_even = k + 1 if k % 2 == 1 else k
            g = nx.connected_watts_strogatz_graph(n=num_nodes, k=k_even, p=p, tries=100, seed=seed)

            # If original k was odd, remove edges to reduce degree
            if k % 2 == 1:
                edges_to_remove = []
                if num_nodes % 2 == 0:
                    # Remove perfect matching edges
                    for i in range(num_nodes // 2):
                        u, v = i, (i + num_nodes // 2) % num_nodes
                        if g.has_edge(u, v):
                            edges_to_remove.append((u, v))

                # Fallback: random edge removal
                while len(edges_to_remove) < num_nodes // 2:
                    edge = random.choice(list(g.edges()))
                    if edge not in edges_to_remove and (edge[1], edge[0]) not in edges_to_remove:
                        edges_to_remove.append(edge)

                g.remove_edges_from(edges_to_remove)

                # Check connectivity
                if not nx.is_connected(g):
                    logger.warning(f"Graph became disconnected after edge removal. Reverting to k_even={k_even}")
                    g = nx.connected_watts_strogatz_graph(n=num_nodes, k=k_even, p=p, tries=100, seed=seed)
                else:
                    logger.info(f"Successfully reduced to approximate k={k} (originally k_even={k_even})")

            logger.info(f"Built WS topology: {num_nodes} nodes, kapproximately{k}, p={p}")
            return g

    raise ValueError(f"Unknown topology kind: {kind}. Supported: 'ring', 'ws'")


def graph_to_neighbors(g: nx.Graph) -> dict:
    """
    Convert NetworkX graph to neighbor adjacency dictionary.

    Args:
        g: NetworkX graph

    Returns:
        Dictionary mapping node_id to sorted list of neighbor_ids
    """
    neighbors = {int(u): sorted([int(v) for v in g.neighbors(u)]) for u in g.nodes()}
    return neighbors


def save_topology(neighbors: dict, filepath: str, metadata: dict = None):
    """
    Save topology to JSON file.

    Args:
        neighbors: Neighbor adjacency dictionary
        filepath: Path to save JSON file
        metadata: Optional metadata (kind, k, p, seed, etc.)
    """
    data = {
        "neighbors": neighbors,
        "metadata": metadata or {}
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Topology saved to {filepath}")


def load_topology(filepath: str) -> tuple:
    """
    Load topology from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Tuple of (neighbors_dict, metadata_dict)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data["neighbors"], data.get("metadata", {})


class ServerlessNeighborAverager:
    """
    Manages fixed topology and performs single-round gossip averaging for serverless FL.
    """

    def __init__(self, num_clients: int, topo_kind: str, k: int = 2, p: float = 0.2, seed: int = 42):
        """
        Initialize the neighbor averager.

        Args:
            num_clients: Number of clients
            topo_kind: Topology type, "ring" or "ws"
            k: Number of nearest neighbors (for WS)
            p: Rewiring probability (for WS)
            seed: Random seed
        """
        self.num_clients = num_clients
        self.topo_kind = topo_kind
        self.k = k
        self.p = p
        self.seed = seed
        self.neighbors = None
        self.graph = None

    def initialize_topology(self):
        """Initialize and cache the topology. Call once before training."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Initializing Serverless Topology")
        logger.info(f"{'='*60}")
        logger.info(f"Type: {self.topo_kind}")
        logger.info(f"Clients: {self.num_clients}")
        if self.topo_kind == "ws":
            logger.info(f"k={self.k}, p={self.p}")
        logger.info(f"Seed: {self.seed}")

        # Build topology
        self.graph = build_topology(
            kind=self.topo_kind,
            num_nodes=self.num_clients,
            k=self.k,
            p=self.p,
            seed=self.seed
        )

        # Convert to neighbor dictionary
        self.neighbors = graph_to_neighbors(self.graph)

        # Sanity check: print first 3 nodes' neighbors
        logger.info(f"\nSanity check - First 3 nodes' neighbors:")
        for i in range(min(3, self.num_clients)):
            logger.info(f"  Node {i} -> {self.neighbors[i]}")

        # Statistics
        degrees = [len(nbrs) for nbrs in self.neighbors.values()]
        logger.info(f"\nTopology statistics:")
        logger.info(f"  Min degree: {min(degrees)}")
        logger.info(f"  Max degree: {max(degrees)}")
        logger.info(f"  Avg degree: {sum(degrees) / len(degrees):.2f}")
        logger.info(f"  Connected: {nx.is_connected(self.graph)}")
        logger.info(f"{'='*60}\n")

    def save_topology_to_file(self, results_dir: str):
        """Save topology to results directory."""
        from pathlib import Path
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        filename = f"topology_{self.topo_kind}_k{self.k}_p{self.p}_n{self.num_clients}_seed{self.seed}.json"
        filepath = results_path / filename

        metadata = {
            "kind": self.topo_kind,
            "num_clients": self.num_clients,
            "k": self.k,
            "p": self.p,
            "seed": self.seed,
            "connected": nx.is_connected(self.graph)
        }

        save_topology(self.neighbors, str(filepath), metadata)
        return str(filepath)

    def single_gossip(self, client_params: List[List[np.ndarray]],
                     client_sample_counts: List[int]) -> List[List[np.ndarray]]:
        """
        Perform one round of gossip averaging with neighbors using weighted averaging by sample counts.

        Args:
            client_params: List of parameter lists, one per client
                          Shape: [num_clients][num_layers][layer_shape]
            client_sample_counts: List of sample counts for each client
                                 Shape: [num_clients]

        Returns:
            Averaged parameters with same shape (weighted by sample counts)
        """
        if self.neighbors is None:
            raise RuntimeError("Topology not initialized. Call initialize_topology() first.")

        new_params = []

        for cid in range(self.num_clients):
            # Include self + neighbors
            group = [cid] + self.neighbors[cid]

            # Get sample counts for all nodes in group
            group_sample_counts = [client_sample_counts[node_id] for node_id in group]
            total_samples = sum(group_sample_counts)

            # Average each layer using weighted average
            avg_layers = []
            for layer_idx in range(len(client_params[cid])):
                # Compute weighted average: sum(weight_i * param_i) / sum(weight_i)
                weighted_sum = sum(
                    client_sample_counts[node_id] * client_params[node_id][layer_idx]
                    for node_id in group
                )
                avg_layer = weighted_sum / float(total_samples)
                avg_layers.append(avg_layer)

            new_params.append(avg_layers)

        return new_params
