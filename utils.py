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
    criterion: nn.Module
) -> Dict[str, List[float]]:
    """
    Generic PyTorch training function.

    Args:
        net (nn.Module): The model to train.
        trainloader (DataLoader): The DataLoader for the training set.
        epochs (int): Number of training epochs.
        device (torch.device): The device (CPU or GPU) to run training on.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.

    Returns:
        A dictionary containing training metrics, e.g., {"loss": [...]}.
    """
    net.train()
    metrics = {"loss": []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        metrics["loss"].append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    return metrics


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
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    
    return avg_loss, accuracy



def load_datasets(name: str, num_partitions: int, partition_id: int, data_dir: str = "./data") -> Tuple[Subset, datasets.VisionDataset]:
    """
    Main data loading function that handles both CIFAR-10 and MNIST datasets.
    
    Args:
        name (str): Dataset name ('cifar10' or 'mnist').
        num_partitions (int): The number of client partitions.
        partition_id (int): The ID of the partition to return (0-indexed).
        data_dir (str): The directory to store data and cache.

    Returns:
        Tuple of (train_dataset, test_dataset) for the specified partition.
    """
    if name.lower() == 'cifar10':
        partitions = prepare_cifar10_partitions(num_partitions, data_dir)
    elif name.lower() == 'mnist':
        partitions = prepare_mnist_partitions(num_partitions, data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: 'cifar10', 'mnist'")
    
    if partition_id >= len(partitions):
        raise ValueError(f"Partition ID {partition_id} out of range. Available partitions: 0-{len(partitions)-1}")
    
    return partitions[partition_id]


def prepare_cifar10_partitions(num_partitions: int, data_dir: str = "./data") -> List[Tuple[Subset, datasets.CIFAR10]]:
    """
    Downloads, partitions, and caches the CIFAR-10 dataset for federated learning.
    This function ensures the process is run only once and loads from cache on subsequent calls.

    Args:
        num_partitions (int): The number of client partitions to create.
        data_dir (str): The directory to store data and cache.

    Returns:
        A list of tuples, where each tuple contains (train_subset, full_test_set).
    """
    cache_dir = Path(data_dir) / "cache"
    cache_path = cache_dir / f"cifar10_partitions_{num_partitions}.pkl"
    
    if cache_path.exists():
        logger.info(f"Loading cached CIFAR-10 partitions from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-preparing dataset.")
    
    logger.info(f"Preparing and caching CIFAR-10 partitions for {num_partitions} clients...")
    
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
    
    # Split training data into partitions
    partition_size = len(full_trainset) // num_partitions
    lengths = [partition_size] * num_partitions
    # Distribute the remainder
    for i in range(len(full_trainset) % num_partitions):
        lengths[i] += 1
        
    train_subsets = random_split(full_trainset, lengths, generator=torch.Generator().manual_seed(42))
    
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


def prepare_mnist_partitions(num_partitions: int, data_dir: str = "./data") -> List[Tuple[Subset, datasets.MNIST]]:
    """
    Downloads, partitions, and caches the MNIST dataset.
    """
    cache_dir = Path(data_dir) / "cache"
    cache_path = cache_dir / f"mnist_partitions_{num_partitions}.pkl"
    
    if cache_path.exists():
        logger.info(f"Loading cached MNIST partitions from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-preparing dataset.")

    logger.info(f"Preparing and caching MNIST partitions for {num_partitions} clients...")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    full_trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    partition_size = len(full_trainset) // num_partitions
    lengths = [partition_size] * num_partitions
    for i in range(len(full_trainset) % num_partitions):
        lengths[i] += 1

    train_subsets = random_split(full_trainset, lengths, generator=torch.Generator().manual_seed(42))
    client_partitions = [(subset, full_testset) for subset in train_subsets]

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(client_partitions, f)
        logger.info(f"MNIST partitions cached at {cache_path}")
    except Exception as e:
        logger.warning(f"Could not cache partitions: {e}")
        
    return client_partitions



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
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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