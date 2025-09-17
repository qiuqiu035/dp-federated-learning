"""
Federated Learning Client with Differential Privacy (Corrected Implementation)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from typing import Dict, List, Tuple
import logging
import gc
import traceback
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "cifar10"
NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_WORKERS = 0


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
    
    def __init__(self, partition_id: int):
        """
        Initializes the client by loading data for its partition.
        """
        self.partition_id = partition_id
        self.device = DEVICE
        
        logger.info(f"[Client {partition_id}] Initializing and loading data...")
        use_pin_memory = self.device.type == 'cuda'
        try:
            train_dataset, test_dataset = load_datasets(
                name=DATASET_NAME,
                num_partitions=NUM_CLIENTS,
                partition_id=self.partition_id
            )
            
            self.train_dataset_len = len(train_dataset)
            self.test_dataset_len = len(test_dataset)
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS,
                pin_memory=use_pin_memory,
                drop_last=True 
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS,
                pin_memory=use_pin_memory
            )
            
            logger.info(f"[Client {partition_id}] Initialization complete. Train: {self.train_dataset_len}, Test: {self.test_dataset_len}")
            
        except Exception as e:
            logger.error(f"[Client {partition_id}] FAILED to initialize: {e}")
            logger.error(traceback.format_exc())
            raise

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on the client's local data.
        """
        try:
            logger.info(f"[Client {self.partition_id}] Starting fit for round {config.get('server_round', 'N/A')}")
            
            if not self.train_loader or self.train_dataset_len == 0:
                raise RuntimeError("Train data loader not initialized properly.")

            model = load_model(DATASET_NAME).to(self.device)
            set_model_parameters(model, parameters)
            
            learning_rate = config.get("learning_rate", 0.01)
            momentum = config.get("momentum", 0.9)
            weight_decay = config.get("weight_decay", 0.0)
            optimizer_type = config.get("optimizer", "sgd").lower()  # New optimizer type configuration

            # Select optimizer based on configuration
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
            criterion = nn.CrossEntropyLoss()
            
            use_dp = config.get("use_dp", False)
            metrics = {}
            
            if use_dp:
                logger.info(f"[Client {self.partition_id}] Using DP training.")
                target_epsilon = config.get("target_epsilon")
                if target_epsilon is None:
                    raise ValueError("target_epsilon must be provided in config for DP training.")
            
                max_grad_norm = config.get("max_grad_norm")
                if max_grad_norm is None:
                    raise ValueError("max_grad_norm must be provided in config for DP training.")

                delta = config.get("delta", 1e-5)
                sample_rate = BATCH_SIZE / self.train_dataset_len 
                
                # Support selecting noise mechanism (backward compatible)
                noise_mechanism = config.get("noise_mechanism", "gaussian")  # Default to Gaussian noise

                if noise_mechanism == "laplace":
                    # Laplace noise mode: requires epsilon_per_step
                    epsilon_per_step = config.get("epsilon_per_step")
                    if epsilon_per_step is None:
                        # If epsilon_per_step is not provided, estimate it based on target_epsilon and the number of training steps
                        estimated_steps = len(self.train_loader) * config["local_epochs"]
                        epsilon_per_step = target_epsilon / estimated_steps
                        logger.info(f"[Client {self.partition_id}] Estimated epsilon_per_step: {epsilon_per_step:.6f} (based on target_epsilon)")
                    
                    dp_engine = OpacusClientDP(
                        max_grad_norm=max_grad_norm,
                        sample_rate=sample_rate,
                        noise_mechanism="laplace",
                        epsilon_per_step=epsilon_per_step,
                        strict_mode=False
                    )
                    
                    private_model, private_optimizer, private_data_loader = dp_engine.attach(
                        model=model, optimizer=optimizer, data_loader=self.train_loader
                    )
                    logger.info(f"[Client {self.partition_id}] Using Laplace noise with epsilon_per_step: {epsilon_per_step}")
                else:
                    # Gaussian noise mode (original method, backward compatible)
                    dp_engine = OpacusClientDP(
                        noise_multiplier=1, max_grad_norm=max_grad_norm, delta=delta,
                        epochs=config["local_epochs"], sample_rate=sample_rate, strict_mode=False
                    )
                    
                    private_model, private_optimizer, private_data_loader = dp_engine.attach(
                        model=model, optimizer=optimizer, data_loader=self.train_loader, target_epsilon=target_epsilon 
                    )
                    logger.info(f"[Client {self.partition_id}] Using Gaussian noise with target_epsilon: {target_epsilon}")
                
                train_metrics = train(
                    net=private_model, trainloader=private_data_loader, epochs=config["local_epochs"],
                    device=self.device, optimizer=private_optimizer, criterion=criterion
                )
                
                privacy_metrics = dp_engine.get_privacy_spent()
                # Evaluate on held-out test set, not training data
                final_loss, final_accuracy = test(private_model, self.test_loader, self.device)
                
                metrics.update({
                    "final_train_loss": float(final_loss), "final_train_accuracy": float(final_accuracy),
                    "privacy_epsilon": privacy_metrics["epsilon"], "privacy_delta": privacy_metrics["delta"],
                    "target_epsilon": target_epsilon,
                    "max_grad_norm": max_grad_norm, "sample_rate": sample_rate,
                })
                if privacy_metrics.get("actual_noise_multiplier") is not None:
                     metrics["actual_noise_multiplier"] = privacy_metrics["actual_noise_multiplier"]
                     logger.info(f"[Client {self.partition_id}] Actual noise_multiplier: {metrics['actual_noise_multiplier']:.4f}")
                     
                trained_parameters = get_model_parameters(private_model)
                
            else:
                logger.info(f"[Client {self.partition_id}] Using standard training.")
                train_metrics = train(
                    net=model, trainloader=self.train_loader, epochs=config["local_epochs"],
                    device=self.device, optimizer=optimizer, criterion=criterion
                )
                final_loss, final_accuracy = test(model, self.test_loader, self.device)
                metrics.update({"final_train_loss": float(final_loss), "final_train_accuracy": float(final_accuracy)})
                trained_parameters = get_model_parameters(model)
            
            metrics.update({
                "learning_rate": learning_rate, "momentum": momentum, "weight_decay": weight_decay,
                "local_epochs": config["local_epochs"], "batch_size": BATCH_SIZE,
            })
            
            logger.info(f"[Client {self.partition_id}] Fit completed. Final loss: {metrics['final_train_loss']:.4f}, Acc: {metrics['final_train_accuracy']:.2f}%")
            return trained_parameters, self.train_dataset_len, metrics

        except Exception as e:
            logger.error(f"[Client {self.partition_id}] FIT FAILED: {e}")
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on the client's local test data.
        """
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


def client_fn(context: Context) -> fl.client.Client:
    """
    Factory function to create a client instance.
    """
    partition_id = context.node_config["partition-id"]
    client = FlowerClient(partition_id)
    return client.to_client()


# Flower ClientApp for simulation engine
app = ClientApp(client_fn=client_fn)
