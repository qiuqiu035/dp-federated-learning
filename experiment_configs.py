# experiment_configs.py

"""
Central configuration file for all experiments.
This is now the SINGLE source of truth for experiment configurations.

Key Changes:
1. Removed flawed set_environment_from_config function entirely
2. Renamed target_epsilon to round_target_epsilon for clarity
3. Simplified to be a pure configuration dictionary
4. Server.py is now responsible for loading and applying these configs
"""

# Privacy budgets are calibrated for the full thesis run, independently of a
# shorter smoke-test execution.
DEFAULT_PRIVACY_CALIBRATION_ROUNDS = 500

# The definitive experiment configurations dictionary
EXPERIMENT_CONFIGS = {
    # ==================== FedAvg Experiments ====================
    "fedavg_baseline": {
        "description": "FedAvg without DP (baseline).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": False,
        "enable_gradient_diagnostics": True,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 1,  # 100% client participation
        "seed": 2,
    },

    "fedavg_privacy": {
        "description": "FedAvg with sample-level Gaussian DP and full participation.",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "gaussian",
        "total_epsilon": 30,
        "max_grad_norm": 1.3,
        "delta": 1e-5,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 1,
        "seed": 2,
    },
    "fedavg_privacy_0.1": {
        "description": "FedAvg with sample-level Gaussian DP and 10% participation.",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "gaussian",
        "total_epsilon": 30,
        "max_grad_norm": 1.3,
        "delta": 1e-5,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 0.1,
        "seed": 2,
    },

    "fedavg_laplace_privacy": {
        "description": "FedAvg with sample-level L2 clipping and coordinate-wise Laplace noise (engineering comparison).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "laplace",
        "total_epsilon": 30,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.3,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 1,
        "seed": 2,
    },

    # ==================== LocalDpMod Experiments ====================


    "fedavg_localdpmod_laplace": {
        "description": "FedAvg with client-level Laplace DP (total epsilon=30).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "laplace",
        "max_grad_norm": 0.03,
        "total_epsilon": 30,  # Fixed: Client-level DP unified budget
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 1,
        "seed": 2,
    },

    "fedavg_localdpmod_gaussian": {
        "description": "FedAvg with client-level Gaussian DP and full participation.",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "gaussian",
        "max_grad_norm": 0.03,
        "noise_multiplier": 5.015147,
        "total_epsilon": 30,  # Adjusted from 150 to 30
        "delta": 1e-5,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 1,
        "seed": 2,
    },

    "fedavg_localdpmod_gaussian_0.1": {
        "description": "FedAvg with client-level Gaussian DP and 10% participation.",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "gaussian",
        "max_grad_norm": 0.03,
        "noise_multiplier": 0.785528,
        "total_epsilon": 30,  # Adjusted from 150 to 30
        "delta": 1e-5,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "client_fraction": 0.1,
        "seed": 2,
    },


}


def get_experiment_config(name: str) -> dict:
    """
    Returns the configuration dictionary for a given experiment name.
    This is now the ONLY function needed in this file.
    """
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Experiment '{name}' not defined in experiment_configs.py")
    config = EXPERIMENT_CONFIGS[name].copy()
    config.setdefault("privacy_calibration_rounds", DEFAULT_PRIVACY_CALIBRATION_ROUNDS)
    return config


def list_available_experiments() -> list:
    """
    Returns a list of all available experiment names.
    """
    return list(EXPERIMENT_CONFIGS.keys())


def list_experiments_by_category() -> dict:
    """
    Returns experiments grouped by category for easier browsing.
    """
    categories = {
        "FedAvg": [],
        "FedAdam": [],
        "Single Krum": [],
        "Multi-Krum": [],
        "Laplace DP": [],
        "LocalDpMod": [],
    }

    for name, config in EXPERIMENT_CONFIGS.items():
        desc = config["description"].lower()

        if "localdpmod" in desc or config.get("dp_mode") == "client":
            categories["LocalDpMod"].append(name)
        elif "laplace" in desc:
            categories["Laplace DP"].append(name)
        elif config.get("aggregation_method") == "fedadam":
            categories["FedAdam"].append(name)
        elif config.get("aggregation_method") == "krum" and config.get("krum_multi", False):
            categories["Multi-Krum"].append(name)
        elif config.get("aggregation_method") == "krum":
            categories["Single Krum"].append(name)
        else:
            categories["FedAvg"].append(name)

    return categories


def print_available_experiments() -> None:
    """
    Print all available experiments organized by category.
    """
    categories = list_experiments_by_category()

    print("=== Available Experiments ===\n")

    for category, experiments in categories.items():
        if experiments:  # Only show categories that have experiments
            print(f"{category}:")
            for exp in experiments:
                config = get_experiment_config(exp)
                print(f"  - {exp}: {config['description']}")
            print()

    print(f"Total: {len(list_available_experiments())} experiments available")
    print("\nUsage:")
    print("  python run_experiment.py <experiment_name>")
    print("  Example: python run_experiment.py fedavg_privacy")


# --- Clean and simple: These are the essential utility functions ---
