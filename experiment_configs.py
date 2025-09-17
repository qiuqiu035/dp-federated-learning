# experiment_configs.py

"""
Central configuration file for all experiments.
Includes support for different aggregation methods and optimization strategies.

Key Changes:
1. FedAdam uses Adam clients, other aggregation methods use SGD clients
2. Target epsilon values updated to proper DP levels (0.15, 0.08, 0.04)
3. Number of rounds is controlled by the server, not individual experiments
"""

# A dictionary holding the configuration for each experiment.
EXPERIMENT_CONFIGS = {
    # ==================== FedAvg Experiments ====================
    "fedavg_baseline": {
        "description": "FedAvg without differential privacy (baseline).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": False,
    },
    "fedavg_low_privacy": {
        "description": "FedAvg + SGD with Gaussian DP (Low Privacy, target_epsilon=0.15).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.15,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "fedavg_medium_privacy": {
        "description": "FedAvg + SGD with Gaussian DP (Medium Privacy, target_epsilon=0.08).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.08,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "fedavg_high_privacy": {
        "description": "FedAvg + SGD with Gaussian DP (High Privacy, target_epsilon=0.04).",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.04,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },

    # ==================== FedAdam Experiments ====================
    "fedadam_baseline": {
        "description": "FedAdam server + Adam client without DP.",
        "aggregation_method": "fedadam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eta": 0.001,
        "client_optimizer": "adam",  
        "client_adam_beta1": 0.9,
        "client_adam_beta2": 0.999,
        "client_adam_eps": 1e-8,
        "use_dp": False,
    },
    "fedadam_low_privacy": {
        "description": "FedAdam server + Adam client with Gaussian DP (Low Privacy).",
        "aggregation_method": "fedadam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eta": 0.001,
        "client_optimizer": "adam",
        "client_adam_beta1": 0.9,
        "client_adam_beta2": 0.999,
        "client_adam_eps": 1e-8,
        "use_dp": True,
        "target_epsilon": 0.15,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "fedadam_medium_privacy": {
        "description": "FedAdam server + Adam client with Gaussian DP (Medium Privacy).",
        "aggregation_method": "fedadam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eta": 0.001,
        "client_optimizer": "adam",
        "client_adam_beta1": 0.9,
        "client_adam_beta2": 0.999,
        "client_adam_eps": 1e-8,
        "use_dp": True,
        "target_epsilon": 0.08,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "fedadam_high_privacy": {
        "description": "FedAdam server + Adam client with Gaussian DP (High Privacy).",
        "aggregation_method": "fedadam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eta": 0.001,
        "client_optimizer": "adam",
        "client_adam_beta1": 0.9,
        "client_adam_beta2": 0.999,
        "client_adam_eps": 1e-8,
        "use_dp": True,
        "target_epsilon": 0.04,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },

    # ==================== Single Krum Experiments ====================
    "single_krum_baseline": {
        "description": "Single Krum (Byzantine-robust) without DP.",
        "aggregation_method": "krum",
        "krum_f": 2,  
        "krum_multi": False,
        "client_optimizer": "sgd",  
        "use_dp": False,
    },
    "single_krum_low_privacy": {
        "description": "Single Krum with Gaussian DP (Low Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": False,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.15,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "single_krum_medium_privacy": {
        "description": "Single Krum with Gaussian DP (Medium Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": False,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.08,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "single_krum_high_privacy": {
        "description": "Single Krum with Gaussian DP (High Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": False,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.04,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },

    # ==================== Multi-Krum Experiments ====================
    "multi_krum_baseline": {
        "description": "Multi-Krum (selects n-f clients) without DP.",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": True,
        "client_optimizer": "sgd",  
        "use_dp": False,
    },
    "multi_krum_low_privacy": {
        "description": "Multi-Krum with Gaussian DP (Low Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": True,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.15,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "multi_krum_medium_privacy": {
        "description": "Multi-Krum with Gaussian DP (Medium Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": True,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.08,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },
    "multi_krum_high_privacy": {
        "description": "Multi-Krum with Gaussian DP (High Privacy).",
        "aggregation_method": "krum",
        "krum_f": 2,
        "krum_multi": True,
        "client_optimizer": "sgd",
        "use_dp": True,
        "target_epsilon": 0.04,
        "max_grad_norm": 1.2,
        "delta": 1e-5,
    },

    # ==================== Laplace DP Experiments ====================
    "fedavg_laplace_low_privacy": {
        "description": "FedAvg with Laplace DP (Low Privacy). epsilon_per_step=0.004687 (~15.0 total)",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "noise_mechanism": "laplace",
        "epsilon_per_step": 0.004687,
        "target_epsilon": 15.0,  
        "max_grad_norm": 1.2,
    },
    "fedavg_laplace_medium_privacy": {
        "description": "FedAvg with Laplace DP (Medium Privacy). epsilon_per_step=0.0025 (~8.0 total)",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "noise_mechanism": "laplace",
        "epsilon_per_step": 0.0025,
        "target_epsilon": 8.0,
        "max_grad_norm": 1.2,
    },
    "fedavg_laplace_high_privacy": {
        "description": "FedAvg with Laplace DP (High Privacy). epsilon_per_step=0.00125 (~4.0 total)",
        "aggregation_method": "fedavg",
        "client_optimizer": "sgd",
        "use_dp": True,
        "noise_mechanism": "laplace",
        "epsilon_per_step": 0.00125,
        "target_epsilon": 4.0,
        "max_grad_norm": 1.2,
    },
}

def get_experiment_config(name: str) -> dict:
    """
    Returns the configuration dictionary for a given experiment name.
    """
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Experiment '{name}' not defined in experiment_configs.py")
    return EXPERIMENT_CONFIGS[name]

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
    }
    
    for name, config in EXPERIMENT_CONFIGS.items():
        desc = config["description"].lower()
        
        if "laplace" in desc:
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

def set_environment_from_config(config: dict) -> None:
    """
    Set environment variables based on configuration dictionary.
    This bridges the gap between the config file and the existing server implementation.
    """
    import os
    
    # Map config keys to environment variable names
    env_mapping = {
        "aggregation_method": "AGGREGATION_METHOD",
        "client_optimizer": "CLIENT_OPTIMIZER",
        
        # FedAdam parameters
        "adam_beta1": "ADAM_BETA1",
        "adam_beta2": "ADAM_BETA2", 
        "adam_eta": "ADAM_ETA",
        
        # Krum parameters
        "krum_f": "KRUM_F",
        "krum_multi": "KRUM_MULTI",
        
        # Client Adam parameters
        "client_adam_beta1": "CLIENT_ADAM_BETA1",
        "client_adam_beta2": "CLIENT_ADAM_BETA2",
        "client_adam_eps": "CLIENT_ADAM_EPS",
        
        # DP parameters
        "noise_mechanism": "NOISE_MECHANISM",
        "epsilon_per_step": "EPSILON_PER_STEP",
        "target_epsilon": "TARGET_EPSILON",
        "max_grad_norm": "MAX_GRAD_NORM",
        "delta": "DELTA",
    }
    
    # Set environment variables
    for config_key, env_var in env_mapping.items():
        if config_key in config:
            value = config[config_key]
            if isinstance(value, bool):
                os.environ[env_var] = "true" if value else "false"
            else:
                os.environ[env_var] = str(value)
    
    # Set use_dp based on the config
    if "use_dp" in config:
        # If use_dp is False, set EXPERIMENT_MODE to baseline
        # If use_dp is True, we'll let the server determine the mode based on other parameters
        if not config["use_dp"]:
            os.environ["EXPERIMENT_MODE"] = "baseline"
        else:
            # Determine DP mode based on parameters
            if "noise_mechanism" in config and config["noise_mechanism"] == "laplace":
                if "epsilon_per_step" in config:
                    eps = config["epsilon_per_step"]
                    if eps >= 0.004:
                        os.environ["EXPERIMENT_MODE"] = "laplace_low_privacy"
                    elif eps >= 0.002:
                        os.environ["EXPERIMENT_MODE"] = "laplace_medium_privacy"
                    else:
                        os.environ["EXPERIMENT_MODE"] = "laplace_high_privacy"
            elif "target_epsilon" in config:
                target_eps = config["target_epsilon"]
                if target_eps >= 0.12:
                    os.environ["EXPERIMENT_MODE"] = "low_privacy"
                elif target_eps >= 0.06:
                    os.environ["EXPERIMENT_MODE"] = "medium_privacy"
                else:
                    os.environ["EXPERIMENT_MODE"] = "high_privacy"
            else:
                os.environ["EXPERIMENT_MODE"] = "medium_privacy" 

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
                print(f"  • {exp}: {config['description']}")
            print()
    
    print(f"Total: {len(list_available_experiments())} experiments available")
    print("\nUsage:")
    print("  python run_experiment.py <experiment_name>")
    print("  Example: python run_experiment.py fedadam_adam_medium_privacy")
