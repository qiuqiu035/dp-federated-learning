# serverless_experiment_configs.py
"""
Serverless version of experiment configurations - Complete mirror of experiment_configs.py
Ensures Server and Serverless versions use identical parameters for fair comparison

Key Features:
1. Field alignment: num_rounds/local_epochs/batch_size/learning_rate/momentum/max_grad_norm etc.
2. DP config alignment: noise_mechanism/sensitivity/delta etc.
3. Names and default values remain consistent
4. Support same experiment modes: baseline, sample-level/client-level x {Gaussian, Laplace}
5. **New**: Unified DP calibration - noise_multiplier and round_target_epsilon auto-calculated by code

Note: round_target_epsilon values in config files are now for documentation purposes only,
actual noise parameters are auto-calculated by DP calibration code based on total privacy budget.
"""

SERVERLESS_DEFAULT_CONFIG = {
    "num_rounds": 500,
    "privacy_calibration_rounds": 500,
    "local_epochs": 1,
    "batch_size": 64,
    "learning_rate": 0.001,
    "momentum": 0.9,  # Changed from 0.0 to 0.9 for better convergence
    "weight_decay": 0.0001,  # Changed from 0.0 to 0.0001 for regularization
    "optimizer": "sgd",

    "num_clients": 100,
    "dataset": "mnist",

    "use_dp": False,

    # Data partitioning (Non-IID by default with Dirichlet)
    "non_iid": False,  # Use DirichletPartitioner for flexible non-IID control
    "alpha": 0.2,  # Dirichlet concentration parameter (smaller = more heterogeneous)

    # Topology configuration (default ring)
    "serverless_topology": "ring",  # "ring" or "ws"
    "ws_k": 2,  # Watts-Strogatz k parameter (only used if topology="ws")
    "ws_p": 0.2,  # Watts-Strogatz p parameter (only used if topology="ws")

    # Diagnostics
    "enable_gradient_diagnostics": False,
}

SERVERLESS_EXPERIMENT_CONFIGS = {

    # ==================== FedAvg Baseline (WS Topology) ====================
    "serverless_fedavg_baseline_ws": {
        "description": "Serverless FedAvg without DP (baseline, WS topology, k=2, p=0.2, Non-IID).",
        "use_dp": False,
        "enable_gradient_diagnostics": False,
        "client_fraction": 1.0,
        "seed": 2,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "serverless_topology": "ws",
        "ws_k": 2,
        "ws_p": 0.2,
        "non_iid": True,
    },

    "serverless_fedavg_baseline_ws_4": {
        "description": "Serverless FedAvg without DP (baseline, WS topology, k=4, p=0.2, Non-IID).",
        "use_dp": False,
        "enable_gradient_diagnostics": False,
        "client_fraction": 1.0,
        "seed": 2,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "serverless_topology": "ws",
        "ws_k": 4,
        "ws_p": 0.2,
        "non_iid": True,
    },


    # ==================== Sample-Level Gaussian DP (WS Topology) ====================
    "serverless_fedavg_privacy_ws": {
        "description": "Serverless FedAvg + Sample-Level Gaussian DP (WS topology, k=2, p=0.2, Non-IID).",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "gaussian",
        "total_epsilon": 30,
        "max_grad_norm": 1.3,
        "delta": 1e-5,
        "client_fraction": 1.0,
        "seed": 2,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "serverless_topology": "ws",
        "ws_k": 2,
        "ws_p": 0.2,
        "non_iid": True,
    },

    "serverless_fedavg_privacy_ws_4": {
        "description": "Serverless FedAvg + Sample-Level Gaussian DP (WS topology, k=4, p=0.2, Non-IID).",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "gaussian",
        "total_epsilon": 30,
        "max_grad_norm": 1.3,
        "delta": 1e-5,
        "client_fraction": 1.0,
        "seed": 2,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "serverless_topology": "ws",
        "ws_k": 4,
        "ws_p": 0.2,
        "non_iid": True,
    },



    "serverless_fedavg_laplace_privacy_ws_4": {
        "description": "Serverless FedAvg with sample-level L2 clipping and coordinate-wise Laplace noise (engineering comparison, WS k=4).",
        "use_dp": True,
        "dp_mode": "opacus",
        "noise_mechanism": "laplace",
        "total_epsilon": 30,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.3,
        "client_fraction": 1.0,
        "seed": 2,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "serverless_topology": "ws",
        "ws_k": 4,
        "ws_p": 0.2,
        "non_iid": True,
    },

    "serverless_fedavg_localdpmod_gaussian_ws": {
        "description": "Serverless FedAvg with Client-Level Gaussian DP (WS topology, k=2, p=0.2, Non-IID).",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "gaussian",
        "max_grad_norm": 0.03,
        "noise_multiplier": 5.015147,
        "total_epsilon": 30,
        "delta": 1e-5,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "client_fraction": 1.0,
        "seed": 2,
        "serverless_topology": "ws",
        "ws_k": 2,
        "ws_p": 0.2,
        "non_iid": True,
    },

    "serverless_fedavg_localdpmod_gaussian_ws_4": {
        "description": "Serverless FedAvg with Client-Level Gaussian DP (WS topology, k=4, p=0.2, Non-IID).",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "gaussian",
        "max_grad_norm": 0.03,
        "noise_multiplier": 5.015147,
        "total_epsilon": 30,
        "delta": 1e-5,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "client_fraction": 1.0,
        "seed": 2,
        "serverless_topology": "ws",
        "ws_k": 4,
        "ws_p": 0.2,
        "non_iid": True,
    },



    "serverless_fedavg_localdpmod_laplace_ws_4": {
        "description": "Serverless FedAvg with Client-Level Laplace DP (WS topology, k=4, p=0.2, Non-IID).",
        "use_dp": True,
        "dp_mode": "client",
        "noise_mechanism": "laplace",
        "max_grad_norm": 0.03,
        "total_epsilon": 30,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "client_fraction": 1.0,
        "seed": 2,
        "serverless_topology": "ws",
        "ws_k": 4,
        "ws_p": 0.2,
        "non_iid": True,
    },

}


def get_serverless_experiment_config(name: str) -> dict:
    if name not in SERVERLESS_EXPERIMENT_CONFIGS:
        raise ValueError(f"Serverless experiment '{name}' not defined in serverless_experiment_configs.py")

    config = SERVERLESS_DEFAULT_CONFIG.copy()
    config.update(SERVERLESS_EXPERIMENT_CONFIGS[name])

    return config


def list_serverless_experiments() -> list:
    return list(SERVERLESS_EXPERIMENT_CONFIGS.keys())


def list_serverless_experiments_by_category() -> dict:
    categories = {
        "Baseline (Ring)": [],
        "Baseline (WS)": [],
        "Sample-Level DP (Ring)": [],
        "Sample-Level DP (WS)": [],
        "Client-Level DP (Ring)": [],
        "Client-Level DP (WS)": [],
        "Quick Tests": [],
    }

    for name, config in SERVERLESS_EXPERIMENT_CONFIGS.items():
        desc = config["description"].lower()
        is_ring = config.get("serverless_topology", "ring") == "ring"
        is_ws = config.get("serverless_topology", "ring") == "ws"

        if "test" in name:
            categories["Quick Tests"].append(name)
        elif not config.get("use_dp", False):
            # Baseline
            if is_ring:
                categories["Baseline (Ring)"].append(name)
            else:
                categories["Baseline (WS)"].append(name)
        elif config.get("dp_mode") == "client":
            # Client-Level DP
            if is_ring:
                categories["Client-Level DP (Ring)"].append(name)
            else:
                categories["Client-Level DP (WS)"].append(name)
        elif config.get("dp_mode") == "opacus":
            # Sample-Level DP
            if is_ring:
                categories["Sample-Level DP (Ring)"].append(name)
            else:
                categories["Sample-Level DP (WS)"].append(name)
        else:
            # Fallback
            if is_ring:
                categories["Baseline (Ring)"].append(name)
            else:
                categories["Baseline (WS)"].append(name)

    return categories


def print_serverless_experiments() -> None:
    categories = list_serverless_experiments_by_category()

    print("=== Available Serverless Experiments ===\n")

    for category, experiments in categories.items():
        if experiments:  # Only show categories with experiments
            print(f"{category}:")
            for exp in experiments:
                config = get_serverless_experiment_config(exp)
                print(f"  - {exp}: {config['description']}")
            print()

    print(f"Total: {len(list_serverless_experiments())} serverless experiments available")
    print("\nUsage:")
    print("  python run_serverless_experiment.py <experiment_name>")
    print("  Example: python run_serverless_experiment.py serverless_fedavg_privacy_ws")


def get_comparison_pairs() -> list:
    pairs = [
        ("fedavg_baseline", "serverless_fedavg_baseline_ws"),
        ("fedavg_privacy", "serverless_fedavg_privacy_ws"),
        ("fedavg_laplace_privacy", "serverless_fedavg_laplace_privacy_ws_4"),
        ("fedavg_localdpmod_gaussian", "serverless_fedavg_localdpmod_gaussian_ws"),
        ("fedavg_localdpmod_laplace", "serverless_fedavg_localdpmod_laplace_ws_4"),
    ]

    return pairs


def validate_config_consistency(strict_mode: bool = True) -> None:
    """
    Validate server and serverless comparison pairs to prevent configuration drift.

    Args:
        strict_mode: If True, raises RuntimeError on inconsistency; if False, only warns
    """
    import sys
    sys.path.append('.')

    try:
        from experiment_configs import get_experiment_config
    except ImportError:
        print("  Warning: Cannot import experiment_configs.py for validation")
        if strict_mode:
            raise RuntimeError("Cannot validate config consistency - experiment_configs.py not found!")
        return

    pairs = get_comparison_pairs()
    print("=" * 80)
    print("  CONFIG CONSISTENCY VALIDATION (Server vs Serverless)")
    print("=" * 80)
    print()

    inconsistent_count = 0
    critical_errors = []

    # These fields must match in each server/serverless comparison pair.
    critical_fields = [
        "use_dp",
        "dp_mode",
        "noise_mechanism",
        "max_grad_norm",
        "delta",
        "noise_multiplier",
        "total_epsilon",
    ]

    # Differences in these fields are reported as warnings.
    warning_fields = [
        "round_target_epsilon",
        "client_fraction",
        "seed",
        "weight_decay",
    ]

    for server_exp, serverless_exp in pairs:
        try:
            server_config = get_experiment_config(server_exp)
            serverless_config = get_serverless_experiment_config(serverless_exp)

            # Check the fields that define a controlled comparison.
            critical_inconsistent = []
            for field in critical_fields:
                server_val = server_config.get(field)
                serverless_val = serverless_config.get(field)

                # Compare privacy fields only when both configurations use DP.
                if field in ["noise_mechanism", "max_grad_norm", "delta", "noise_multiplier", "total_epsilon"]:
                    if not (server_config.get("use_dp") and serverless_config.get("use_dp")):
                        continue

                # Noise multipliers are directly comparable for client-level Gaussian DP.
                if field == "noise_multiplier":
                    is_client_gaussian_server = (server_config.get("dp_mode") == "client" and
                                                 server_config.get("noise_mechanism") == "gaussian")
                    is_client_gaussian_serverless = (serverless_config.get("dp_mode") == "client" and
                                                     serverless_config.get("noise_mechanism") == "gaussian")
                    if not (is_client_gaussian_server and is_client_gaussian_serverless):
                        continue

                if server_val != serverless_val:
                    critical_inconsistent.append(
                        f"   {field}: server={server_val} vs serverless={serverless_val}"
                    )

            # Report non-critical configuration differences.
            warning_inconsistent = []
            for field in warning_fields:
                server_val = server_config.get(field)
                serverless_val = serverless_config.get(field)
                if server_val != serverless_val:
                    warning_inconsistent.append(
                        f"    {field}: server={server_val} vs serverless={serverless_val}"
                    )

            if critical_inconsistent or warning_inconsistent:
                print(f" {server_exp} <-> {serverless_exp}:")

                if critical_inconsistent:
                    print("   CRITICAL INCONSISTENCIES:")
                    for msg in critical_inconsistent:
                        print(msg)
                    critical_errors.append((server_exp, serverless_exp, critical_inconsistent))
                    inconsistent_count += 1

                if warning_inconsistent:
                    print("    WARNINGS (Non-critical):")
                    for msg in warning_inconsistent:
                        print(msg)

                print()
            else:
                print(f" {server_exp} <-> {serverless_exp}: All critical fields match")

        except Exception as e:
            error_msg = f"Error comparing {server_exp} <-> {serverless_exp}: {e}"
            print(f"   {error_msg}")
            critical_errors.append((server_exp, serverless_exp, [error_msg]))
            inconsistent_count += 1

    print()
    print("=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  Total pairs checked: {len(pairs)}")
    print(f"   Consistent: {len(pairs) - inconsistent_count}")
    print(f"   Inconsistent: {inconsistent_count}")
    print()

    if critical_errors:
        print(" CRITICAL ERRORS DETECTED:")
        for server_exp, serverless_exp, errors in critical_errors:
            print(f"  - {server_exp} <-> {serverless_exp}")
            for err in errors[:3]:  # Show first 3 errors
                print(f"    {err}")
        print()

        if strict_mode:
            print(" VALIDATION FAILED: Config inconsistencies detected!")
            print("   Fix the configurations before running experiments.")
            print("   Or set strict_mode=False to proceed with warnings.")
            raise RuntimeError(
                f"Config validation failed: {inconsistent_count} inconsistent pairs found. "
                f"See details above."
            )
    else:
        print(" ALL CONFIGURATIONS VALIDATED SUCCESSFULLY!")
        print("   Server and Serverless configs are consistent.")

    print("=" * 80)
