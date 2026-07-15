# run_unified_serverless_pytorch.py
"""

Run federated learning experiments using a unified Serverless Runner
1. Fully reuses the `FlowerClient` and DP modules from `client.py`.
2. Uses exactly the same algorithmic implementation as the Server version.
3. Supports running all experiment types through configuration files.

"""

import sys
import os
import torch
import numpy as np
import argparse
import logging
from pathlib import Path

# Keep runner logs together with experiment output.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

from serverless_runner import run_unified_serverless_fl, UnifiedDPConfig
from serverless_experiment_configs import (
    SERVERLESS_EXPERIMENT_CONFIGS,
    get_serverless_experiment_config,
    list_serverless_experiments,
    print_serverless_experiments,
    list_serverless_experiments_by_category
)


def config_to_dp_config(config: dict) -> UnifiedDPConfig:
    dp_config = UnifiedDPConfig()

    dp_config.enable_dp = config.get("use_dp", False)

    if dp_config.enable_dp:
        dp_mode = config.get("dp_mode", "client")
        if dp_mode in {"opacus", "sample"}:
            dp_config.mode = "sample"
        else:
            dp_config.mode = "client"

        dp_config.mechanism = config.get("noise_mechanism", "gaussian")
        dp_config.C = config.get("max_grad_norm", 1.0)
        dp_config.total_epsilon = config.get("total_epsilon", 1.0)
        dp_config.delta = config.get("delta", 1e-5)

        dp_config.noise_multiplier = config.get("noise_multiplier", None)

        print(f" [Debug] DP Config: clip={dp_config.C}, total_epsilon={dp_config.total_epsilon}, mechanism={dp_config.mechanism}, noise_multiplier={dp_config.noise_multiplier}")

        if "round_target_epsilon" in config and "total_epsilon" not in config:
            dp_config.total_epsilon = (
                config["round_target_epsilon"]
                * config.get("privacy_calibration_rounds", 500)
            )
            print(f" [Debug] Legacy mode: round_epsilon={config['round_target_epsilon']} -> total_epsilon={dp_config.total_epsilon}")

    return dp_config


def _run_config_dict(config: dict, results_dir: str = "./results"):
    dp_config = config_to_dp_config(config)

    params = {
        "dataset_name": config.get("dataset", "mnist"),
        "num_clients": config.get("num_clients", 2),
        "num_rounds": config.get("num_rounds", 3),
        "local_epochs": config.get("local_epochs", 1),
        "batch_size": config.get("batch_size", 64),
        "client_fraction": config.get("client_fraction", 1.0),
        "seed": config.get("seed", 0),
        "use_inmemory": True,
        "dp_config": dp_config,
        "experiment_config": config,
        "non_iid": config.get("non_iid", True),  # Pass Non-IID configuration
        "alpha": config.get("alpha", 0.2),  # Pass Dirichlet alpha parameter
        "results_dir": results_dir,
    }

    return run_unified_serverless_fl(**params)


def run_single_experiment(experiment_name: str):
    import json
    import time
    import datetime

    try:
        config = get_serverless_experiment_config(experiment_name)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path("results") / f"{experiment_name}_{timestamp}"
        results_path.mkdir(parents=True, exist_ok=True)

        with open(results_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"  Start training Serverless experiments: {experiment_name}")
        print(f"   Description: {config['description']}")
        print(f"   Result list: {results_path.resolve()}")
        print("=" * 60)

        log_path = results_path / "output.log"
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file

        start_time = time.time()
        result = _run_config_dict(config, results_dir=str(results_path))
        end_time = time.time()
        duration = end_time - start_time

        log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        run_info = {
            "experiment": experiment_name,
            "description": config.get("description", ""),
            "timestamp": timestamp,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "success": result is not None,
            "final_accuracy": result.get("final_accuracy") if result else None,
            "final_loss": result.get("final_loss") if result else None,
            "result": result
        }

        with open(results_path / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        if result and 'final_accuracy' in result:
            print(f"Experiment completed! Final accuracy: {result['final_accuracy']:.4f}")
            if 'final_loss' in result:
                print(f"   Final loss: {result['final_loss']:.4f}")
            print(f"   Duration: {duration:.2f} seconds")
        else:
            print("  Experiment completed, but no accuracy information returned")

        print(f" Experiment results saved to: {results_path.resolve()}")
        print(f"   - config.json - Experiment configuration")
        print(f"   - run_info.json - Run information and results")
        print(f"   - output.log - Detailed log")

        return result

    except Exception as e:
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        if 'log_file' in locals() and not log_file.closed:
            log_file.close()

        print(f" Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()

        if 'results_path' in locals():
            error_info = {
                "experiment": experiment_name,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            with open(results_path / "error.json", "w") as f:
                json.dump(error_info, f, indent=2)
            print(f" Error information saved to: {results_path / 'error.json'}")

        return None


def run_category_experiments(category: str):
    categories = list_serverless_experiments_by_category()

    if category not in categories:
        print(f" Unknown experiment category: {category}")
        print(f"Available categories: {list(categories.keys())}")
        return

    experiments = categories[category]
    if not experiments:
        print(f"  No experiments found in category '{category}'")
        return

    print(f" Starting all experiments in category '{category}'")
    print(f" Contains {len(experiments)} experiments")
    print("=" * 60)

    results = {}
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n Experiment {i}/{len(experiments)}: {exp_name}")
        print("-" * 40)

        result = run_single_experiment(exp_name)
        results[exp_name] = result

        print("=" * 60)

    print(f"\n All experiments in category '{category}' completed!")
    print("\n Summary of results:")
    for exp_name, result in results.items():
        if result and 'final_accuracy' in result:
            print(f"  - {exp_name}: {result['final_accuracy']:.4f}")
        else:
            print(f"  - {exp_name}: Failed")


def run_comparison_experiments():
    test_experiments = [
        "serverless_fedavg_baseline_ws",
        "serverless_fedavg_privacy_ws",
        "serverless_fedavg_laplace_privacy_ws_4",
    ]

    print(" Starting quick comparison experiments")
    print(f" Contains {len(test_experiments)} experiments")
    print("=" * 60)

    results = {}
    for i, exp_name in enumerate(test_experiments, 1):
        print(f"\n Experiment {i}/{len(test_experiments)}: {exp_name}")
        print("-" * 40)

        result = run_single_experiment(exp_name)
        results[exp_name] = result

        print("=" * 60)

    print("\n Quick comparison experiments completed!")
    print("\n Summary of results:")
    for exp_name, result in results.items():
        if result and 'final_accuracy' in result:
            print(f"  - {exp_name}: {result['final_accuracy']:.4f}")
        else:
            print(f"  - {exp_name}: Failed")


def main():
    parser = argparse.ArgumentParser(description="Run serverless federated learning experiments")
    parser.add_argument("--config", type=str, required=False, help="Experiment configuration name")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--category", type=str, help="Run all experiments in a category")
    parser.add_argument("--quick", action="store_true", help="Run quick comparison experiments")

    # Support legacy positional arguments for backward compatibility
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('--'):
        # Legacy mode
        arg = sys.argv[1]

        if arg == "--list":
            print_serverless_experiments()
            return

        elif arg == "--quick":
            run_comparison_experiments()
            return

        elif arg == "--category":
            if len(sys.argv) < 3:
                print(" Please specify a category name")
                categories = list_serverless_experiments_by_category()
                print(f"Available categories: {list(categories.keys())}")
                return

            category_name = sys.argv[2]
            run_category_experiments(category_name)
            return

        elif arg.startswith("--"):
            print(f" Unknown option: {arg}")
            print("Use --help to see usage information")
            return
        else:
            experiment_name = arg
            # Use config file seed for legacy mode (no explicit override)
            run_experiment_with_config(experiment_name, seed=None)
            return

    # New argparse mode
    args = parser.parse_args()

    if args.list:
        print_serverless_experiments()
        return

    if args.quick:
        run_comparison_experiments()
        return

    if args.category:
        run_category_experiments(args.category)
        return

    if args.config:
        if args.config not in list_serverless_experiments():
            print(f"Error: Experiment '{args.config}' not found.")
            print("Available experiments:")
            print_serverless_experiments()
            sys.exit(1)

        print(f"Running experiment with config={args.config}, seed={args.seed}")
        run_experiment_with_config(args.config, seed=args.seed)
    else:
        print(" Unified Serverless PyTorch Federated Learning Experiments")
        print("=" * 60)
        print_serverless_experiments()
        print("\nUsage:")
        print("  python run_serverless_experiment.py --config <experiment_name> [--seed <seed>]")
        print("  python run_serverless_experiment.py --category <category_name>")
        print("  python run_serverless_experiment.py --quick")
        print("  python run_serverless_experiment.py --list")
        print("\nExamples:")
        print("  python run_serverless_experiment.py --config serverless_fedavg_baseline_ws --seed 2025")
        print("  python run_serverless_experiment.py --quick")


def run_experiment_with_config(experiment_name: str, seed: int = None):
    """Run experiment with optional seed override"""
    import datetime
    import json
    import traceback
    import sys
    import time
    from pathlib import Path
    from io import StringIO

    # Get base config first
    config = get_serverless_experiment_config(experiment_name).copy()  # Make a copy

    # Use provided seed if given, otherwise use config's seed, otherwise default to 2025
    if seed is None:
        seed = config.get("seed", 2025)
    else:
        config["seed"] = seed  # Override seed only if explicitly provided

    print(f"\n{'='*60}")
    print(f"Running serverless experiment: {experiment_name} (seed={seed})")
    print(f"{'='*60}")

    print(f"Description: {config.get('description', 'No description available')}")
    print(f"Seed: {seed}")

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{experiment_name}_seed{seed}" if seed != 2025 else experiment_name
    results_path = Path("results") / f"{experiment_dir}_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Prepare log files (like server framework)
    stdout_log_path = results_path / "output.log"
    stderr_log_path = results_path / "error.log"

    print(f"\n Results will be saved to: {results_path.resolve()}")
    print(f" Output log: {stdout_log_path}")

    try:
        dp_config = config_to_dp_config(config)

        # Extract Non-IID configuration
        non_iid = config.get("non_iid", True)
        data_mode = "Non-IID (PathologicalPartitioner)" if non_iid else "IID (Uniform Random)"
        print(f"\n Data Distribution: {data_mode}")
        print(f" non_iid parameter: {non_iid}")

        params = {
            "dataset_name": config.get("dataset", "mnist"),
            "num_clients": config.get("num_clients", 2),
            "num_rounds": config.get("num_rounds", 3),
            "local_epochs": config.get("local_epochs", 1),
            "batch_size": config.get("batch_size", 64),
            "client_fraction": config.get("client_fraction", 1.0),
            "seed": seed,  # Use the overridden seed
            "use_inmemory": True,
            "dp_config": dp_config,
            "experiment_config": config,
            # Pass topology parameters
            "serverless_topology": config.get("serverless_topology", "ring"),
            "ws_k": config.get("ws_k", 2),
            "ws_p": config.get("ws_p", 0.2),
            # Pass Non-IID configuration
            "non_iid": non_iid,
            "results_dir": str(results_path),
        }

        start_time = time.time()
        print(f"\nStarting experiment at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Capture stdout and stderr like server framework
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create string buffers and file objects
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        class TeeWriter:
            def __init__(self, original, buffer, file_path):
                self.original = original
                self.buffer = buffer
                self.file = open(file_path, 'w', buffering=1, encoding='utf-8')

            def write(self, text):
                self.original.write(text)  # Still show in console
                self.buffer.write(text)    # Capture in buffer
                self.file.write(text)      # Save to file
                self.file.flush()          # Ensure immediate write

            def flush(self):
                self.original.flush()
                self.file.flush()

            def close(self):
                self.file.close()

        # Redirect stdout and stderr
        stdout_tee = TeeWriter(original_stdout, stdout_buffer, stdout_log_path)
        stderr_tee = TeeWriter(original_stderr, stderr_buffer, stderr_log_path)

        sys.stdout = stdout_tee
        sys.stderr = stderr_tee

        try:
            # Run the experiment
            results = run_unified_serverless_fl(**params)
        finally:
            # Always restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            stdout_tee.close()
            stderr_tee.close()

        end_time = time.time()
        duration = end_time - start_time

        # Save run information (like server framework)
        run_info = {
            "experiment_name": experiment_name,
            "seed": seed,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "return_code": 0,
            "success": True,
            "results": results if results else "No results returned"
        }

        with open(results_path / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        print(f"\n Experiment '{experiment_name}' completed successfully in {duration:.2f} seconds")
        print(f" Results saved to: {results_path.resolve()}")
        print(f"   - config.json - Experiment configuration")
        print(f"   - output.log - Console output")
        print(f"   - error.log - Error output")
        print(f"   - run_info.json - Run information and results")

        return True

    except Exception as e:
        # Restore stdout/stderr if exception occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n Experiment failed with error:")
        print(f"   {str(e)}")

        # Save error information
        error_info = {
            "experiment_name": experiment_name,
            "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "return_code": 1,
            "success": False
        }

        with open(results_path / "error.json", "w") as f:
            json.dump(error_info, f, indent=2)

        # Also save to run_info.json for consistency
        with open(results_path / "run_info.json", "w") as f:
            json.dump(error_info, f, indent=2)

        print(f" Error information saved to: {results_path / 'error.json'}")
        print(f" Check log for details: {stdout_log_path}")
        return False


if __name__ == "__main__":
    main()
