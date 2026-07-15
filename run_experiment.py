# run_experiment.py

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from pathlib import Path
from experiment_configs import (
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    list_available_experiments,
    print_available_experiments
)

# Keep Flower logs together with the experiment output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

def main():
    parser = argparse.ArgumentParser(description="Run federated learning experiments")
    parser.add_argument("--config", type=str, required=False, help="Experiment configuration name")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")

    # Support legacy positional arguments for backward compatibility
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('--'):
        # Legacy mode: python run_experiment.py <experiment_name> [results_dir]
        experiment_name = sys.argv[1]

        if experiment_name == "--list":
            print_available_experiments()
            return

        results_dir = sys.argv[2] if len(sys.argv) > 2 else "./results"

        if experiment_name == "all":
            experiments = list_available_experiments()

            success_count = 0

            for exp in experiments:
                # Use config file seed for legacy mode (no explicit override)
                success = run_experiment_with_config(exp, results_dir, seed=None)
                if success:
                    success_count += 1
                time.sleep(5)

            print(f"\n{'='*60}")
            print(f"Completed {success_count}/{len(experiments)} experiments successfully")
            print(f"Results saved in: {Path(results_dir).resolve()}")

        else:
            if experiment_name not in list_available_experiments():
                print(f"Error: Experiment '{experiment_name}' not found.")
                sys.exit(1)
            success = run_experiment_with_config(experiment_name, results_dir, seed=None)
            if success:
                print(f"\nResults saved in: {Path(results_dir, experiment_name).resolve()}")
        return

    # New argparse mode
    args = parser.parse_args()

    if args.list:
        print_available_experiments()
        return

    if args.all:
        experiments = list_available_experiments()

        success_count = 0

        for exp in experiments:
            success = run_experiment_with_config(exp, args.results_dir, seed=args.seed)
            if success:
                success_count += 1
            time.sleep(5)

        print(f"\n{'='*60}")
        print(f"Completed {success_count}/{len(experiments)} experiments successfully")
        print(f"Results saved in: {Path(args.results_dir).resolve()}")

    elif args.config:
        if args.config not in list_available_experiments():
            print(f"Error: Experiment '{args.config}' not found.")
            print("Available experiments:")
            print_available_experiments()
            sys.exit(1)

        print(f"Running experiment with config={args.config}, seed={args.seed}")
        success = run_experiment_with_config(args.config, args.results_dir, seed=args.seed)
        if success:
            print(f"\nResults saved in: {Path(args.results_dir, args.config).resolve()}")
    else:
        print("Error: Must specify --config <name>, --all, or --list")
        print("Available experiments:")
        print_available_experiments()
        sys.exit(1)


def run_experiment_with_config(experiment_name: str, results_dir: str, seed: int = None):
    """Run experiment with optional seed override"""

    # Get base config first
    config = get_experiment_config(experiment_name).copy()  # Make a copy to avoid modifying original

    # Use provided seed if given, otherwise use config's seed, otherwise default to 2025
    if seed is None:
        seed = config.get("seed", 2025)
    else:
        config["seed"] = seed  # Override seed only if explicitly provided

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name} (seed={seed})")
    print(f"{'='*60}")

    print(f"Description: {config['description']}")
    print(f"Seed: {seed}")

    results_path = Path(results_dir) / f"{experiment_name}_seed{seed}" if seed != 2025 else Path(results_dir) / experiment_name
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    env = os.environ.copy()

    env["EXPERIMENT_MODE"] = os.environ.get("EXPERIMENT_MODE", experiment_name)
    env["FL_EXPERIMENT_CONFIG"] = json.dumps(config)
    env["FL_RESULTS_DIR"] = str(results_path)

    print("\nConfiguration:")
    print(f"  Aggregation Method: {config.get('aggregation_method', 'fedavg')}")
    print(f"  Client Optimizer: {config.get('client_optimizer', 'sgd')}")
    print(f"  Use DP: {config.get('use_dp', False)}")
    print(f"  Number of Rounds: {config.get('num_rounds', 500)}")
    print(f"  Seed: {config.get('seed', 2025)}")

    if config.get('aggregation_method') == 'krum':
        print(f"  Krum F: {config.get('krum_f', 1)}")
        print(f"  Multi-Krum: {config.get('krum_multi', False)}")

    if config.get('use_dp', False):
        if 'target_epsilon' in config:
            print(f"  Target Epsilon: {config['target_epsilon']}")
        if 'epsilon_per_step' in config:
            print(f"  Epsilon per Step: {config['epsilon_per_step']}")
        if 'total_epsilon' in config:
            print(f"  Total Epsilon: {config['total_epsilon']}")
        if 'round_target_epsilon' in config:
            print(f"  Round Target Epsilon: {config['round_target_epsilon']}")
        if 'max_grad_norm' in config:
            print(f"  Max Grad Norm: {config['max_grad_norm']}")
        if 'noise_mechanism' in config:
            print(f"  Noise Mechanism: {config['noise_mechanism']}")

    print()

    stdout_log_path = results_path / "output.log"
    stderr_log_path = results_path / "error.log"


    cmd = [
        "flwr", "run", ".",
        "--run-config", f"seed={seed}" # Pass seed to run config
    ]

    start_time = time.time()

    try:
        with open(stdout_log_path, "w", buffering=1) as stdout_file, \
             open(stderr_log_path, "w", buffering=1) as stderr_file:

            process = subprocess.run(
                cmd,
                env=env,
                cwd=os.getcwd(),
                stdout=stdout_file,
                stderr=stderr_file,
            )

            end_time = time.time()
            duration = end_time - start_time
            return_code = process.returncode

        run_info = {
            "experiment_name": experiment_name,
            "seed": seed,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "return_code": return_code,
            "success": return_code == 0
        }

        with open(results_path / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        if return_code == 0:
            print(f"\nExperiment '{experiment_name}' completed successfully in {duration:.2f} seconds")
        else:
            print(f"\nExperiment '{experiment_name}' failed with return code {return_code}")
            print(f"Check log for details: {stdout_log_path}")

        return return_code == 0

    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after 30 minutes")
        return False

    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
