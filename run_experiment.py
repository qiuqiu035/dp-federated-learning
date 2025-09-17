# run_experiment.py

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from experiment_configs import (
    get_experiment_config, 
    list_available_experiments, 
    set_environment_from_config,
    print_available_experiments
)

def run_single_experiment(experiment_name: str, results_dir: str = "./results"):
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    
    config = get_experiment_config(experiment_name)
    print(f"Description: {config['description']}")
    
    results_path = Path(results_dir) / experiment_name
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    env = os.environ.copy()
    
    set_environment_from_config(config)
    
    env["EXPERIMENT_MODE"] = os.environ.get("EXPERIMENT_MODE", experiment_name)
    env["FL_EXPERIMENT_CONFIG"] = json.dumps(config)
    env["FL_RESULTS_DIR"] = str(results_path)
    
    print("\nConfiguration:")
    print(f"  Aggregation Method: {config.get('aggregation_method', 'fedavg')}")
    print(f"  Client Optimizer: {config.get('client_optimizer', 'sgd')}")
    print(f"  Use DP: {config.get('use_dp', False)}")
    print(f"  Number of Rounds: {config.get('num_rounds', 100)}")
    
    if config.get('aggregation_method') == 'krum':
        print(f"  Krum F: {config.get('krum_f', 1)}")
        print(f"  Multi-Krum: {config.get('krum_multi', False)}")
    
    if config.get('use_dp', False):
        if 'target_epsilon' in config:
            print(f"  Target Epsilon: {config['target_epsilon']}")
        if 'epsilon_per_step' in config:
            print(f"  Epsilon per Step: {config['epsilon_per_step']}")
    print()
    
    start_time = time.time()
    
    stdout_log_path = results_path / "output.log" 
    
    try:
        with open(stdout_log_path, "w") as log_file:
            process = subprocess.Popen(
                ["flwr", "run", "."],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  
                text=True,
                bufsize=1  
            )

           
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')  
                    log_file.write(line) 
            
            process.wait(timeout=1800) 
            return_code = process.poll()

        end_time = time.time()
        duration = end_time - start_time
        
        run_info = {
            "experiment_name": experiment_name,
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
        process.kill()
        return False
    
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        if 'process' in locals() and process.poll() is None:
            process.kill()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_name> [results_dir]")
        print("       python run_experiment.py --list (show all experiments)")
        print("       python run_experiment.py all (run all experiments)")
        print()
        print_available_experiments()
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    
    if experiment_name == "--list":
        print_available_experiments()
        return
    
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "./results"
    
    if experiment_name == "all":
        experiments_to_run = ["baseline","low_privacy", "medium_privacy", "high_privacy"]
        available_experiments = list_available_experiments()
        experiments = [exp for exp in experiments_to_run if exp in available_experiments]
        
        success_count = 0
        
        for exp in experiments:
            success = run_single_experiment(exp, results_dir)
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
        success = run_single_experiment(experiment_name, results_dir)
        if success:
            print(f"\nResults saved in: {Path(results_dir, experiment_name).resolve()}")

if __name__ == "__main__":
    main()