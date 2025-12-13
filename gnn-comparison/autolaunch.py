"""
Automated experiment launcher that runs multiple experiments sequentially.
Handles result folder naming to avoid conflicts and logs all runs.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
import json
import argparse


class ExperimentLauncher:
    def __init__(self, base_result_folder='results', log_file='experiment_log.json'):
        self.base_result_folder = base_result_folder
        self.log_file = log_file
        self.experiment_log = []
        
    def generate_result_folder(self, dataset_name, config_file, suffix=''):
        """Generate unique result folder name with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        
        folder_name = f"{self.base_result_folder}/{dataset_name}_{config_name}_{timestamp}"
        if suffix:
            folder_name += f"_{suffix}"
            
        return folder_name
    
    def run_experiment(self, config_file, dataset_name, outer_folds=10, 
                  inner_folds=5, outer_processes=2, inner_processes=1,
                  debug=False, suffix='', extra_args=None, show_output=True):
        """
        Run a single experiment and return success status
        
        Args:
            config_file: Path to config YAML file
            dataset_name: Name of dataset
            outer_folds: Number of outer CV folds
            inner_folds: Number of inner CV folds  
            outer_processes: Parallel processes for outer loop
            inner_processes: Parallel processes for inner loop
            debug: Enable debug mode
            suffix: Custom suffix for result folder
            extra_args: List of additional command line arguments
        """
        result_folder = self.generate_result_folder(dataset_name, config_file, suffix)
        
        # Build command
        cmd = [
            'python', 'Launch_Experiments.py',
            '--config-file', config_file,
            '--dataset-name', dataset_name,
            '--result-folder', result_folder,
            '--outer-folds', str(outer_folds),
            '--inner-folds', str(inner_folds),
            '--outer-processes', str(outer_processes),
            '--inner-processes', str(inner_processes),
        ]
        
        if debug:
            cmd.append('--debug')
            
        if extra_args:
            cmd.extend(extra_args)
        
        # Log experiment info
        experiment_info = {
            'timestamp': datetime.now().isoformat(),
            'config_file': config_file,
            'dataset_name': dataset_name,
            'result_folder': result_folder,
            'command': ' '.join(cmd),
            'status': 'running'
        }
        
        print("=" * 80)
        print(f"Starting Experiment: {dataset_name} with {config_file}")
        print(f"Result folder: {result_folder}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80)
        
        start_time = time.time()
        
        # NEW:
        try:
            if show_output:
                # Stream output in real-time
                result = subprocess.run(cmd, check=True, stdout=None, stderr=None, text=True)
                
                elapsed_time = time.time() - start_time
                experiment_info['status'] = 'success'
                experiment_info['elapsed_time'] = elapsed_time
            else:
                # Capture output
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
                elapsed_time = time.time() - start_time
                experiment_info['status'] = 'success'
                experiment_info['elapsed_time'] = elapsed_time
                experiment_info['stdout'] = result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
            
            print(f"✓ Experiment completed successfully in {elapsed_time:.2f}s")
            print(f"Results saved to: {result_folder}")
            
            return True, experiment_info
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            experiment_info['status'] = 'failed'
            experiment_info['elapsed_time'] = elapsed_time
            experiment_info['error'] = str(e)
            experiment_info['stderr'] = e.stderr[-1000:] if e.stderr else ''
            
            print(f"✗ Experiment failed after {elapsed_time:.2f}s")
            print(f"Error: {e}")
            
            return False, experiment_info
            
        finally:
            self.experiment_log.append(experiment_info)
            self._save_log()
    
    def _save_log(self):
        """Save experiment log to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
    
    def run_experiment_suite(self, experiments, stop_on_error=False, show_output=True):       
        """
        Run a suite of experiments sequentially
        
        Args:
            experiments: List of experiment configs (dicts with run_experiment args)
            stop_on_error: If True, stop running experiments after first failure
        """
        total = len(experiments)
        successful = 0
        failed = 0
        
        print(f"\n{'='*80}")
        print(f"Starting experiment suite: {total} experiments queued")
        print(f"{'='*80}\n")
        
        for i, exp_config in enumerate(experiments, 1):
            print(f"\n[{i}/{total}] ", end='')
            
            if 'show_output' not in exp_config:
                exp_config['show_output'] = show_output
            
            success, info = self.run_experiment(**exp_config)
            
            if success:
                successful += 1
            else:
                failed += 1
                if stop_on_error:
                    print(f"\nStopping due to error (stop_on_error=True)")
                    break
            
            print()  # Spacing between experiments
        
        # Summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUITE SUMMARY")
        print("=" * 80)
        print(f"Total experiments: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Log saved to: {self.log_file}")
        print("=" * 80)
        
        return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Automated experiment launcher')
    parser.add_argument('--config', help='Path to experiment config JSON file')
    parser.add_argument('--stop-on-error', action='store_true', 
                    help='Stop running experiments after first failure')
    parser.add_argument('--silent', action='store_true',
                    help='Suppress real-time output (capture logs instead)')
    args = parser.parse_args()
    
    launcher = ExperimentLauncher(base_result_folder='results', 
                                   log_file='experiment_log.json')
    
    if args.config:
        # Load experiments from JSON config file
        with open(args.config, 'r') as f:
            experiments = json.load(f)
    else:
        # Define experiments directly in code
        experiments = [
            # GIN experiments
            # {
            #     'config_file': 'config_fixed_gin.yml',
            #     'dataset_name': 'IMDB-MULTI',
            #     'debug': True,
            #     'suffix': 'last_layer',
            # },
            # {
            #     'config_file': 'config_fixed_gin2.yml',
            #     'dataset_name': 'IMDB-MULTI',
            #     'debug': True,
            #     'suffix': 'penultimate_layer',
            # },
            # {
            #     'config_file': 'config_fixed.yml',
            #     'dataset_name': 'IMDB-MULTI',
            #     'debug': True,
            #     'suffix': 'last_layer',
            # },
            # {
            #     'config_file': 'config_fixed2.yml',
            #     'dataset_name': 'IMDB-MULTI',
            #     'debug': True,
            #     'suffix': 'penultimate_layer',
            # }
            {
                'config_file': 'config_fixed_DGCNN.yml',
                'dataset_name': 'IMDB-MULTI',
                'debug': True,
                'suffix': 'last_layer',
            },
            {
                'config_file': 'config_fixed_DGCNN2.yml',
                'dataset_name': 'IMDB-MULTI',
                'debug': True,
                'suffix': 'penultimate_layer',
            }

            
            # # DiffPool experiments
            # {
            #     'config_file': 'config_DiffPool_fixed.yml',
            #     'dataset_name': 'ENZYMES',
            #     'debug': True,
            #     'suffix': 'diffpool',
            # },
            # {
            #     'config_file': 'config_DiffPool_fixed.yml',
            #     'dataset_name': 'PROTEINS',
            #     'debug': True,
            #     'suffix': 'diffpool',
            # },
            
            # # GraphSAGE experiments
            # {
            #     'config_file': 'config_GraphSAGE.yml',
            #     'dataset_name': 'ENZYMES',
            #     'outer_folds': 10,
            #     'debug': False,
            #     'suffix': 'sage',
            # },
            
            # Add more experiments here...
        ]
    
    # Run all experiments
    launcher.run_experiment_suite(
    experiments, 
    stop_on_error=args.stop_on_error,
    show_output=not args.silent
)


if __name__ == '__main__':
    main()

#python autolaunch.py
#python autolaunch.py --config experiment_suite.json