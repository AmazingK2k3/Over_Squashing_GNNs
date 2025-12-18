"""
Automated experiment runner for edge percentage experiments.

Workflow:
1. Finds generated .pt files from DATA/CHEMICAL/{DATASET}_rewired_seed{seed}/
2. Copies each .pt file to DATA/{DATASET}/processed/{DATASET}.pt
3. Runs Launch_Experiments.py
4. Moves results and repeats for next percentage
5. Collects all results and plots
"""

import subprocess
import os
import shutil
import json
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EdgeExperimentRunner:
    def __init__(self, data_dir='DATA', results_dir='RESULTS_EDGE_EXP', seed=42):
        """
        Args:
            data_dir: Base data directory
            results_dir: Where to store results
            seed: Random seed used for dataset generation
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed = seed
        
    def generate_percentages(self, start_p=0.30, end_p=1.0, p_increment=0.05):
        """Generate list of percentages."""
        percentages = []
        current = start_p
        while current <= end_p + 1e-9:
            percentages.append(round(current, 2))
            current += p_increment
        return percentages

    def clear_temp_results(self, result_folder, dataset_name):
        """Clear temporary results folder to prevent conflicts between runs."""
        if not os.path.exists(result_folder):
            return
            
        # Specific patterns that Launch_Experiments.py creates
        # Format: {MODEL}_{DATASET}_none_self_loops_{bool}
        patterns_to_clear = [
            f"GIN_{dataset_name}_",
            f"GRAPHSAGE_{dataset_name}_",
            f"GraphSAGE_{dataset_name}_",
            f"GCN_{dataset_name}_",
        ]
        
        for item in os.listdir(result_folder):
            item_path = os.path.join(result_folder, item)
            if os.path.isdir(item_path):
                # Check if folder matches any of the experiment output patterns
                if any(item.startswith(pattern) for pattern in patterns_to_clear):
                    logging.info(f"Clearing previous results: {item_path}")
                    shutil.rmtree(item_path)
    
    def get_rewired_files_dir(self, dataset_name):
        """
        Get directory where rewired .pt files are stored.
        Path: DATA/CHEMICAL/{dataset}_rewired_seed{seed}/
        """
        return os.path.join(self.data_dir, 'CHEMICAL', f"{dataset_name}_rewired_seed{self.seed}")
    
    def get_pt_filename(self, dataset_name, percentage):
        """
        Get the .pt filename.
        Format: {dataset}_{pct}pct_seed{seed}.pt
        """
        pct_int = int(percentage * 100)
        return f"{dataset_name}_{pct_int}pct_seed{self.seed}.pt"
    
    def get_source_pt_path(self, dataset_name, percentage):
        """Get full path to source .pt file."""
        rewired_dir = self.get_rewired_files_dir(dataset_name)
        filename = self.get_pt_filename(dataset_name, percentage)
        return os.path.join(rewired_dir, filename)
    
    def verify_pt_file_exists(self, dataset_name, percentage):
        """Check if the .pt file exists for given percentage."""
        pt_path = self.get_source_pt_path(dataset_name, percentage)
        
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Dataset not found: {pt_path}\n\n"
                f"Please generate datasets first using:\n"
                f"  python iterative_rewire.py --data-dir {self.data_dir}/CHEMICAL --dataset-name {dataset_name} --seed {self.seed}"
            )
        
        return pt_path
    
    def get_target_pt_path(self, dataset_name):
        """
        Get the target path where Launch_Experiments.py expects the data.
        Target: DATA/{DATASET}/processed/{DATASET}.pt
        """
        target_dir = os.path.join(self.data_dir, dataset_name, 'processed')
        target_file = os.path.join(target_dir, f"{dataset_name}.pt")
        return target_dir, target_file
    
    def backup_original_pt(self, dataset_name):
        """Backup the original .pt file if it exists."""
        target_dir, target_file = self.get_target_pt_path(dataset_name)
        backup_file = os.path.join(target_dir, f"{dataset_name}_original_backup.pt")
        
        if os.path.exists(target_file) and not os.path.exists(backup_file):
            shutil.copy2(target_file, backup_file)
            logging.info(f"Backed up original: {target_file} -> {backup_file}")
        
        return backup_file
    
    def setup_dataset_for_run(self, dataset_name, percentage):
        """
        Copy the .pt file for given percentage to the expected location.
        
        Source: DATA/CHEMICAL/{DATASET}_rewired_seed{seed}/{DATASET}_{pct}pct_seed{seed}.pt
        Target: DATA/{DATASET}/processed/{DATASET}.pt
        """
        pct_int = int(percentage * 100)
        
        # Verify source exists
        source_path = self.verify_pt_file_exists(dataset_name, percentage)
        
        # Get target path
        target_dir, target_file = self.get_target_pt_path(dataset_name)
        
        # Create target directory if needed
        os.makedirs(target_dir, exist_ok=True)
        
        # Backup original if exists (only first time)
        self.backup_original_pt(dataset_name)
        
        # Copy the file directly
        logging.info(f"Copying: {source_path} -> {target_file}")
        shutil.copy2(source_path, target_file)
        
        return target_file
    
    def run_launch_experiments(self, config_file, dataset_name, result_folder, debug=False):
        """
        Run Launch_Experiments.py with the given parameters.
        """
        cmd = [
            'python', 'Launch_Experiments.py',
            '--config-file', config_file,
            '--dataset-name', dataset_name,
            '--result-folder', result_folder,
        ]
        
        if debug:
            cmd.append('--debug')
        
        logging.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, text=True)
            return True, None
        except subprocess.CalledProcessError as e:
            logging.error(f"Launch_Experiments.py failed: {e}")
            return False, str(e)
    
    def extract_results(self, result_folder, dataset_name):
        """Extract all results from assessment_results.json."""
        # Find the actual model output folder (e.g., results/GIN_ENZYMES_none_self_loops_False/)
        search_paths = []
        
        if os.path.exists(result_folder):
            for item in os.listdir(result_folder):
                item_path = os.path.join(result_folder, item)
                if os.path.isdir(item_path) and dataset_name in item:
                    if any(item.startswith(prefix) for prefix in ['GIN_', 'GRAPHSAGE_', 'GraphSAGE_', 'GCN_']):
                        # Add nested CV path where assessment_results.json lives
                        search_paths.append(os.path.join(item_path, '10_NESTED_CV'))
                        search_paths.append(item_path)
        
        result_files = [
            'assessment_results.json',
            'results.json',
        ]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            for result_file in result_files:
                filepath = os.path.join(search_path, result_file)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            results = json.load(f)
                        
                        # Return all results as a dict
                        return {
                            'avg_TR_score': results.get('avg_TR_score'),
                            'std_TR_score': results.get('std_TR_score'),
                            'avg_TS_score': results.get('avg_TS_score'),
                            'std_TS_score': results.get('std_TS_score'),
                            'source_file': filepath
                        }
                                
                    except Exception as e:
                        logging.warning(f"Error reading {filepath}: {e}")
        
        logging.warning(f"Could not extract results from {result_folder}")
        return None

    def move_results(self, result_folder, dataset_name, config_file, percentage, timestamp):
        """Move/rename results folder to include percentage info."""
        pct_int = int(percentage * 100)
        
        # Extract config name without extension
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        
        # New folder name with percentage
        new_folder_name = f"{dataset_name}_{config_name}_{pct_int}pct_{timestamp}"
        new_folder_path = os.path.join(self.results_dir, new_folder_name)
        
        # Find the actual model output folder (e.g., results/GIN_ENZYMES_none_self_loops_False/)
        source_folder = None
        if os.path.exists(result_folder):
            for item in os.listdir(result_folder):
                item_path = os.path.join(result_folder, item)
                if os.path.isdir(item_path) and dataset_name in item:
                    if any(item.startswith(prefix) for prefix in ['GIN_', 'GRAPHSAGE_', 'GraphSAGE_', 'GCN_']):
                        source_folder = item_path
                        break
        
        if source_folder and os.path.exists(source_folder):
            os.makedirs(self.results_dir, exist_ok=True)
            shutil.copytree(source_folder, new_folder_path, dirs_exist_ok=True)
            logging.info(f"Copied results to: {new_folder_path}")
        else:
            logging.warning(f"No model results found in {result_folder} for {dataset_name}")
        
        return new_folder_path
    
    def run_single_experiment(self, dataset_name, config_file, percentage, 
                               result_folder='results', debug=False):
        """Run a single experiment for one dataset, config, and percentage."""
        pct_int = int(percentage * 100)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Verify config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Running: {dataset_name} | {config_file} | {pct_int}% edges")
        logging.info(f"{'='*60}")
        
        # Step 1: Clear previous results FIRST to prevent conflicts
        self.clear_temp_results(result_folder, dataset_name)
        
        # Step 2: Setup dataset (copy .pt file to expected location)
        self.setup_dataset_for_run(dataset_name, percentage)
        
        # Step 3: Run training
        success, error = self.run_launch_experiments(
            config_file=config_file,
            dataset_name=dataset_name,
            result_folder=result_folder,
            debug=debug
        )
        
        # Step 4: Extract results (now returns full dict)
        results_data = None
        if success:
            results_data = self.extract_results(result_folder, dataset_name)
        
        # Step 5: Move/rename results folder
        final_result_folder = self.move_results(
            result_folder, dataset_name, config_file, percentage, timestamp
        )
        
        return {
            'dataset': dataset_name,
            'config_file': config_file,
            'percentage': pct_int,
            'results': results_data,  # Full results dict
            'success': success,
            'error': error,
            'result_folder': final_result_folder,
            'timestamp': timestamp
        }
        
    def run_full_experiment(self, dataset_name, config_file,
                            start_p=0.30, end_p=1.0, p_increment=0.05,
                            result_folder='results', debug=False):
        """
        Run full experiment for all percentages with given config.
        """
        percentages = self.generate_percentages(start_p, end_p, p_increment)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract config name
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        
        all_results = {
            'dataset': dataset_name,
            'config_file': config_file,
            'config_name': config_name,
            'seed': self.seed,
            'timestamp': timestamp,
            'percentages': [int(p*100) for p in percentages],
            'results': {
                'percentages': [],
                'avg_TR_scores': [],
                'std_TR_scores': [],
                'avg_TS_scores': [],
                'std_TS_scores': [],
                'errors': [],
                'result_folders': []
            }
        }
        
        logging.info(f"\n{'#'*60}")
        logging.info(f"Starting experiments: {dataset_name} with {config_file}")
        logging.info(f"Percentages: {[int(p*100) for p in percentages]}%")
        logging.info(f"{'#'*60}")
        
        for p in percentages:
            result = self.run_single_experiment(
                dataset_name=dataset_name,
                config_file=config_file,
                percentage=p,
                result_folder=result_folder,
                debug=debug
            )
            
            all_results['results']['percentages'].append(result['percentage'])
            all_results['results']['errors'].append(result['error'])
            all_results['results']['result_folders'].append(result['result_folder'])
            
            # Extract all metrics from results
            if result['results']:
                all_results['results']['avg_TR_scores'].append(result['results'].get('avg_TR_score'))
                all_results['results']['std_TR_scores'].append(result['results'].get('std_TR_score'))
                all_results['results']['avg_TS_scores'].append(result['results'].get('avg_TS_score'))
                all_results['results']['std_TS_scores'].append(result['results'].get('std_TS_score'))
            else:
                all_results['results']['avg_TR_scores'].append(None)
                all_results['results']['std_TR_scores'].append(None)
                all_results['results']['avg_TS_scores'].append(None)
                all_results['results']['std_TS_scores'].append(None)
            
            # Log progress
            status = "✓" if result['success'] else "✗"
            if result['results'] and result['results'].get('avg_TS_score'):
                acc_str = f"{result['results']['avg_TS_score']:.2f} ± {result['results'].get('std_TS_score', 0):.2f}"
            else:
                acc_str = "N/A"
            logging.info(f"{status} {config_name} @ {result['percentage']}%: test_acc = {acc_str}")
            
            # Save intermediate results (in case of crash)
            self._save_results(all_results, dataset_name, config_name, timestamp)
        
        # Save final results
        results_file = self._save_results(all_results, dataset_name, config_name, timestamp)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"All experiments complete!")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"{'='*60}")
        
        return all_results, results_file
    
    def _save_results(self, results, dataset_name, config_name, timestamp):
        """Save results to JSON file."""
        os.makedirs(self.results_dir, exist_ok=True)
        results_file = os.path.join(
            self.results_dir, 
            f"{dataset_name}_{config_name}_{timestamp}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_file
    
    def plot_results(self, results_file_or_dict):
        """Plot accuracy vs percentage with error bars."""
        import matplotlib.pyplot as plt
        
        if isinstance(results_file_or_dict, str):
            with open(results_file_or_dict, 'r') as f:
                results = json.load(f)
            results_file = results_file_or_dict
        else:
            results = results_file_or_dict
            results_file = None
        
        plt.figure(figsize=(10, 6))
        
        # New format with all metrics
        percentages = results['results']['percentages']
        
        # Handle both old (accuracies) and new (avg_TS_scores) format
        if 'avg_TS_scores' in results['results']:
            accuracies = results['results']['avg_TS_scores']
            std_devs = results['results'].get('std_TS_scores', [None] * len(accuracies))
        else:
            accuracies = results['results'].get('accuracies', [])
            std_devs = [None] * len(accuracies)
        
        config_name = results.get('config_name', 'config')
        
        # Filter out None values
        valid_data = [(p, a, s) for p, a, s in zip(percentages, accuracies, std_devs) if a is not None]
        
        if valid_data:
            p_vals, a_vals, s_vals = zip(*valid_data)
            
            # Plot with error bars if std_dev available
            if any(s is not None for s in s_vals):
                s_vals_clean = [s if s is not None else 0 for s in s_vals]
                plt.errorbar(p_vals, a_vals, yerr=s_vals_clean,
                            marker='o', label=config_name, 
                            linewidth=2, markersize=8, capsize=5)
            else:
                plt.plot(p_vals, a_vals, 
                        marker='o', label=config_name, 
                        linewidth=2, markersize=8)
        
        plt.xlabel('Edge Percentage (%)', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title(f"Test Accuracy vs Edge Percentage - {results['dataset']}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(results['percentages'])
        plt.tight_layout()
        
        # Save plot
        if results_file:
            plot_file = results_file.replace('.json', '.png')
        else:
            plot_file = os.path.join(self.results_dir, f"{results['dataset']}_plot.png")
        
        plt.savefig(plot_file, dpi=150)
        logging.info(f"Plot saved to: {plot_file}")
        plt.show()
        
        return plot_file




def main():
    parser = argparse.ArgumentParser(description='Run edge percentage experiments')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (NCI1 or ENZYMES)')
    parser.add_argument('--config-file', type=str, required=True,
                        help='Config file to use (e.g., config_fixed_gin.yml)')
    
    # Optional arguments
    parser.add_argument('--percentage', type=float, default=None,
                        help='Single percentage to run (0.0-1.0). If not specified, runs all.')
    parser.add_argument('--start-p', type=float, default=0.30,
                        help='Starting percentage (default: 0.30)')
    parser.add_argument('--end-p', type=float, default=1.0,
                        help='Ending percentage (default: 1.0)')
    parser.add_argument('--p-increment', type=float, default=0.05,
                        help='Percentage increment (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used for dataset generation (default: 42)')
    parser.add_argument('--data-dir', type=str, default='DATA',
                        help='Data directory (default: DATA)')
    parser.add_argument('--results-dir', type=str, default='RESULTS_EDGE_EXP',
                        help='Results directory (default: RESULTS_EDGE_EXP)')
    parser.add_argument('--result-folder', type=str, default='results',
                        help='Temp result folder for Launch_Experiments.py (default: results)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Path to results JSON to plot (skip training)')
    
    args = parser.parse_args()
    
    runner = EdgeExperimentRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        seed=args.seed
    )
    
    # Plot only mode
    if args.plot_only:
        runner.plot_results(args.plot_only)
        return
    
    # Single percentage mode
    if args.percentage is not None:
        result = runner.run_single_experiment(
            dataset_name=args.dataset,
            config_file=args.config_file,
            percentage=args.percentage,
            result_folder=args.result_folder,
            debug=args.debug
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        # Full experiment mode
        results, results_file = runner.run_full_experiment(
            dataset_name=args.dataset,
            config_file=args.config_file,
            start_p=args.start_p,
            end_p=args.end_p,
            p_increment=args.p_increment,
            result_folder=args.result_folder,
            debug=args.debug
        )
        
        # Plot results
        runner.plot_results(results_file)


if __name__ == '__main__':
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# STEP 1: Generate datasets first (run once per dataset)
# -------------------------------------------------------
#   python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name NCI1 --seed 42
#   python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name ENZYMES --seed 42
#
#
# STEP 2: Run experiments
# -------------------------------------------------------
#
# Full experiment (all percentages):
#   python iterative_runs.py --dataset ENZYMES --config-file config_fixed_gin.yml --debug
#   python iterative_runs.py --dataset NCI1 --config-file config_fixed_graphsage.yml --debug
#
# Single percentage test:
#   python iterative_runs.py --dataset ENZYMES --config-file config_fixed_gin.yml --percentage 0.30 --debug
#
# Custom range (test with 3 percentages):
#   python iterative_runs.py --dataset ENZYMES --config-file config_fixed_gin.yml --start-p 0.30 --end-p 0.40 --p-increment 0.05 --debug
#
# Plot existing results:
#   python iterative_runs.py --dataset ENZYMES --config-file config_fixed_gin.yml --plot-only RESULTS_EDGE_EXP/ENZYMES_config_fixed_gin_20241201_120000.json
#
#
# FOLDER STRUCTURE
# -------------------------------------------------------
#
# Generated datasets (from iterative_rewire.py):
#   DATA/CHEMICAL/ENZYMES_rewired_seed42/
#   ├── ENZYMES_30pct_seed42.pt
#   ├── ENZYMES_35pct_seed42.pt
#   ├── ENZYMES_40pct_seed42.pt
#   └── ...
#
# This script copies to (before each run):
#   DATA/ENZYMES/processed/ENZYMES.pt
#
# Results are saved to:
#   RESULTS_EDGE_EXP/
#   ├── ENZYMES_config_fixed_gin_30pct_20241201_120000/
#   ├── ENZYMES_config_fixed_gin_35pct_20241201_120005/
#   ├── ENZYMES_config_fixed_gin_20241201_120000.json
#   └── ENZYMES_config_fixed_gin_20241201_120000.png