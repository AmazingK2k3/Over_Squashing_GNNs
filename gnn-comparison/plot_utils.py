import json
import os
import argparse
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_plot_data(json_file):
    """Extracts and cleans data from the experiment JSON."""
    if not os.path.exists(json_file):
        logging.warning(f"File not found: {json_file}")
        return None

    with open(json_file, 'r') as f:
        data = json.load(f)

    res = data.get('results', {})
    percentages = res.get('percentages', [])
    
    # Try the new key format first, fallback to 'accuracies'
    if 'avg_TS_scores' in res:
        accuracies = res['avg_TS_scores']
        std_devs = res.get('std_TS_scores', [0] * len(accuracies))
    else:
        accuracies = res.get('accuracies', [])
        std_devs = [0] * len(accuracies)

    # Filter out None values (failed runs)
    valid_data = [(p, a, s) for p, a, s in zip(percentages, accuracies, std_devs) if a is not None]
    
    if not valid_data:
        logging.warning(f"No valid data points found in {json_file}")
        return None

    p_vals, a_vals, s_vals = zip(*valid_data)
    
    return {
        'x': p_vals,
        'y': a_vals,
        'yerr': [s if s is not None else 0 for s in s_vals],
        'label': data.get('config_name', os.path.basename(json_file).replace('.json', '')),
        'dataset': data.get('dataset', 'Unknown'),
        'all_xticks': percentages
    }

def plot_comparison(json_files, output_path):
    """Generates the combined plot."""
    plt.figure(figsize=(12, 7))
    
    main_dataset = "Unknown"
    all_p_points = set()

    for file_path in json_files:
        data = extract_plot_data(file_path)
        if data:
            plt.errorbar(data['x'], data['y'], yerr=data['yerr'],
                         marker='o', label=data['label'], 
                         linewidth=2, markersize=6, capsize=4, alpha=0.8)
            main_dataset = data['dataset']
            all_p_points.update(data['all_xticks'])

    plt.xlabel('Edge Percentage (%)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f"Model Performance Comparison - {main_dataset}", fontsize=14)
    
    # Place legend outside the plot for better visibility
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(sorted(list(all_p_points)))
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logging.info(f"Comparison plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multiple experiment results together.")
    parser.add_argument('files', nargs='+', help="Path to one or more JSON result files.")
    parser.add_argument('--output', type=str, default="combined_plot.png", 
                        help="Filename for the output image.")
    
    args = parser.parse_args()
    plot_comparison(args.files, args.output)



#. python plot_utils.py /teamspace/studios/this_studio/Over_Squashing_GNNs/gnn-comparison/RESULTS_EDGE_EXP_gin_enzymes/ENZYMES_config_fixed_gin_20251229_025749.json /teamspace/studios/this_studio/Over_Squashing_GNNs/gnn-comparison/RESULTS_EDGE_EXP_sage_enzymes/ENZYMES_config_fixed_20251227_060505.json /teamspace/studios/this_studio/Over_Squashing_GNNs/gnn-comparison/RESULTS_EDGE_EXP_sage_enzymes_penu/ENZYMES_config_fixed_20260105_160023.json