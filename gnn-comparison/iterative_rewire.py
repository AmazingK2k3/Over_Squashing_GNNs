import argparse
import os
import torch
import logging
from Exp_2 import partial_sampled_edges, get_edge_stats
from torch_geometric.datasets import TUDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description="Iterative edge rewiring script")
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Dataset name (e.g., NCI1, ENZYMES)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--start-p', type=float, default=0.30,
                        help='Starting percentage of edges (default: 0.30)')
    parser.add_argument('--end-p', type=float, default=1.0,
                        help='Ending percentage of edges (default: 1.0)')
    parser.add_argument('--p-increment', type=float, default=0.05,
                        help='Percentage increment per iteration (default: 0.05)')
    parser.add_argument('--single-p', type=float, default=None,
                        help='Generate only a single percentage (overrides start/end/increment)')
    
    return parser.parse_args()


def generate_percentages(start_p, end_p, p_increment):
    """Generate list of percentages from start to end with given increment."""
    percentages = []
    current = start_p
    while current <= end_p + 1e-9:  # small epsilon for float comparison
        percentages.append(round(current, 2))
        current += p_increment
    return percentages


def iterative_rewiring(dataset_path, dataset_name, seed=42,
                       start_p=0.30, end_p=1.0, p_increment=0.05, single_p=None):
    """
    Generate rewired datasets with varying edge percentages.
    """
    logging.info(f"Preprocessing started for {dataset_name} at {dataset_path}")
    logging.info(f"Using seed: {seed}")

    # Load dataset
    try:
        if dataset_name == "ENZYMES":
            dataset = TUDataset(root=dataset_path, name=dataset_name, use_node_attr=True)
        else:
            dataset = TUDataset(root=dataset_path, name=dataset_name)
    except Exception as e:
        logging.error(f"Failed to load TUDataset: {e}")
        return

    data_list = [data.clone() for data in dataset]
    logging.info(f"Loaded {len(data_list)} graphs from the dataset.")

    # Determine percentages to generate
    if single_p is not None:
        percentages = [single_p]
    else:
        percentages = generate_percentages(start_p, end_p, p_increment)
    
    logging.info(f"Will generate datasets for percentages: {[int(p*100) for p in percentages]}%")

    # Create output directory
    output_dir = os.path.join(dataset_path, f"{dataset_name}_rewired_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for p in percentages:
        current_p_int = int(p * 100)
        logging.info(f"Generating rewiring for {current_p_int}% edges...")

        rewired_data_list = []

        for idx, data in enumerate(data_list):
            # Clone original data to preserve it
            new_data = data.clone()
            
            # Use (seed + idx) so each graph has different but reproducible edges
            graph_seed = seed + idx
            
            # Generate sampled edges
            # Same seed guarantees subset property automatically
            sampled_edges = partial_sampled_edges(new_data, p=p, seed=graph_seed)
            
            # Store rewired edges 
            new_data.rewired_edge_index = sampled_edges
            
            rewired_data_list.append(new_data)
            
            # Log stats for first few graphs (debugging)
            if idx < 3:
                n, max_e, actual_e, actual_pct = get_edge_stats(new_data)
                logging.debug(f"  Graph {idx}: N={n}, max_edges={max_e}, "
                             f"sampled={actual_e}, actual_pct={actual_pct:.1f}%")

        # Save rewired dataset
        save_name = f"{dataset_name}_{current_p_int}pct_seed{seed}.pt"
        save_path = os.path.join(output_dir, save_name)
        torch.save(rewired_data_list, save_path)
        logging.info(f"Saved rewired data with {current_p_int}% edges at {save_path}")

    logging.info("All percentages generated successfully!")
    return output_dir


if __name__ == "__main__":
    args = get_args()

    iterative_rewiring(
        dataset_path=args.data_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        start_p=args.start_p,
        end_p=args.end_p,
        p_increment=args.p_increment,
        single_p=args.single_p
    )


# Example usage:
# Full run (30% to 100% in 5% increments):
#   python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name NCI1 --seed 42
#
# Single percentage:
#   python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name ENZYMES --single-p 0.50 --seed 42
#
# Custom range:
#   python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name NCI1 --start-p 0.40 --end-p 0.80 --p-increment 0.10