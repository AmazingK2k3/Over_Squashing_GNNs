import argparse
import os
import torch
import logging
from Exp_2 import partial_sampled_edges
from torch_geometric.datasets import TUDataset

logging.basicConfig(level = logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s' )


def get_args():
    
    parser = argparse.ArgumentParser(description = "Iterative edge rewiring script")

    parser.add_argument('--data-dir', type=str, required=True,
        help = 'Path to the dataset directory')
    parser.add_argument('--dataset-name', type=str, required = True,
        help = 'Dataset name')

    return parser.parse_args()

    

def iterative_rewiring(dataset_path, dataset_name ):
    logging.info(f"Preprocessing started for {dataset_name} at {dataset_path}")

    try:
        if dataset_name == "ENZYMES":
            dataset = TUDataset(root=dataset_path, name=dataset_name, use_node_attr=True)
        else:
            dataset = TUDataset(root=dataset_path, name=dataset_name)
    except Exception as e:
        logging.error(f"Failed to load TUDataset: {e}")
        return

    data_list = [data for data in dataset]
    logging.info(f"Loaded {len(data_list)} graphs from the dataset.")


    p = 0.30
    p_increment = 0.10
    iterations = 8


    for i in range(iterations): # run till 100% edges are added
        prev_edges = False if i == 0 else True

        current_p = int((p + i * p_increment) * 100)

        logging.info(f'Iteration {i+1}: Generating rewiring for {current_p}%')

        current_iteration_list = []

        for idx, data in enumerate(data_list):
            original_edge_index = data.edge_index.clone() # save original edges

            if hasattr(data, 'rewired_edge_index'):
                data.edge_index = data.rewired_edge_index # swapping for networkx conversion

        #data.edge_index = data.rewired_edge_index # swapping for networkx conversion
            sampled_edges = partial_sampled_edges(data,previous_edges = prev_edges)

            data.rewired_edge_index = sampled_edges

            data.edge_index = original_edge_index

            current_iteration_list.append(data)

        data_list = current_iteration_list

        save_name = f"{dataset_name}_{current_p}.pt"

        save_path = os.path.join(dataset_path, save_name)

        torch.save(data_list, save_path)

        logging.info(f"Saved rewired data with {current_p}% edges at {save_path}")
  

if __name__ == "__main__":
    args = get_args()

    iterative_rewiring(args.data_dir, args.dataset_name)

# python iterative_rewire.py DATA/CHEMICAL   --dataset-name ENZYMES 

#python iterative_rewire.py --data-dir /teamspace/studios/this_studio/Over_Squashing_GNNs/gnn-comparison/DATA/CHEMICAL/ --dataset-name ENZYMES