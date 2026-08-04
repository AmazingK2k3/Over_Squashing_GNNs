import argparse
import os
import torch
import logging
#from Exp_2 import partial_sampled_edges
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected
from rewire_functions import complement_graph

logging.basicConfig(level = logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s' )


def get_args():
    
    parser = argparse.ArgumentParser(description = "Iterative edge rewiring script")

    parser.add_argument('--data-dir', type=str, required=True,
        help = 'Path to the dataset directory')
    parser.add_argument('--dataset-name', type=str, required = True,
        help = 'Dataset name')

    return parser.parse_args()

def partial_sampled_edges(data, previous_edges=False, target_p=0.30, final_self_loops=False): 
    N = data.x.size(0)
    device = data.edge_index.device
    
  #Cap Pairs (N * (N-1) / 2)
    total_possible_edges = int((N * (N - 1)) / 2)
    target_count = int(target_p * total_possible_edges)

    
    if hasattr(data, 'rewired_edge_index'):
        current_count = data.rewired_edge_index.size(1)
    else:
        current_count = 0

    needed = target_count - current_count
    
    # If we don't need edges, return what we have
    if needed <= 0:
        if hasattr(data, 'rewired_edge_index'):
             return data.rewired_edge_index
        else:
             return torch.empty((2,0), dtype=torch.long, device=device)

  
    if previous_edges:
        # Get unique pairs from complement
        comp_edges, _ = complement_graph(data, add_self_loops=False)
        comp_edges = comp_edges.to(device)
        num_candidates = comp_edges.size(1)
        candidate_index = comp_edges
    else:
        # Generate all unique pairs (Upper Triangle)
        nodes = torch.arange(N, device=device)
        U, V = torch.meshgrid(nodes, nodes, indexing='ij')
        # Keep only U < V (Upper Triangle)
        mask = U < V
        candidate_index = torch.stack([U[mask], V[mask]], dim=0)
        num_candidates = candidate_index.size(1)

    #
    n_sample = min(needed, num_candidates)
    
    # 
    if n_sample == 0 and needed > 0 and num_candidates > 0:
        n_sample = 1

    if n_sample > 0:
        perm = torch.randperm(num_candidates, device=device)[:n_sample]
        new_edges = candidate_index[:, perm]
        
        if hasattr(data, 'rewired_edge_index'):
            final_edges = torch.cat([data.rewired_edge_index, new_edges], dim=1)
        else:
            final_edges = new_edges
    else:
        final_edges = data.rewired_edge_index if hasattr(data, 'rewired_edge_index') else torch.empty((2,0), dtype=torch.long, device=device)

    return final_edges


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


    p_start = 0.30
    p_increment = 0.10
    iterations = 8


    for i in range(iterations): # run till 100% edges are added
        prev_edges = False if i == 0 else True
        
        current_target_p = p_start + (i * p_increment)
        curr_save_name = current_target_p

        if current_target_p > 0.99: current_target_p = 1.0


        logging.info(f'Iteration {i+1}: Generating rewiring for {current_target_p}%')

        current_iteration_list = []

        for idx, data in enumerate(data_list):
            original_edge_index = data.edge_index.clone() # save original edges

            if hasattr(data, 'rewired_edge_index'):
                data.edge_index = data.rewired_edge_index # swapping for networkx conversion

        #data.edge_index = data.rewired_edge_index # swapping for networkx conversion
            sampled_edges = partial_sampled_edges(data,target_p = current_target_p, previous_edges = prev_edges)

            data.rewired_edge_index = sampled_edges

            data.edge_index = original_edge_index

            current_iteration_list.append(data)

        data_list = current_iteration_list

        save_list = []
        for data in data_list:
            data_save = data.clone()

            if hasattr(data_save, 'rewired_edge_index'):
                data_save.rewired_edge_index = to_undirected(data_save.rewired_edge_index)
            save_list.append(data_save)

        save_name = f"{dataset_name}_{int(curr_save_name*100)}.pt"

        save_path = os.path.join(dataset_path, save_name)

        torch.save(save_list, save_path)

        logging.info(f"Saved rewired data with {current_target_p}% edges at {save_path}")
  

if __name__ == "__main__":
    args = get_args()

    iterative_rewiring(args.data_dir, args.dataset_name)

# python iterative_rewire.py DATA/CHEMICAL   --dataset-name ENZYMES 

#python iterative_rewiring_2.py --data-dir $PROJECT_ROOT/DATA/CHEMICAL/ --dataset-name ENZYMES