#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import argparse
import os
import logging
from numpy import ogrid
import torch
from datasets import *
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import to_networkx
import pickle
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from Exp_2 import partial_sampled_edges
from rewire_functions import (
    rewire_Graph,
    rewire_Graph_local_bridges,
    rewire_Graph_betweenness,
    apply_rewiring_strategy,
    complement_graph)



logging.basicConfig(level = logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s' )
DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD,
  
}

def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('DATA_DIR',
                        help='where to save the datasets')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        default='all', help='dataset name [Default: \'all\']')
    parser.add_argument('--outer-k', dest='outer_k', type=int,
                        default=10, help='evaluation folds [Default: 10]')
    parser.add_argument('--inner-k', dest='inner_k', type=int,
                        default=None, help='model selection folds [Default: None]')
    parser.add_argument('--use-one', action='store_true',
                        default=False, help='use 1 as feature')
    parser.add_argument('--use-degree', dest='use_node_degree', action='store_true',
                        default=False, help='use degree as feature')
    parser.add_argument('--no-kron', dest='precompute_kron_indices', action='store_false',
                        default=True, help='don\'t precompute kron reductions')
    parser.add_argument('--use-rewired', action = 'store_true', 
                        default = False, help = 'Add rewired edges to the dataset.')
    parser.add_argument('--rewiring-strategy', type=str, default='bridges', 
                    choices=['bridges', 'betweenness', 'local_bridges','complement','partial_complement'],
                    help='Rewiring strategy to use: bridges (default), betweenness, or local_bridges')
    parser.add_argument('--top-n-edges', type=int, default=2,
                    help='Number of top edges to rewire (for betweenness and local_bridges strategies)')


    return vars(parser.parse_args())

from utils.custom_data import CustomData



def preprocess_dataset(dataset_path, dataset_name, use_rewired=False, rewiring_strategy='bridges', top_n=2, use_one=False, use_node_degree=False):
    """
    Preprocess the dataset and optionally add rewired edges.
    Makes sure:
     The data object has both the edge_index(unaltered) and rewired_edge_index
     To change the rewiring(eg: rewire1 --> betweenness) only make changes to graph.py
    """
    logging.info(f"Preprocessing started for {dataset_name}")
    if dataset_name == "ENZYMES":
        dataset = TUDataset(root=dataset_path, name=dataset_name, use_node_attr=True)
    else:
        dataset = TUDataset(root=dataset_path, name=dataset_name)
    
    rewired_data_list = []
    self_loops_flag = False

    for i, data in enumerate(dataset):
        logging.info(f"Processing graph {i + 1}/{len(dataset)} in dataset {dataset_name}")
        
        # CRITICAL FIX: Handle missing node features based on flags
        if data.x is None or data.x.numel() == 0:
            if use_one:
                # Add dummy features [1] for each node
                num_nodes = data.num_nodes if data.num_nodes is not None else 0
                data.x = torch.ones(num_nodes, 1, dtype=torch.float)
                logging.info(f"Added ones as node features with shape {data.x.shape}")
            elif use_node_degree:
                # Use node degree as features
                from torch_geometric.utils import degree
                row, _ = data.edge_index
                deg = degree(row, data.num_nodes, dtype=torch.float)
                data.x = deg.view(-1, 1)
                logging.info(f"Added degree as node features with shape {data.x.shape}")
            else:
                # Default: use ones if no features exist
                num_nodes = data.num_nodes if data.num_nodes is not None else 0
                data.x = torch.ones(num_nodes, 1, dtype=torch.float)
                logging.warning(f"No features specified but none exist. Defaulting to ones with shape {data.x.shape}")
        
        if use_rewired:
            if rewiring_strategy == 'bridges':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph(data)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
        
            elif rewiring_strategy == 'betweenness':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph_betweenness(data, top_n=top_n)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
                
            elif rewiring_strategy == 'complement':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index, self_loops_flag = complement_graph(data)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index 
     
            elif rewiring_strategy == 'local_bridges':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph_local_bridges(data, top_n=top_n)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
                
            else:
                logging.warning(f"Unknown rewiring strategy: {rewiring_strategy}. Using bridges.")
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph(data)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
            
            logging.info(f"Original edges: {data.edge_index.size(1)} | Rewired edges: {data.rewired_edge_index.size(1)} | Strategy: {rewiring_strategy}")
            rewired_data_list.append(data)
        else: 
            data.rewired_edge_index = data.edge_index.clone()
            rewired_data_list.append(data)
            logging.info(f"Original edges: {data.edge_index.size(1)} | No rewiring applied.")

    if rewired_data_list:
        sample_data = rewired_data_list[0]
        if hasattr(sample_data, 'x') and sample_data.x is not None:
            logging.info(f"Final dataset - Feature dimensions: {sample_data.x.shape[1]}")
        else:
            logging.warning("Final dataset - No node features found")
    
    os.makedirs(dataset_path, exist_ok=True)
    save_name = f"{dataset_name}_{rewiring_strategy}_{top_n}.pt" if use_rewired else f"{dataset_name}_processed.pt"
    metadata = {
        "rewiring_strategy": rewiring_strategy,
        "add_self_loops": self_loops_flag,
        "use__one": use_one,
        "use_node_degree": use_node_degree
    }
    torch.save({"data_list": rewired_data_list, "metadata": metadata},os.path.join(dataset_path, save_name)
)
    print(f"Dataset {dataset_name} processed & saved as {save_name} in {dataset_path}.")


if __name__ == "__main__":
    
    args_dict = get_args_dict()
    print(args_dict)

    dataset_name = args_dict['dataset_name']
    dataset_path = args_dict['DATA_DIR']
    use_rewired = args_dict['use_rewired']
    rewiring_strategy = args_dict.get('rewiring_strategy', 'bridges')
    top_n = args_dict.get('top_n_edges', 2)
    use_one = args_dict.get('use_one', False)  
    use_node_degree = args_dict.get('use_node_degree', False) 

    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(dataset_path, name, use_rewired=use_rewired, 
                             rewiring_strategy=rewiring_strategy, top_n=top_n,
                             use_one=use_one, use_node_degree=use_node_degree)  # FIXED: Pass the flags
    else:
        preprocess_dataset(dataset_path, dataset_name, use_rewired=use_rewired,
                         rewiring_strategy=rewiring_strategy, top_n=top_n,
                         use_one=use_one, use_node_degree=use_node_degree)  # FIXED: Pass the flags


# eg:  python PrepareDatasets.py DATA/CHEMICAL --dataset-name PROTEINS --use-rewired

# For Social datasets with no node features:
#python PrepareDatasets.py DATA/SOCIAL_1 --dataset-name <name> --use-one --outer-k 10

