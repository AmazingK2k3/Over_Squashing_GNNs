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

from rewire_functions import (
    rewire_Graph,
    rewire_Graph_local_bridges,
    rewire_Graph_betweenness,
    apply_rewiring_strategy)




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
                    choices=['bridges', 'betweenness', 'local_bridges'],
                    help='Rewiring strategy to use: bridges (default), betweenness, or local_bridges')
    parser.add_argument('--top-n-edges', type=int, default=2,
                    help='Number of top edges to rewire (for betweenness and local_bridges strategies)')


    return vars(parser.parse_args())

from utils.custom_data import CustomData

# class CustomData(Data):
#     def __inc__(self,key,value):
#         if key =='rewired_edge_index':
#             return self.num_nodes
#         return super().__inc__(key,value

def preprocess_dataset(dataset_path, dataset_name, use_rewired=False, rewiring_strategy='bridges', top_n=2):
    """
    Preprocess the dataset and optionally add rewired edges.
    Makes sure:
     The data object has both the edge_index(unaltered) and rewired_edge_index
     To change the rewiring(eg: rewire1 --> betweenness) only make changes to graph.py
    """
    logging.info(f"Preprocessing started for {dataset_name}")
    
    # Use TUDataset directly like the original working code
    dataset = TUDataset(root=dataset_path, name=dataset_name)
    rewired_data_list = [] #-----> saves the rewired dataset.

    for i, data in enumerate(dataset): # data is a single graph object
        logging.info(f"Processing graph {i + 1}/{len(dataset)} in dataset {dataset_name}")
        
        # data_dict = data.to_dict()
        if use_rewired:
 
            if rewiring_strategy == 'bridges': # made changes to accomodate last layer rewiring 
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph(data)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
        
            elif rewiring_strategy == 'betweenness':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph_betweenness(data, top_n=top_n)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index

         
            elif rewiring_strategy == 'local_bridges':
                original_edge_index = data.edge_index.clone()
                rewired_edge_index = rewire_Graph_local_bridges(data, top_n=top_n)
                data.rewired_edge_index = rewired_edge_index
                data.edge_index = original_edge_index
                
            else: # can it be without rewiring?
                logging.warning(f"Unknown rewiring strategy: {rewiring_strategy}. Using bridges.")
                data.edge_index = rewire_Graph(data)
            
            logging.info(f"Original edges: {data.edge_index.size(1)} | Strategy: {rewiring_strategy}")
            rewired_data_list.append(data) #-------> this is important 

    # 1st VERSION:
    #Save the dataset with a different name if rewired
    save_name = f"{dataset_name}_{rewiring_strategy}_{top_n}.pt" if use_rewired else f"{dataset_name}_processed.pt"
    torch.save(rewired_data_list, os.path.join(dataset_path, save_name))
    print(f"Dataset {dataset_name} processed & saved as {save_name} in {dataset_path}.")

    # # # # <DATA, SLICES>. VERSION
    
    # data,slices = InMemoryDataset.collate(rewired_data_list)
    # save_name = f"{dataset_name}_{rewiring_strategy}_slices.pt" if use_rewired else f"{dataset_name}_processed.pt"
    # torch.save((data,slices),os.path.join(dataset_path,save_name))



if __name__ == "__main__":
    
    args_dict = get_args_dict()
    print(args_dict)

    dataset_name = args_dict['dataset_name']
    dataset_path = args_dict['DATA_DIR']
    use_rewired = args_dict['use_rewired']
    rewiring_strategy = args_dict.get('rewiring_strategy', 'bridges')  # Use get() for safety
    top_n = args_dict.get('top_n_edges', 2)  # Use get() for safety

    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(dataset_path, name, use_rewired=use_rewired, 
                             rewiring_strategy=rewiring_strategy, top_n=top_n)
    else:
        preprocess_dataset(dataset_path, dataset_name, use_rewired=use_rewired,
                         rewiring_strategy=rewiring_strategy, top_n=top_n)

    # parser = argparse.ArgumentParser(description="Preprocess datasets.")
    # parser.add_argument("dataset_path", type=str, help="Path to the dataset folder.")
    # parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    # parser.add_argument("--use-rewired", action="store_true", help="Add rewired edges to the dataset.")
    # args = parser.parse_args()
    
    # preprocess_dataset(args.dataset_path, args.dataset_name, use_rewired=args.use_rewired)

# use it like: python PrepareDatasets.py DATA/CHEMICAL --dataset-name NCI1 --use-rewired


# ~ python PrepareDatasets.py DATA/CHEMICAL --dataset-name NCI1 --outer-k 10 --use-rewired --rewiring-strategy local_bridges --top-n-edges 5

# python Launch_Experiments.py --config-file config_fixed.yml --dataset-name NCI1 --result-folder NCI1_RESULTS_LAST --debug
