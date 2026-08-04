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
# CHANGES MADE:
# New Functions Added:
    # rewire_edge function, 
# Altered Functions:
    # altered get_edge_index function, 
import networkx as nx
import logging
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import to_networkx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Graph(nx.Graph):
    def __init__(self, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.laplacians = None
        self.v_plus = None
    
    def rewire_edges(self):
        """
        Rewire the edges of the graph using the rewire2 

        Returns:
            torch.Tensor: The rewired edge index: (2, num_edges) format.
        """

        logging.info("Edge Rewiring starting for graph with %d nodes and %d edges", self.number_of_nodes(), self.number_of_edges())

        g6 = self
        bridges = list(nx.bridges(g6))
        adj_node_dict = {}

        # Filter bridges to ensure nodes have more than one neighbor
        filtered_bridges = [bridge for bridge in bridges
                            if len(list(g6.neighbors(bridge[0]))) > 1 and
                            len(list(g6.neighbors(bridge[1]))) > 1]
        logging.info("Filtered down to %d bridges after ensuring neighbors", len(filtered_bridges))

        # Get all neighbors for each bridge node
        for u, v in filtered_bridges:
            for node in (u, v):
                adj_nodes = list(nx.all_neighbors(g6, node))
                adj_node_dict[node] = adj_nodes

        # Filter to only include nodes with more than one neighbor
        adj_node_dict = {key: value for key, value in adj_node_dict.items() if len(value) > 1}
        keys = set(adj_node_dict.keys())
        logging.info("Filtered adjacency dictionary to %d nodes with more than one neighbor", len(adj_node_dict))

        for key, values in adj_node_dict.items():
            # Remove any bridge node found in the list
            adj_node_dict[key] = [v for v in values if v not in keys]

        for u, v in filtered_bridges:
            neighbors_u = adj_node_dict.get(u, [])
            neighbors_v = adj_node_dict.get(v, [])

            # Connect neighbors of u to v
            for node_u in neighbors_u:
                if node_u != v and not g6.has_edge(node_u, v):
                    g6.add_edge(node_u, v)

            # Connect neighbors of v to u
            for node_v in neighbors_v:
                if node_v != u and not g6.has_edge(node_v, u):
                    g6.add_edge(node_v, u)

        # Convert the modified graph back to edge_index
        adj_matrix = nx.adjacency_matrix(g6).toarray()
        edge_index = torch.tensor(adj_matrix, dtype=torch.long).nonzero().t().contiguous()
        logging.info("Rewiring complete. New edge count: %d", edge_index.size(1))
        return edge_index


    def get_edge_index(self, rewire = False ):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)

        if rewire:
            edge_index = self.rewire_edges()
        return edge_index
    
    def get_original_and_rewired_edge_index(self):
        """
        Get both the original and rewired edge indices.

        Returns:
            tuple: (original_edge_index, rewired_edge_index)
        """
        original_edge_index = self.get_edge_index(rewire=False)
        rewired_edge_index = self.get_edge_index(rewire=True)
        return original_edge_index, rewired_edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False):
        features = []
        for node, node_attrs in self.nodes(data=True):
            data = []

            if node_attrs["label"] is not None:
                data.extend(node_attrs["label"])

            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_node_degree:
                data.extend([self.degree(node)])

            if use_one:
                data.extend([1])
            
            features.append(data)
        
        return torch.Tensor(features)

    def get_target(self, classification=True):
        if classification:
            return torch.LongTensor([self.target])

        return torch.Tensor([self.target])

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None
