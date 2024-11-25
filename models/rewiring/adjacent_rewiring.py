# adjacent_rewiring.py

import torch
import networkx as nx
from typing import Dict, Any

from .base_rewiring import BaseRewiring

class AdjacentRewiring(BaseRewiring):
    """
    Implementation of rewire1 strategy that connects all adjacent nodes 
    of bridges together without triangular rewiring.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def rewire(self, data, device: str) -> torch.Tensor:
        """
        Implement rewire1 strategy: connect adjacent nodes of bridge nodes.
        
        Args:
            data: PyG Data object
            device: Target device for the output tensor
            
        Returns:
            torch.Tensor: Rewired edge_index tensor
        """
        # Convert to NetworkX graph
        graph = self._to_undirected_networkx(data)
        
        # Get bridges
        bridges = self._get_bridges(graph)
        
        # Get adjacent nodes for bridge nodes
        adj_node_dict = self._get_bridge_adjacency_dict(graph, bridges)
        
        # Perform rewiring
        self._rewire_adjacent_nodes(graph, bridges, adj_node_dict)
        
        # Convert back to edge_index tensor
        return self._create_edge_index(graph, device)
    
    def _get_bridge_adjacency_dict(self, graph: nx.Graph, bridges: list) -> Dict:
        """Create dictionary of adjacent nodes for bridge nodes."""
        adj_node_dict = {}
        
        # Get all neighbors for each bridge node
        for u, v in bridges:
            for node in (u, v):
                adj_nodes = self._get_adjacent_nodes(graph, node)
                adj_node_dict[node] = adj_nodes
                
        # Filter to only include nodes with more than one neighbor
        return {key: value for key, value in adj_node_dict.items() 
                if len(value) > 1}
    
    def _rewire_adjacent_nodes(self, graph: nx.Graph, bridges: list, 
                             adj_node_dict: Dict) -> None:
        """Connect adjacent nodes of bridge nodes."""
        for u, v in bridges:
            neighbors_u = adj_node_dict.get(u, [])
            neighbors_v = adj_node_dict.get(v, [])
            
            # Connect neighbors of u to neighbors of v
            for node_u in neighbors_u:
                for node_v in neighbors_v:
                    if not graph.has_edge(node_u, node_v):
                        graph.add_edge(node_u, node_v)