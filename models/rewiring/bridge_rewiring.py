# bridge_rewiring.py

import torch
import networkx as nx
from typing import Dict, Any, Set

from .base_rewiring import BaseRewiring

class BridgeRewiring(BaseRewiring):
    """
    Implementation of rewire2 strategy that connects nodes to bridge endpoints
    without connecting neighbors to each other.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def rewire(self, data, device: str) -> torch.Tensor:
        """
        Implement rewire2 strategy: connect nodes to bridge endpoints.
        
        Args:
            data: PyG Data object
            device: Target device for the output tensor
            
        Returns:
            torch.Tensor: Rewired edge_index tensor
        """
        # Convert to NetworkX graph
        graph = self._to_undirected_networkx(data)
        
        # Get filtered bridges (both endpoints have degree > 1)
        filtered_bridges = self._get_filtered_bridges(graph)
        
        # Get and process adjacent nodes
        adj_node_dict = self._get_processed_adjacency_dict(graph, filtered_bridges)
        
        # Perform rewiring
        self._rewire_bridge_endpoints(graph, filtered_bridges, adj_node_dict)
        
        # Convert back to edge_index tensor
        return self._create_edge_index(graph, device)
    
    def _get_processed_adjacency_dict(self, graph: nx.Graph, bridges: list) -> Dict:
        """Create processed dictionary of adjacent nodes for bridge nodes."""
        adj_node_dict = {}
        
        # Get all neighbors for each bridge node
        for u, v in bridges:
            for node in (u, v):
                adj_nodes = self._get_adjacent_nodes(graph, node)
                adj_node_dict[node] = adj_nodes
                
        # Filter to only include nodes with more than one neighbor
        adj_node_dict = {key: value for key, value in adj_node_dict.items() 
                        if len(value) > 1}
        
        # Get set of bridge nodes
        bridge_nodes = set(adj_node_dict.keys())
        
        # Remove bridge nodes from adjacency lists
        for key in adj_node_dict:
            adj_node_dict[key] = [v for v in adj_node_dict[key] 
                                 if v not in bridge_nodes]
        
        return adj_node_dict
    
    def _rewire_bridge_endpoints(self, graph: nx.Graph, bridges: list, 
                               adj_node_dict: Dict) -> None:
        """Connect nodes to bridge endpoints."""
        for u, v in bridges:
            neighbors_u = adj_node_dict.get(u, [])
            neighbors_v = adj_node_dict.get(v, [])
            
            # Connect neighbors of u to v
            for node_u in neighbors_u:
                if node_u != v and not graph.has_edge(node_u, v):
                    graph.add_edge(node_u, v)
            
            # Connect neighbors of v to u
            for node_v in neighbors_v:
                if node_v != u and not graph.has_edge(node_v, u):
                    graph.add_edge(node_v, u)