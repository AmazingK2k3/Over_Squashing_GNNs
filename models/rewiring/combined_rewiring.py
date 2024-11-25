# combined_rewiring.py

import torch
import networkx as nx
from typing import Dict, Any

from .base_rewiring import BaseRewiring

class CombinedRewiring(BaseRewiring):
    """
    Implementation of rewire_combined strategy that combines both 
    adjacent node connections and bridge endpoint connections.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def rewire(self, data, device: str) -> torch.Tensor:
        """
        Implement combined rewiring strategy.
        
        Args:
            data: PyG Data object
            device: Target device for the output tensor
            
        Returns:
            torch.Tensor: Rewired edge_index tensor
        """
        # Convert to NetworkX graph
        graph = self._to_undirected_networkx(data)
        
        # Get filtered bridges
        filtered_bridges = self._get_filtered_bridges(graph)
        
        # Get and process adjacent nodes
        adj_node_dict = self._get_processed_adjacency_dict(graph, filtered_bridges)
        
        # Perform combined rewiring
        self._rewire_combined(graph, filtered_bridges, adj_node_dict)
        
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
        
        # Process the dictionary to exclude bridge nodes from neighbor lists
        adj_node_dict = {
            key: [v for v in value 
                  if len(value) > 1 and v not in adj_node_dict]
            for key, value in adj_node_dict.items()
        }
        
        return adj_node_dict
    
    def _rewire_combined(self, graph: nx.Graph, bridges: list, 
                        adj_node_dict: Dict) -> None:
        """Perform combined rewiring strategy."""
        for u, v in bridges:
            neighbors_u = adj_node_dict.get(u, [])
            neighbors_v = adj_node_dict.get(v, [])
            
            # Connect neighbors to bridge endpoints
            for node_u in neighbors_u:
                if node_u != v and not graph.has_edge(node_u, v):
                    graph.add_edge(node_u, v)
                    
            for node_v in neighbors_v:
                if node_v != u and not graph.has_edge(node_v, u):
                    graph.add_edge(node_v, u)
            
            # Connect neighbors to each other
            for node_u in neighbors_u:
                for node_v in neighbors_v:
                    if node_u != node_v and not graph.has_edge(node_u, node_v):
                        graph.add_edge(node_u, node_v)