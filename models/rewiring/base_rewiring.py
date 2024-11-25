# base_rewiring.py

import torch
import networkx as nx
from abc import ABC, abstractmethod
from torch_geometric.utils import to_networkx
from typing import Optional, Tuple, Dict, Any

class BaseRewiring(ABC):
    """Abstract base class for graph rewiring strategies."""
    
    def __init__(self, **kwargs):
        """
        Initialize rewiring strategy with optional parameters.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        self.params = kwargs
        
    @abstractmethod
    def rewire(self, data, device: str) -> torch.Tensor:
        """
        Rewire the graph and return new edge indices.
        
        Args:
            data: PyG Data object containing the graph
            device: Target device for the output tensor
            
        Returns:
            torch.Tensor: Rewired edge_index tensor
        """
        pass
    
    def _to_undirected_networkx(self, data) -> nx.Graph:
        """Convert PyG data to undirected NetworkX graph."""
        return to_networkx(data, to_undirected=True)
    
    def _create_edge_index(self, graph: nx.Graph, device: str) -> torch.Tensor:
        """
        Create edge_index tensor from NetworkX graph.
        
        Args:
            graph: NetworkX graph object
            device: Target device for the tensor
            
        Returns:
            torch.Tensor: Edge index tensor in PyG format
        """
        # Get edges as a list of tuples
        edges = list(graph.edges())
        
        # Convert to tensor and transpose to get (2, num_edges) format
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Make undirected by adding reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Remove duplicate edges and sort
        edge_index = torch.unique(edge_index, dim=1)
        
        # Ensure contiguous memory layout
        return edge_index.contiguous().to(device)
        
    def _get_bridges(self, graph: nx.Graph) -> list:
        """Get bridge edges from graph."""
        return list(nx.bridges(graph))
    
    def _get_filtered_bridges(self, graph: nx.Graph) -> list:
        """Get bridges where both nodes have degree > 1."""
        bridges = self._get_bridges(graph)
        return [bridge for bridge in bridges 
                if len(list(graph.neighbors(bridge[0]))) > 1 
                and len(list(graph.neighbors(bridge[1]))) > 1]
    
    def _get_adjacent_nodes(self, graph: nx.Graph, node) -> list:
        """Get all adjacent nodes for a given node."""
        return list(nx.all_neighbors(graph, node))


# # Create rewiring strategy
# rewiring = CombinedRewiring()

# # Apply rewiring to a graph
# new_edge_index = rewiring.rewire(data, device='cuda')

# # Use in model
# # For last layer:
# x = model(x, data.edge_index)
# x = final_layer(x, new_edge_index)

# # Or for full model:
# x = model(x, new_edge_index)