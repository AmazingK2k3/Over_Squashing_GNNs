import torch


def partial_sampled_edges(data, p=0.30, seed=42):
    """
    Generate edges for a graph starting from empty graph.
    
    Args:
        data: PyG data object (used only for number of nodes)
        p: percentage of max possible edges to sample (0.0 to 1.0)
        seed: random seed for reproducibility
    
    Returns:
        edge_index: [2, num_edges] tensor with sampled edges + self-loops (bidirectional)
    """
    torch.manual_seed(seed)
    
    N = data.x.size(0)  # number of nodes
    
    # Max possible undirected edges (no self-loops): n(n-1)/2
    max_edges = (N * (N - 1)) // 2
    
    # Number of edges to sample based on percentage
    n_sample = int(p * max_edges)
    n_sample = max(1, min(n_sample, max_edges))  # clamp to valid range
    
    # Generate all possible undirected edges (i < j to avoid duplicates)
    rows, cols = torch.triu_indices(N, N, offset=1)
    all_possible_edges = torch.stack([rows, cols], dim=0)  # [2, max_edges]
    
    # Randomly sample n_sample edges
    perm = torch.randperm(max_edges)[:n_sample]
    sampled_edges = all_possible_edges[:, perm]
    
    # Make bidirectional (PyG convention for undirected graphs)
    sampled_edges_bidirectional = torch.cat([
        sampled_edges,
        sampled_edges.flip(0)  # reverse direction
    ], dim=1)
    
    # Add self-loops
    # self_loops = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    
    # # Combine: sampled edges (bidirectional) + self-loops
    # edge_index = torch.cat([sampled_edges_bidirectional, self_loops], dim=1)

    edge_index = sampled_edges_bidirectional
    
    return edge_index


def get_edge_stats(data):
    """Helper to print edge statistics for debugging."""
    N = data.x.size(0)
    max_edges = (N * (N - 1)) // 2
    if hasattr(data, 'rewired_edge_index'):
        # Count unique undirected edges (excluding self-loops)
        ei = data.rewired_edge_index
        mask = ei[0] < ei[1]  # only count one direction
        unique_edges = mask.sum().item()
        pct = (unique_edges / max_edges) * 100 if max_edges > 0 else 0
        return N, max_edges, unique_edges, pct
    return N, max_edges, 0, 0.0