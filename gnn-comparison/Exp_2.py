import torch

def partial_sampled_edges(x,previous_edges = None, p = 0.30): # x: node feature
    N = x.size(0) # number of nodes
    n_sample = int(p * N * N) # number of edges to sample
    perm = torch.randperm(N*N)[:n_sample]# rand smple

    nodes = torch.arange(N)
    U, V = torch.meshgrid(nodes, nodes, indexing='ij')
    full_edge_index = torch.stack([U.flatten(), V.flatten()], dim=0)

#####  fine till here
    # 

    if previous_edges:
        p = 0.10
        n_sample = int(p * N * N) # number of edges to sample
        perm = torch.randperm(N*N)[:n_sample]# rand smple
        # again sample edges from full_edge_index exculding the previous_edges - how?
        # temp_edges = full_edge_index - previous_edges
        # sampled_edges = temp_edges[:,perm]
    
    else:
        sampled_edges = full_edge_index[:,perm]

    return sampled_edges







