import torch

def partial_sampled_edges(x, p = 0.30): # x: node feature
    N = x.size(0) # number of nodes
    n_sample = int(p * N * N) # number of edges to sample
    perm = torch.randperm(N*N)[:n_sample]# rand smple

    nodes = torch.arange(N)
    U, V = torch.meshgrid(nodes, nodes, indexing='ij')
    full_edge_index = torch.stack([U.flatten(), V.flatten()], dim=0)


    sampled_edges = full_edge_index[:,perm]
    return sampled_edges


def partial_sample_eges(x, p = 0.3):
    N = x.size(0)
    n_sample = int(p* N *)