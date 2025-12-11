import torch
from rewire_functions import complement_graph


def partial_sampled_edges(data,previous_edges = False, p = 0.30, p_increment = 0.10): # x: node feature
    N = data.x.size(0) # number of nodes
    n_sample = int(p * N * N) # number of edges to sample
    perm = torch.randperm(N*N)[:n_sample]# rand smple

    nodes = torch.arange(N)
    U, V = torch.meshgrid(nodes, nodes, indexing='ij')
    full_edge_index = torch.stack([U.flatten(), V.flatten()], dim=0)


    if previous_edges:
        

        complement_edges, _ = complement_graph(data, add_self_loops=True)


        num_comp = complement_edges.size(1)

        n_sample = int(p_increment * num_comp) # number of remaining edges to sample

        perm = torch.randperm(num_comp)[:n_sample]

        sampled_comp = complement_edges[:,perm]
        # add sampled edges to previous edges

        sampled_edges = torch.cat([data.edge_index, sampled_comp], dim=1)


    else:
        sampled_edges = full_edge_index[:,perm]

    return sampled_edges







