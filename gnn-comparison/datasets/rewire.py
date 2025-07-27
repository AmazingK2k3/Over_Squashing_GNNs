
import networkx as nx
import logging
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import to_networkx


def rewire_graph(g: nx.Graph) -> torch.Tensor:
    logging.info("Edge Rewiring starting for graph with %d nodes and %d edges", g.number_of_nodes(), g.number_of_edges())

    g6 = g.copy()
    bridges = list(nx.bridges(g6))
    adj_node_dict = {}

    filtered_bridges = [bridge for bridge in bridges
                        if g6.degree[bridge[0]] > 1 and g6.degree[bridge[1]] > 1]

    logging.info("Filtered down to %d bridges after degree check", len(filtered_bridges))

    for u, v in filtered_bridges:
        for node in (u, v):
            neighbors = list(nx.neighbors(g6, node))
            if len(neighbors) > 1:
                adj_node_dict[node] = [n for n in neighbors if n not in (u, v)]

    for u, v in filtered_bridges:
        for nu in adj_node_dict.get(u, []):
            if nu != v and not g6.has_edge(nu, v):
                g6.add_edge(nu, v)
        for nv in adj_node_dict.get(v, []):
            if nv != u and not g6.has_edge(nv, u):
                g6.add_edge(nv, u)

    edge_index = torch.tensor(list(g6.edges), dtype=torch.long).t().contiguous()
    logging.info("Rewiring complete. New edge count: %d", edge_index.size(1))
    return edge_index