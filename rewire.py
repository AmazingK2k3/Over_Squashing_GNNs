import networkx as nx
import torch_geometric
import torch
from torch_geometric.utils import to_networkx

def rewire1(data,device): # this connects all the adjacent nodes of the bridges together(no triangular rewiring)
  g6 = to_networkx(data, to_undirected = True)
  bridges = list(nx.bridges(g6))

  adj_node_dict = {}
# Get all neighbors for each bridge node
  for u, v in bridges:
      for node in (u, v):
          adj_nodes = list(nx.all_neighbors(g6, node))
          adj_node_dict[node] = adj_nodes

  adj_node_dict = {key: value for key, value in adj_node_dict.items() if len(value) > 1}


 # Rewire nodes by connecting neighbors of each bridge node
  for u, v in bridges:
    neighbors_u = adj_node_dict.get(u,[])
    neighbors_v = adj_node_dict.get(v,[])

    for node_u in neighbors_u:
      for node_v in neighbors_v:
        if not g6.has_edge(node_u, node_v): # if the edge does not already exist
          g6.add_edge(node_u, node_v) # add the edge

      # Create edge_index tensor directly from edge
    edges = list(g6.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Make undirected by adding reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Remove duplicate edges and sort
    edge_index = torch.unique(edge_index, dim=1)

    # Ensure contiguous memory layout
    edge_index = edge_index.contiguous()

  return edge_index.to(device)



def rewire2(data,device):
    g6 = to_networkx(data, to_undirected=True)
    bridges = list(nx.bridges(g6))
    adj_node_dict = {}
    #print(bridges)

    filtered_bridges = [bridge for bridge in bridges
                        if len(list(g6.neighbors(bridge[0]))) > 1 and
                        len(list(g6.neighbors(bridge[1]))) > 1]
    #print(filtered_bridges)

    # Get all neighbors for each bridge node
    for u, v in filtered_bridges:
        for node in (u, v):
            adj_nodes = list(nx.all_neighbors(g6, node))
            adj_node_dict[node] = adj_nodes

    # Filter to only include nodes with more than one neighbor
    adj_node_dict = {key: value for key, value in adj_node_dict.items() if len(value) > 1}
    keys = set(adj_node_dict.keys())

    for key, values in adj_node_dict.items():
      # Remove any bridge node found in the list
      adj_node_dict[key] = [v for v in values if v not in keys]

    for u, v in filtered_bridges: # 17,18
        neighbors_u = adj_node_dict.get(u, []) # 18, 19
        neighbors_v = adj_node_dict.get(v, []) #
        #print(f"Bridge ({u}, {v}): Neighbors of {u}: {neighbors_u}, Neighbors of {v}: {neighbors_v}")
        # Only connect each neighbor of u to v, without connecting neighbors to each other
        for node_u in neighbors_u:
            if node_u != v and not g6.has_edge(node_u, v):  # Ensure we only connect to v

                g6.add_edge(node_u, v)

        # Only connect each neighbor of v to u, without connecting neighbors to each other
        for node_v in neighbors_v:
            if node_v != u and not g6.has_edge(node_v, u):  # Ensure we only connect to u
                g6.add_edge(node_v, u)

    adj_matrix = nx.adjacency_matrix(g6).toarray()
    # function should return the edge_index

    edge_index = torch.tensor(adj_matrix, dtype=torch.long)


    edge_index = edge_index.nonzero().t().contiguous() # (2,num_edges) format and contigous memory.



    return edge_index.to(device)


def rewire_combined(data,device):
    g = to_networkx(data, to_undirected=True)
    g6 = g.copy()


    bridges = list(nx.bridges(g6))
    adj_node_dict = {}


    filtered_bridges = [bridge for bridge in bridges
                        if len(list(g6.neighbors(bridge[0]))) > 1 and
                        len(list(g6.neighbors(bridge[1]))) > 1]


    for u, v in filtered_bridges:
        for node in (u, v):
            adj_nodes = list(nx.all_neighbors(g6, node))
            adj_node_dict[node] = adj_nodes


    adj_node_dict = {key: [v for v in value if len(value) > 1 and v not in adj_node_dict]
                     for key, value in adj_node_dict.items()}


    for u, v in filtered_bridges:
        neighbors_u = adj_node_dict.get(u, [])
        neighbors_v = adj_node_dict.get(v, [])


        for node_u in neighbors_u:
            if node_u != v and not g6.has_edge(node_u, v):
                g6.add_edge(node_u, v)
        for node_v in neighbors_v:
            if node_v != u and not g6.has_edge(node_v, u):
                g6.add_edge(node_v, u)


        for node_u in neighbors_u:
            for node_v in neighbors_v:
                if node_u != node_v and not g6.has_edge(node_u, node_v):
                    g6.add_edge(node_u, node_v)

        adj_matrix = nx.adjacency_matrix(g6).toarray()
        # function should return the edge_index

        edge_index = torch.tensor(adj_matrix, dtype=torch.long)

        edge_index = edge_index.nonzero().t().contiguous() # (2,num_edges) format and contigous memory.



    return edge_index.to(device)
