
# Copyright (C)  2020  University of Pisa
# need to put copyright?

from rewire import * 
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

# GraphSAGE model from Errica et al repo
class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        # Configuration settings
        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # 'mean' or 'max'
    

        # Optional max aggregation layer
        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        # Define the SAGEConv layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding
            self.layers.append(SAGEConv(dim_input, dim_embedding, aggr=self.aggregation))

        # Classification head
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_all = []

        for layer in self.layers:
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = F.relu(self.fc_max(x))
            x_all.append(x)

        # Concatenate features from all layers
        x = torch.cat(x_all, dim=1)

        # Apply global pooling
        x = global_max_pool(x, batch) if self.aggregation == 'max' else global_mean_pool(x, batch)

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
