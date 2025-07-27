from torch_geometric.data import Data

class CustomData(Data):
    def __inc__(self, key, value):
        if key == 'rewired_edge_index':
            return self.num_nodes
        return super().__inc__(key, value)
