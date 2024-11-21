from torch_geometric.datasets import TUDataset



def data_import(data):
    if data == "MUTAG": 
        mutag = TUDataset(root = 'Over_Squashing_GNNs/data/Mutag', name = 'MUTAG')

    elif data == "enzymes":
        enzymes = TUDataset(root = 'Over_Squashing_GNNs/data/Enzymes', name = 'ENZYMES')
    
    return data


