from rewire import *
from GraphSAGE import * 
from torch_geometric.loader import DataLoader

class DatasetGetter:
    def __init__(self,dataset):
        self.dataset = dataset
    
    def get_data_loaders(self):
        # Split dataset into training and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

class GraphSAGE_exp:
    def __init__(self, model_config, exp_path):
        self.model_config   = model_config
        self.exp_path       = exp_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init GraphSAGE
        self.model = GraphSAGE(
            dim_features = config['dim_features'],
            dim_target = config['dim_target']
            config = config)
        
    def run_valid(self, dataset_getter, logger, other=None):
        train_loader, val_loader = dataset_getter.get_data_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])

        # Training loop
        self.model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = self.model(data)
            loss = F.binary_cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

        # Validation
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                out = self.model(data)
                pred = (out > 0.5).float()
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

        # Return training and validation accuracy
        tr_score = 1 - loss.item()  # Example metric
        vl_score = correct / total
        return tr_score, vl_score

model_configs = [
    {'num_layers': 3, 'dim_embedding': 64, 'aggregation': 'mean', 'lr': 0.001, 'dim_features': 7, 'dim_target': 1},
    {'num_layers': 5, 'dim_embedding': 32, 'aggregation': 'max', 'lr': 0.0005, 'dim_features': 7, 'dim_target': 1},
    {'num_layers': 3, 'dim_embedding': 64, 'aggregation': 'mean', 'lr': 0.001, 'dim_features': 7, 'dim_target': 1},
    {'num_layers': 5, 'dim_embedding': 64, 'aggregation': 'mean', 'lr': 0.001, 'dim_features': 7, 'dim_target': 1},
    {'num_layers': 3, 'dim_embedding': 64, 'aggregation': 'mean', 'lr': 0.001, 'dim_features': 7, 'dim_target': 1},
    
]

selector = HoldOutSelector(max_processes = 4)
dataset_getter = DatasetGetter(data)
exp_path = "/teamspace/studios/this_studio/Over_Squashing_GNNs"

best_config = selector.model_selection(
    dataset_getter = data,
    experiment_class = ,
    exp_path = exp_path,
    model_configs = model_configs,
    debug = False
)