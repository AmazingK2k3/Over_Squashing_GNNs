from copy import deepcopy
import yaml
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, Any

class ConfigError(Exception):
    pass

class Config:
    """
    Simplified configuration system supporting GNN models with rewiring strategies.
    """
    # Dataset configurations
    datasets = {
        'ENZYMES': 'Enzymes',
        'MUTAG': 'Mutag'
    }

    # Model configurations
    models = {
        'GIN': 'GIN',
        'GraphSAGE': 'GraphSAGE'
    }

    # Rewiring strategy configurations
    rewiring_strategies = {
        'adjacent': 'AdjacentRewiring',    # rewire1
        'bridge': 'BridgeRewiring',        # rewire2
        'combined': 'CombinedRewiring',    # rewire_combined
        'none': None
    }

    # Rewiring modes (where to apply the rewiring)
    rewiring_modes = {
        'last_layer': 'LastLayerRewiring',
        'full_model': 'FullModelRewiring',
        'none': None
    }

    # Optimizer configurations
    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'AdamW': optim.AdamW
    }

    # Learning rate scheduler configurations
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR
    }

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dict(kwargs)

        self._validate_config()
        self._set_attributes()
        
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_fields = ['dataset', 'model', 'model_params']
        for field in required_fields:
            if field not in self.config:
                raise ConfigError(f"Missing required field: {field}")

    def _set_attributes(self):
        """Set class attributes from config dictionary"""
        # Basic configurations
        self.dataset_name = self.config['dataset']
        self.model_name = self.config['model']
        self.model_params = self.config.get('model_params', {})
        
        # Parse configurations
        self.dataset = self.parse_dataset(self.dataset_name)
        self.model = self.parse_model(self.model_name)
        
        # Simplified rewiring configurations
        self.rewiring_config = {
            'strategy': self.parse_rewiring_strategy(
                self.config.get('rewiring_strategy', 'none')
            ),
            'mode': self.parse_rewiring_mode(
                self.config.get('rewiring_mode', 'none')
            )
        }
        
        # Training configurations
        self.training_config = {
            'optimizer': self.parse_optimizer(
                self.config.get('optimizer', {'name': 'Adam', 'args': {'lr': 0.01}})
            ),
            'scheduler': self.parse_scheduler(
                self.config.get('scheduler', None)
            ),
            'epochs': self.config.get('epochs', 100),
            'batch_size': self.config.get('batch_size', 32),
            'early_stopping': self.config.get('early_stopping', {'patience': 10, 'min_delta': 0.001})
        }

    @staticmethod
    def parse_rewiring_strategy(strategy_s: str) -> Optional[str]:
        """Parse rewiring strategy: adjacent, bridge, combined, or none"""
        if strategy_s == 'none' or strategy_s is None:
            return None
        if strategy_s not in Config.rewiring_strategies:
            raise ConfigError(f'Rewiring strategy {strategy_s} not found!')
        return Config.rewiring_strategies[strategy_s]

    @staticmethod
    def parse_rewiring_mode(mode_s: str) -> Optional[str]:
        """Parse rewiring mode: last_layer, full_model, or none"""
        if mode_s == 'none' or mode_s is None:
            return None
        if mode_s not in Config.rewiring_modes:
            raise ConfigError(f'Rewiring mode {mode_s} not found!')
        return Config.rewiring_modes[mode_s]

    # [Other parsing methods remain the same...]

    @property
    def exp_name(self) -> str:
        """Generate experiment name based on model, dataset and rewiring strategy"""
        rewiring = self.rewiring_config['strategy'] or 'no_rewiring'
        mode = self.rewiring_config['mode'] or 'no_mode'
        return f"{self.model_name}_{self.dataset_name}_{rewiring}_{mode}"