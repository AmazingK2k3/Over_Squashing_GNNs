
from copy import deepcopy
import yaml
import torch.optim as optim
from pathlib import Path

class ConfigError(Exception):
    pass

class Config:
    """
    Specifies the configuration for a single model with rewiring strategies.
    """
    # Dataset configurations
    datasets = {
        'ENZYMES': 'Enzymes',
        'MUTAG': 'Mutag',
        # Add more datasets as needed
    }

    # Model configurations
    models = {
        'GIN': 'GIN',
        'GraphSAGE': 'GraphSAGE',
        # Add more models as needed
    }

    # Optimizer configurations
    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }

    # Learning rate scheduler configurations
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau
    }

    # Rewiring strategy configurations
    rewiring_strategies = {
        'strategy1': 'Strategy1',
        'strategy2': 'Strategy2',
        'strategy3': 'Strategy3'
    }

    def __init__(self, config_path=None, **attrs):
        """
        Initialize configuration either from file or dictionary
        Args:
            config_path: Path to YAML configuration file
            attrs: Direct dictionary configuration
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dict(attrs)

        # Set attributes from config
        for attrname, value in self.config.items():
            if attrname in ['dataset', 'model', 'optimizer', 'scheduler', 'rewiring_strategy']:
                if attrname == 'dataset':
                    setattr(self, 'dataset_name', value)
                if attrname == 'model':
                    setattr(self, 'model_name', value)
                fn = getattr(self, f'parse_{attrname}')
                setattr(self, attrname, fn(value))
            else:
                setattr(self, attrname, value)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, attrname):
        return attrname in self.__dict__

    def __repr__(self):
        name = self.__class__.__name__
        return f'<{name}: {str(self.__dict__)}>'

    @property
    def exp_name(self):
        """Generate experiment name based on model, dataset and rewiring strategy"""
        rewiring = getattr(self, 'rewiring_strategy', 'no_rewiring')
        return f'{self.model_name}_{self.dataset_name}_{rewiring}'

    @property
    def config_dict(self):
        return self.config

    @staticmethod
    def parse_dataset(dataset_s):
        assert dataset_s in Config.datasets, f'Dataset {dataset_s} not found!'
        return Config.datasets[dataset_s]

    @staticmethod
    def parse_model(model_s):
        assert model_s in Config.models, f'Model {model_s} not found!'
        return Config.models[model_s]

    @staticmethod
    def parse_optimizer(optim_s):
        assert optim_s in Config.optimizers, f'Optimizer {optim_s} not found!'
        return Config.optimizers[optim_s]

    @staticmethod
    def parse_scheduler(sched_dict):
        if sched_dict is None:
            return None

        sched_s = sched_dict['class']
        args = sched_dict['args']

        assert sched_s in Config.schedulers, f'Scheduler {sched_s} not found!'
        return lambda opt: Config.schedulers[sched_s](opt, **args)

    @staticmethod
    def parse_rewiring_strategy(strategy_s):
        if strategy_s is None:
            return None
            
        assert strategy_s in Config.rewiring_strategies, f'Rewiring strategy {strategy_s} not found!'
        return Config.rewiring_strategies[strategy_s]

    @staticmethod
    def parse_gradient_clipping(clip_dict):
        if clip_dict is None:
            return None
        args = clip_dict['args']
        return None if not args['use'] else args['value']

    @classmethod
    def from_dict(cls, dict_obj):
        return Config(**dict_obj)


class Grid:
    """
    Specifies the configuration for multiple models and rewiring strategies.
    """
    def __init__(self, path_or_dict, dataset_name):
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, 'r') as f:
                self.configs_dict = yaml.safe_load(f)
        else:
            self.configs_dict = path_or_dict
            
        self.configs_dict['dataset'] = [dataset_name]
        self.num_configs = 0
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, 'No configurations available'
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        """Creates all possible combinations of configurations"""
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list
