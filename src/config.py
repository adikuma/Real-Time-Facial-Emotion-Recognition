import yaml
import os
from pathlib import Path

class Config:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.create_directories()
        
    def create_directories(self):
        directories = [
            self.config['training']['save_dir'],
            self.config['training']['log_dir'],
            self.config['training']['results_dir'],
            os.path.join(self.config['training']['results_dir'], 'metrics'),
            os.path.join(self.config['training']['results_dir'], 'plots')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    @property
    def device(self):
        return self.config['training']['device']
    
    @property
    def num_classes(self):
        return self.config['model']['num_classes']
    
    @property
    def learning_rate(self):
        return self.config['model']['learning_rate']
    
    @property
    def num_epochs(self):
        return self.config['model']['num_epochs']
    
    @property
    def batch_size(self):
        return self.config['data']['batch_size']
    
    @property
    def image_size(self):
        return self.config['data']['image_size']
    