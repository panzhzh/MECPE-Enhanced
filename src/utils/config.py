"""
Modern configuration management for MECPE-Enhanced
Reads from YAML files with automatic model dimension handling
"""
import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

# Model dimension mapping for automatic handling
MODEL_DIMENSIONS = {
    # RoBERTa models
    "roberta-base": 768,
    "roberta-large": 1024,
    
    # BERT models  
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
    
    # Future audio/video models can be added here
    # "wav2vec2-base": 768,
    # "wav2vec2-large": 1024,
}

class Config:
    """Main configuration class that reads from YAML"""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize config from YAML file
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self._load_config()
        self._post_init()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Load each section
        self.data = self._dict_to_obj(config_dict.get('data', {}))
        self.model = self._dict_to_obj(config_dict.get('model', {}))  
        self.training = self._dict_to_obj(config_dict.get('training', {}))
        self.experiment = self._dict_to_obj(config_dict.get('experiment', {}))
        
        # Auto-detect model dimensions
        self._setup_model_dimensions()
    
    def _dict_to_obj(self, d: Dict[str, Any]) -> object:
        """Convert dict to object for dot notation access"""
        class ConfigSection:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        setattr(self, k, self._dict_to_obj(v))
                    else:
                        setattr(self, k, v)
            
            def _dict_to_obj(self, d):
                return ConfigSection(**d)
        
        return ConfigSection(**d)
    
    def _setup_model_dimensions(self):
        """Automatically set model dimensions based on model names"""
        # Text model dimensions
        if hasattr(self.model, 'text_model'):
            text_model = self.model.text_model
            if text_model in MODEL_DIMENSIONS:
                self.model.text_hidden_size = MODEL_DIMENSIONS[text_model]
                print(f"✅ Auto-detected text model dimension: {text_model} -> {self.model.text_hidden_size}")
            else:
                # Default fallback
                self.model.text_hidden_size = 768
                print(f"⚠️  Unknown text model {text_model}, using default dimension 768")
        
        # Set other dimensions if not specified
        if not hasattr(self.model, 'hidden_size'):
            self.model.hidden_size = self.model.text_hidden_size
    
    def _post_init(self):
        """Post-initialization setup"""
        # Create directories
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs("experiments/logs", exist_ok=True)
        os.makedirs("experiments/results", exist_ok=True)
        
        # Auto-detect device
        if self.training.device == "auto":
            if torch.cuda.is_available():
                self.training.device = "cuda"
                print(f"✅ Auto-detected device: cuda")
            else:
                self.training.device = "cpu"
                print(f"✅ Auto-detected device: cpu")
    
    def get(self, key: str, default=None):
        """Get config value with dot notation (e.g., 'model.text_model')"""
        keys = key.split('.')
        obj = self
        for k in keys:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return default
        return obj
    
    def update(self, updates: Dict[str, Any]):
        """Update config values"""
        for key, value in updates.items():
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                if not hasattr(obj, k):
                    setattr(obj, k, self._dict_to_obj({}))
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
    
    def save(self, path: str):
        """Save current config to YAML file"""
        config_dict = {
            'data': self._obj_to_dict(self.data),
            'model': self._obj_to_dict(self.model),
            'training': self._obj_to_dict(self.training),
            'experiment': self._obj_to_dict(self.experiment)
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _obj_to_dict(self, obj) -> Dict[str, Any]:
        """Convert config object back to dict"""
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = self._obj_to_dict(value)
            else:
                result[key] = value
        return result
    
    def __str__(self):
        """String representation of config"""
        return f"""Config(
  Text Model: {self.model.text_model} (dim: {self.model.text_hidden_size})
  Batch Size: {self.training.batch_size}
  Learning Rate: {self.training.learning_rate}
  Device: {self.training.device}
  Experiment: {self.experiment.name}
)"""