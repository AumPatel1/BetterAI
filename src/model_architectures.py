"""
Model architectures for reward models.

This module provides different reward model architectures and utilities
for model selection and customization.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaModel,
    RobertaPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
    DebertaModel,
    DebertaPreTrainedModel
)
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CustomRewardModel(nn.Module):
    """Base class for custom reward model architectures."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def get_reward_score(self, text: str, tokenizer) -> float:
        """Get reward score for a given text."""
        raise NotImplementedError

class RobertaRewardModel(CustomRewardModel):
    """RoBERTa-based reward model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "roberta-base")
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Reward head
        hidden_size = self.roberta.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get reward score
        reward_score = self.reward_head(pooled_output)
        
        return reward_score
    
    def get_reward_score(self, text: str, tokenizer) -> float:
        """Get reward score for a given text."""
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            reward_score = self.forward(**inputs)
            return reward_score.item()

class BertRewardModel(CustomRewardModel):
    """BERT-based reward model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "bert-base-uncased")
        self.bert = BertModel.from_pretrained(model_name)
        
        # Reward head
        hidden_size = self.bert.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get reward score
        reward_score = self.reward_head(pooled_output)
        
        return reward_score
    
    def get_reward_score(self, text: str, tokenizer) -> float:
        """Get reward score for a given text."""
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            reward_score = self.forward(**inputs)
            return reward_score.item()

class DebertaRewardModel(CustomRewardModel):
    """DeBERTa-based reward model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "microsoft/deberta-base")
        self.deberta = DebertaModel.from_pretrained(model_name)
        
        # Reward head
        hidden_size = self.deberta.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get reward score
        reward_score = self.reward_head(pooled_output)
        
        return reward_score
    
    def get_reward_score(self, text: str, tokenizer) -> float:
        """Get reward score for a given text."""
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            reward_score = self.forward(**inputs)
            return reward_score.item()

class MultiLayerRewardModel(CustomRewardModel):
    """Multi-layer reward model with customizable architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "roberta-base")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Customizable reward head
        hidden_size = self.backbone.config.hidden_size
        layer_sizes = config.get("layer_sizes", [hidden_size, hidden_size // 2, hidden_size // 4, 1])
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Don't add activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.get("dropout", 0.1)))
        
        self.reward_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get reward score
        reward_score = self.reward_head(pooled_output)
        
        return reward_score
    
    def get_reward_score(self, text: str, tokenizer) -> float:
        """Get reward score for a given text."""
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            reward_score = self.forward(**inputs)
            return reward_score.item()

class ModelFactory:
    """Factory for creating reward model instances."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> CustomRewardModel:
        """Create a reward model based on configuration."""
        model_type = config.get("model_type", "roberta")
        
        if model_type == "roberta":
            return RobertaRewardModel(config)
        elif model_type == "bert":
            return BertRewardModel(config)
        elif model_type == "deberta":
            return DebertaRewardModel(config)
        elif model_type == "multilayer":
            return MultiLayerRewardModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_tokenizer(config: Dict[str, Any]):
        """Get tokenizer for the specified model."""
        model_name = config.get("model_name", "roberta-base")
        return AutoTokenizer.from_pretrained(model_name)

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available model configurations."""
    return {
        "roberta-base": {
            "model_type": "roberta",
            "model_name": "roberta-base",
            "description": "RoBERTa base model for reward modeling"
        },
        "roberta-large": {
            "model_type": "roberta",
            "model_name": "roberta-large",
            "description": "RoBERTa large model for reward modeling"
        },
        "bert-base": {
            "model_type": "bert",
            "model_name": "bert-base-uncased",
            "description": "BERT base model for reward modeling"
        },
        "bert-large": {
            "model_type": "bert",
            "model_name": "bert-large-uncased",
            "description": "BERT large model for reward modeling"
        },
        "deberta-base": {
            "model_type": "deberta",
            "model_name": "microsoft/deberta-base",
            "description": "DeBERTa base model for reward modeling"
        },
        "deberta-large": {
            "model_type": "deberta",
            "model_name": "microsoft/deberta-large",
            "description": "DeBERTa large model for reward modeling"
        },
        "multilayer-roberta": {
            "model_type": "multilayer",
            "model_name": "roberta-base",
            "layer_sizes": [768, 384, 192, 96, 1],
            "dropout": 0.1,
            "description": "Multi-layer RoBERTa with custom architecture"
        }
    }

def print_available_models():
    """Print available model configurations."""
    models = get_available_models()
    
    print("Available Reward Model Configurations:")
    print("=" * 50)
    
    for name, config in models.items():
        print(f"\n{name}:")
        print(f"  Type: {config['model_type']}")
        print(f"  Base Model: {config['model_name']}")
        print(f"  Description: {config['description']}")
        
        if "layer_sizes" in config:
            print(f"  Architecture: {config['layer_sizes']}")
        if "dropout" in config:
            print(f"  Dropout: {config['dropout']}")

if __name__ == "__main__":
    print_available_models() 