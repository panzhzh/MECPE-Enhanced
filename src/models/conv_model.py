"""
Simple baseline model for MECPE
Text-only transformer model with simple classification heads
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional

class SimpleBaselineModel(nn.Module):
    """
    Simple baseline model for emotion-cause pair extraction
    Text-only with RoBERTa encoder + classification heads
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder (RoBERTa)
        self.text_encoder = AutoModel.from_pretrained(config.model.text_model)
        self.hidden_size = config.model.text_hidden_size
        
        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(config.model.fusion_dropout),
            nn.Linear(self.hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.fusion_dropout),
            nn.Linear(config.model.fusion_hidden_size, config.model.num_labels)
        )
        
        self.cause_classifier = nn.Sequential(
            nn.Dropout(config.model.fusion_dropout),
            nn.Linear(self.hidden_size, config.model.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.fusion_dropout),
            nn.Linear(config.model.fusion_hidden_size, config.model.num_labels)
        )
        
        print(f"✅ Initialized SimpleBaselineModel with {config.model.text_model}")
        print(f"   Hidden size: {self.hidden_size}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len, max_tokens]
            attention_mask: [batch_size, seq_len, max_tokens]
            
        Returns:
            Dict with emotion_logits and cause_logits
        """
        batch_size, seq_len, max_tokens = input_ids.shape
        
        # Reshape for transformer: [batch_size * seq_len, max_tokens]
        input_ids_flat = input_ids.view(-1, max_tokens)
        attention_mask_flat = attention_mask.view(-1, max_tokens)
        
        # Encode text
        outputs = self.text_encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        
        # Get [CLS] token embeddings: [batch_size * seq_len, hidden_size]
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Reshape back: [batch_size, seq_len, hidden_size]
        text_embeddings = text_embeddings.view(batch_size, seq_len, self.hidden_size)
        
        # Classification
        emotion_logits = self.emotion_classifier(text_embeddings)  # [batch_size, seq_len, 2]
        cause_logits = self.cause_classifier(text_embeddings)      # [batch_size, seq_len, 2]
        
        return {
            'emotion_logits': emotion_logits,
            'cause_logits': cause_logits,
            'embeddings': text_embeddings
        }
    
    def compute_loss(self, 
                    emotion_logits: torch.Tensor,
                    cause_logits: torch.Tensor,
                    emotion_labels: torch.Tensor,
                    cause_labels: torch.Tensor,
                    attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with masking for padding tokens
        
        Args:
            emotion_logits: [batch_size, seq_len, 2]
            cause_logits: [batch_size, seq_len, 2]
            emotion_labels: [batch_size, seq_len]
            cause_labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len, max_tokens] (use seq-level mask)
            
        Returns:
            Dict with losses
        """
        # Create sequence-level mask (any token is not padding)
        seq_mask = attention_mask.sum(dim=-1) > 0  # [batch_size, seq_len]
        
        # Flatten for loss computation
        emotion_logits_flat = emotion_logits.view(-1, 2)
        cause_logits_flat = cause_logits.view(-1, 2)
        emotion_labels_flat = emotion_labels.view(-1)
        cause_labels_flat = cause_labels.view(-1)
        seq_mask_flat = seq_mask.view(-1)
        
        # Compute masked loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        emotion_loss = loss_fn(emotion_logits_flat, emotion_labels_flat)
        cause_loss = loss_fn(cause_logits_flat, cause_labels_flat)
        
        # Apply mask and average
        emotion_loss = (emotion_loss * seq_mask_flat).sum() / seq_mask_flat.sum()
        cause_loss = (cause_loss * seq_mask_flat).sum() / seq_mask_flat.sum()
        
        total_loss = emotion_loss + cause_loss
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'cause_loss': cause_loss
        }

def create_baseline_model(config):
    """Factory function to create baseline model"""
    return SimpleBaselineModel(config)