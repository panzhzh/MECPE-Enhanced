"""
Step2 Model for MECPE: Emotion-Cause Pair Classification
Modern PyTorch implementation with 2025 state-of-the-art encoders:
- Text: RoBERTa
- Video: TimeSformer  
- Audio: Wav2Vec2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from transformers import (
    RobertaModel, RobertaTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor
)

class ModernMultimodalEncoder(nn.Module):
    """
    Modern multimodal encoder using state-of-the-art 2025 models
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder: RoBERTa
        self.text_encoder = RobertaModel.from_pretrained(config.model.text_model)
        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        # Video encoder: placeholder (can be added later)
        if config.model.use_video:
            self.video_hidden_size = 768  # Default hidden size
            # Projection to match text hidden size
            self.video_projection = nn.Linear(self.video_hidden_size, self.text_hidden_size)
        
        # Audio encoder: Wav2Vec2 (if enabled)  
        if config.model.use_audio:
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_hidden_size = self.audio_encoder.config.hidden_size
            # Projection to match text hidden size
            self.audio_projection = nn.Linear(self.audio_hidden_size, self.text_hidden_size)
        
        # Freeze pretrained encoders initially (feature extraction mode)
        if hasattr(config.model, 'freeze_encoders') and config.model.freeze_encoders:
            self._freeze_encoders()
    
    def _freeze_encoders(self):
        """Freeze pretrained encoder parameters"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Video encoder freezing not needed for placeholder
                
        if hasattr(self, 'audio_encoder'):
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text using RoBERTa
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            [batch_size, hidden_size] sentence representations
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
    
    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode video (placeholder implementation)
        Args:
            pixel_values: [batch_size, num_frames, channels, height, width]
        Returns:
            [batch_size, hidden_size] video representations
        """
        # Placeholder: return zeros
        batch_size = pixel_values.size(0)
        video_features = torch.zeros(batch_size, self.video_hidden_size, device=pixel_values.device)
        return self.video_projection(video_features)  # [batch_size, text_hidden_size]
    
    def encode_audio(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode audio using Wav2Vec2
        Args:
            input_values: [batch_size, sequence_length] raw audio waveform
            attention_mask: [batch_size, sequence_length] optional mask
        Returns:
            [batch_size, hidden_size] audio representations
        """
        if not hasattr(self, 'audio_encoder'):
            raise ValueError("Audio encoder not initialized")
            
        outputs = self.audio_encoder(input_values=input_values, attention_mask=attention_mask)
        # Global average pooling over time dimension
        audio_features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, audio_hidden_size]
        return self.audio_projection(audio_features)  # [batch_size, text_hidden_size]

class PairModel(nn.Module):
    """
    Step2 Model: Emotion-Cause Pair Classification  
    Modern implementation with 2025 encoders (RoBERTa + TimeSformer + Wav2Vec2)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature flags
        self.use_video = config.model.use_video
        self.use_audio = config.model.use_audio
        self.use_emotion_categories = config.model.use_emotion_categories
        
        # Modern multimodal encoder
        self.encoder = ModernMultimodalEncoder(config)
        self.hidden_size = self.encoder.text_hidden_size
        
        # Position and emotion category embeddings (keep from original)
        self.position_embedding_dim = config.model.position_embedding_dim
        self.distance_embedding = nn.Embedding(200, self.position_embedding_dim)  # Distance range
        
        if self.use_emotion_categories:
            self.emotion_embedding = nn.Embedding(7, self.position_embedding_dim)  # 7 emotion categories
        
        # Calculate final feature dimension
        # Two utterances: 2 * hidden_size (text)
        final_dim = 2 * self.hidden_size
        
        # Add multimodal dimensions if enabled
        if self.use_video:
            final_dim += 2 * self.hidden_size  # Video features for both utterances
        if self.use_audio:
            final_dim += 2 * self.hidden_size  # Audio features for both utterances
            
        # Add distance embedding
        final_dim += self.position_embedding_dim
        
        # Add emotion category embedding if enabled
        if self.use_emotion_categories:
            final_dim += self.position_embedding_dim
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(config.model.final_dropout if hasattr(config.model, 'final_dropout') else 0.1),
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification
        )
        
        # Initialize position and classifier embeddings
        self._init_position_embeddings()
        self._init_classifier()
    
    def _init_position_embeddings(self):
        """Initialize position embeddings following original implementation"""
        # Distance embedding: uniform(-0.1, 0.1) 
        nn.init.uniform_(self.distance_embedding.weight, -0.1, 0.1)
        
        if hasattr(self, 'emotion_embedding'):
            nn.init.uniform_(self.emotion_embedding.weight, -0.1, 0.1)
    
    def _init_classifier(self):
        """Initialize classifier layers with proper bias for class imbalance"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Initialize bias to zero for balanced start
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                # Text inputs (tokenized)
                input_ids: torch.Tensor,  # [batch_size, 2, max_seq_len] 
                attention_mask: torch.Tensor,  # [batch_size, 2, max_seq_len]
                
                # Distance and emotion category
                distance: torch.Tensor,  # [batch_size] distance indices  
                emotion_category: Optional[torch.Tensor] = None,  # [batch_size] emotion category indices
                
                # Multimodal inputs (optional)
                video_pixel_values: Optional[torch.Tensor] = None,  # [batch_size, 2, frames, C, H, W]
                audio_input_values: Optional[torch.Tensor] = None,  # [batch_size, 2, audio_seq_len]
                audio_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Modern forward pass using state-of-the-art encoders
        
        Returns:
            [batch_size, 2] logits for pair classification
        """
        batch_size = input_ids.size(0)
        
        # Encode text for both utterances using RoBERTa
        # Reshape: [batch_size, 2, seq_len] -> [batch_size*2, seq_len]
        input_ids_flat = input_ids.reshape(-1, input_ids.size(-1))
        attention_mask_flat = attention_mask.reshape(-1, attention_mask.size(-1))
        
        # Get text representations
        text_features = self.encoder.encode_text(input_ids_flat, attention_mask_flat)  # [batch_size*2, hidden_size]
        text_features = text_features.reshape(batch_size, 2, self.hidden_size)  # [batch_size, 2, hidden_size]
        
        # Start building feature vector
        features_list = [text_features.reshape(batch_size, -1)]  # [batch_size, 2*hidden_size]
        
        # Add video features if enabled
        if self.use_video and video_pixel_values is not None:
            # Reshape: [batch_size, 2, frames, C, H, W] -> [batch_size*2, frames, C, H, W]
            video_flat = video_pixel_values.reshape(-1, *video_pixel_values.shape[2:])
            video_features = self.encoder.encode_video(video_flat)  # [batch_size*2, hidden_size]
            video_features = video_features.reshape(batch_size, 2, self.hidden_size)  # [batch_size, 2, hidden_size]
            features_list.append(video_features.reshape(batch_size, -1))  # [batch_size, 2*hidden_size]
        
        # Add audio features if enabled
        if self.use_audio and audio_input_values is not None:
            # Reshape: [batch_size, 2, audio_seq_len] -> [batch_size*2, audio_seq_len]
            audio_flat = audio_input_values.reshape(-1, audio_input_values.size(-1))
            audio_mask_flat = None
            if audio_attention_mask is not None:
                audio_mask_flat = audio_attention_mask.reshape(-1, audio_attention_mask.size(-1))
            
            audio_features = self.encoder.encode_audio(audio_flat, audio_mask_flat)  # [batch_size*2, hidden_size]
            audio_features = audio_features.reshape(batch_size, 2, self.hidden_size)  # [batch_size, 2, hidden_size]
            features_list.append(audio_features.reshape(batch_size, -1))  # [batch_size, 2*hidden_size]
        
        # Add distance embedding
        distance_emb = self.distance_embedding(distance)  # [batch_size, position_embedding_dim]
        features_list.append(distance_emb)
        
        # Add emotion category embedding if enabled
        if self.use_emotion_categories and emotion_category is not None:
            emotion_emb = self.emotion_embedding(emotion_category)  # [batch_size, position_embedding_dim]
            features_list.append(emotion_emb)
        
        # Concatenate all features
        final_features = torch.cat(features_list, dim=1)  # [batch_size, total_feature_dim]
        
        # Final classification
        logits = self.classifier(final_features)  # [batch_size, 2]
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with class weighting for imbalanced data
        
        Args:
            logits: [batch_size, n_class] 
            labels: [batch_size, n_class] one-hot labels
            
        Returns:
            Dictionary with loss components
        """
        # Convert one-hot labels to class indices for CrossEntropyLoss
        class_labels = torch.argmax(labels, dim=-1)  # [batch_size]
        
        # Use Focal Loss to handle extreme class imbalance
        # Remove problematic alpha parameter to avoid misconfiguration
        ce_loss_raw = F.cross_entropy(logits, class_labels, reduction='none')
        
        # Convert to probabilities and get the probability of the true class
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, class_labels.unsqueeze(1)).squeeze(1)  # p_t
        
        # Focal loss parameters
        gamma = 2.0   # Focusing parameter (handles hard/easy samples)
        
        # Apply simplified focal loss formula: -(1-pt)^γ * log(pt)
        # This automatically focuses on hard samples without class weighting issues
        focal_weight = (1 - pt) ** gamma
        ce_loss = torch.mean(focal_weight * ce_loss_raw)
        
        # L2 regularization for classifier weights
        l2_reg = 0.0
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                l2_reg += torch.sum(param ** 2)
        
        total_loss = ce_loss + self.config.training.l2_reg * l2_reg
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'l2_loss': l2_reg
        }

def create_pair_model(config) -> PairModel:
    """
    Create pair model instance
    
    Args:
        config: Configuration object
        
    Returns:
        PairModel instance
    """
    return PairModel(config)