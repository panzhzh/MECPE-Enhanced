"""
MECPE Evaluation Metrics
Unified interface for multiple evaluation methods
"""
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import specialized metrics
from .step1_metrics import Step1Metrics, get_predictions
from .codalab_metrics import CodaLabMetrics, convert_predictions_to_pairs

class UnifiedMetrics:
    """
    Unified metrics tracker that computes both Step1 and CodaLab metrics
    """
    
    def __init__(self, use_step1: bool = True, use_codalab: bool = True):
        self.use_step1 = use_step1
        self.use_codalab = use_codalab
        
        if self.use_step1:
            self.step1_metrics = Step1Metrics()
        if self.use_codalab:
            self.codalab_metrics = CodaLabMetrics()
    
    def reset(self):
        """Reset all metrics"""
        if self.use_step1:
            self.step1_metrics.reset()
        if self.use_codalab:
            self.codalab_metrics.reset()
    
    def update(self, loss: float,
               emotion_logits: torch.Tensor,
               cause_logits: torch.Tensor, 
               emotion_labels: torch.Tensor,
               cause_labels: torch.Tensor,
               attention_mask: torch.Tensor,
               conv_ids: List[int] = None,
               emotions: List[List[str]] = None,
               emotion_cause_pairs: List[List[Tuple]] = None,
               **kwargs):
        """
        Update metrics with batch results
        
        Args:
            loss: Training loss
            emotion_logits: [batch_size, max_utts, num_classes] emotion logits
            cause_logits: [batch_size, max_utts, num_classes] cause logits
            emotion_labels: [batch_size, max_utts] emotion labels
            cause_labels: [batch_size, max_utts] cause labels
            attention_mask: [batch_size, max_utts, max_tokens] attention mask
            conv_ids: List of conversation IDs
            emotions: List of emotion strings for each conversation
            emotion_cause_pairs: List of true emotion-cause pairs
        """
        # Update Step1 metrics
        if self.use_step1:
            self.step1_metrics.update(
                loss=loss,
                emotion_logits=emotion_logits,
                cause_logits=cause_logits,
                emotion_labels=emotion_labels,
                cause_labels=cause_labels,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Update CodaLab metrics
        if self.use_codalab and conv_ids is not None:
            # Get predictions for CodaLab evaluation
            emotion_preds = get_predictions(emotion_logits).cpu().numpy()
            cause_preds = get_predictions(cause_logits).cpu().numpy()
            
            batch_data = {
                'conv_ids': conv_ids,
                'emotions': emotions or [],
                'emotion_cause_pairs': emotion_cause_pairs or [],
                'emotion_preds': emotion_preds,
                'cause_preds': cause_preds
            }
            
            self.codalab_metrics.update(loss, batch_data)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        # Compute Step1 metrics
        if self.use_step1:
            step1_results = self.step1_metrics.compute()
            # Add step1_ prefix to distinguish from other metrics
            for key, value in step1_results.items():
                if key != 'avg_loss':  # Don't duplicate loss
                    metrics[f'step1_{key}'] = value
                else:
                    metrics[key] = value
        
        # Compute CodaLab metrics
        if self.use_codalab:
            codalab_results = self.codalab_metrics.compute()
            # Add codalab_ prefix to distinguish from other metrics  
            for key, value in codalab_results.items():
                if key != 'avg_loss':  # Don't duplicate loss
                    metrics[f'codalab_{key}'] = value
                elif 'avg_loss' not in metrics:
                    metrics[key] = value
        
        return metrics

# Legacy compatibility - alias the old SimpleMetrics to UnifiedMetrics
SimpleMetrics = UnifiedMetrics