"""
Pair-level metrics for MECPE pair classification task
Designed specifically for pair-level training and evaluation
"""
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class PairLevelMetrics:
    """
    Metrics specifically for pair-level emotion-cause pair classification
    Works directly with pair predictions rather than utterance-level predictions
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        
        # Store all predicted and true pairs for final evaluation
        self.all_predicted_pairs = set()  # Set of (doc_id, emo_utt, cause_utt, emotion_cat)
        self.all_true_pairs = set()       # Set of (doc_id, emo_utt, cause_utt, emotion_cat)
        
        # Per-document tracking for detailed analysis
        self.doc_predictions = defaultdict(set)  # doc_id -> set of predicted pairs
        self.doc_true_pairs = defaultdict(set)   # doc_id -> set of true pairs
    
    def update(self, loss: float, doc_id: int, pair_predictions: List[Tuple], true_pairs: List[Tuple]):
        """
        Update metrics with batch results
        
        Args:
            loss: Training loss for this batch
            doc_id: Document/conversation ID
            pair_predictions: List of (emo_utt, cause_utt, emotion_cat) predicted as positive
            true_pairs: List of (emo_utt, cause_utt, emotion_cat) that are actually positive
        """
        batch_size = 1  # One document at a time
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Convert to full format with doc_id
        predicted_full = {(doc_id, emo, cause, cat) for emo, cause, cat in pair_predictions}
        true_full = {(doc_id, emo, cause, cat) for emo, cause, cat in true_pairs}
        
        # Update global sets
        self.all_predicted_pairs.update(predicted_full)
        self.all_true_pairs.update(true_full)
        
        # Update per-document tracking
        self.doc_predictions[doc_id] = predicted_full
        self.doc_true_pairs[doc_id] = true_full
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final pair-level metrics
        
        Returns:
            Dictionary with pair-level precision, recall, F1 metrics
        """
        if self.total_samples == 0:
            return {
                'avg_loss': 0.0,
                'pair_precision': 0.0, 'pair_recall': 0.0, 'pair_f1': 0.0,
                'weighted_precision': 0.0, 'weighted_recall': 0.0, 'weighted_f1': 0.0,
                'num_predicted_pairs': 0, 'num_true_pairs': 0, 'num_correct_pairs': 0
            }
        
        # Use official CodaLab evaluation function
        true_pairs_list = list(self.all_true_pairs)
        pred_pairs_list = list(self.all_predicted_pairs)
        
        # Call official evaluation function
        from src.evaluation.codalab_metrics import cal_prf_pair_emocate
        if len(true_pairs_list) > 0 or len(pred_pairs_list) > 0:
            results = cal_prf_pair_emocate(true_pairs_list, pred_pairs_list)
            micro_p, micro_r, micro_f1, weighted_p, weighted_r, weighted_f1 = results
        else:
            micro_p = micro_r = micro_f1 = 0.0
            weighted_p = weighted_r = weighted_f1 = 0.0
        
        # Calculate basic statistics for debugging
        tp = len(self.all_predicted_pairs & self.all_true_pairs)
        fp = len(self.all_predicted_pairs - self.all_true_pairs)
        fn = len(self.all_true_pairs - self.all_predicted_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Per-document analysis for weighted metrics
        doc_precisions = []
        doc_recalls = []
        doc_f1s = []
        doc_weights = []
        
        for doc_id in set(list(self.doc_predictions.keys()) + list(self.doc_true_pairs.keys())):
            pred_pairs = self.doc_predictions.get(doc_id, set())
            true_pairs = self.doc_true_pairs.get(doc_id, set())
            
            if len(true_pairs) == 0:
                continue  # Skip documents with no true pairs
            
            doc_tp = len(pred_pairs & true_pairs)
            doc_fp = len(pred_pairs - true_pairs)
            doc_fn = len(true_pairs - pred_pairs)
            
            doc_precision = doc_tp / (doc_tp + doc_fp) if (doc_tp + doc_fp) > 0 else 0.0
            doc_recall = doc_tp / (doc_tp + doc_fn) if (doc_tp + doc_fn) > 0 else 0.0
            doc_f1 = 2 * doc_precision * doc_recall / (doc_precision + doc_recall) if (doc_precision + doc_recall) > 0 else 0.0
            
            doc_precisions.append(doc_precision)
            doc_recalls.append(doc_recall)
            doc_f1s.append(doc_f1)
            doc_weights.append(len(true_pairs))  # Weight by number of true pairs
        
        # Calculate weighted averages
        if len(doc_weights) > 0 and sum(doc_weights) > 0:
            total_weight = sum(doc_weights)
            weighted_precision = sum(p * w for p, w in zip(doc_precisions, doc_weights)) / total_weight
            weighted_recall = sum(r * w for r, w in zip(doc_recalls, doc_weights)) / total_weight
            weighted_f1 = sum(f * w for f, w in zip(doc_f1s, doc_weights)) / total_weight
        else:
            weighted_precision = precision
            weighted_recall = recall
            weighted_f1 = f1
        
        return {
            'avg_loss': self.total_loss / self.total_samples,
            # Official CodaLab metrics (these are the authoritative ones)
            'pair_precision': micro_p,      # Official micro precision  
            'pair_recall': micro_r,         # Official micro recall
            'pair_f1': micro_f1,           # Official micro F1
            'weighted_precision': weighted_p, # Official weighted precision (main metric)
            'weighted_recall': weighted_r,    # Official weighted recall  
            'weighted_f1': weighted_f1,      # Official weighted F1 (main metric)
            # Debug information
            'simple_precision': precision,   # Simple TP/(TP+FP) for debugging
            'simple_recall': recall,         # Simple TP/(TP+FN) for debugging  
            'simple_f1': f1,                # Simple F1 for debugging
            'num_predicted_pairs': len(self.all_predicted_pairs),
            'num_true_pairs': len(self.all_true_pairs),
            'num_correct_pairs': tp,
            'num_documents': len(set(list(self.doc_predictions.keys()) + list(self.doc_true_pairs.keys())))
        }