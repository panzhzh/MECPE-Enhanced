"""
Official CodaLab-compliant metrics for MECPE Subtask 2
Strictly follows official evaluation standards - no custom evaluation logic
"""
from typing import Dict, List, Tuple

class PairLevelMetrics:
    """
    Official CodaLab metrics for Subtask 2: Multimodal Emotion-Cause Pair Extraction
    
    Subtask 2 Format: [emo_utt_id, emotion_category, cause_utt_id]
    Main metric: w-avg. F1 (weighted F1)
    Secondary metric: micro F1
    
    This class only collects data and calls official CodaLab evaluation functions.
    No custom evaluation logic is implemented here.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        
        # Store pairs in official CodaLab format for Subtask 2
        # Format: (conv_id, emo_utt_id, cause_utt_id, emotion_category)
        self.all_predicted_pairs = []
        self.all_true_pairs = []
    
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
        
        # Convert to official CodaLab Subtask 2 format: (conv_id, emo_utt_id, cause_utt_id, emotion_category)
        for emo_utt, cause_utt, emotion_cat in pair_predictions:
            self.all_predicted_pairs.append((doc_id, emo_utt, cause_utt, emotion_cat))
        
        for emo_utt, cause_utt, emotion_cat in true_pairs:
            self.all_true_pairs.append((doc_id, emo_utt, cause_utt, emotion_cat))
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics using official CodaLab evaluation function
        
        Returns:
            Dictionary with official CodaLab metrics for Subtask 2
        """
        if self.total_samples == 0:
            return {
                'avg_loss': 0.0,
                'pair_precision': 0.0, 'pair_recall': 0.0, 'pair_f1': 0.0,
                'weighted_precision': 0.0, 'weighted_recall': 0.0, 'weighted_f1': 0.0,
                'num_predicted_pairs': 0, 'num_true_pairs': 0, 'num_correct_pairs': 0,
                'num_documents': 0
            }
        
        # Call official CodaLab evaluation function for Subtask 2
        from src.evaluation.codalab_metrics import cal_prf_pair_emocate
        
        if len(self.all_true_pairs) > 0 or len(self.all_predicted_pairs) > 0:
            # Official evaluation - returns [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]
            results = cal_prf_pair_emocate(self.all_true_pairs, self.all_predicted_pairs)
            micro_p, micro_r, micro_f1, weighted_p, weighted_r, weighted_f1 = results
        else:
            micro_p = micro_r = micro_f1 = 0.0
            weighted_p = weighted_r = weighted_f1 = 0.0
        
        # Calculate basic statistics for monitoring
        predicted_set = set(self.all_predicted_pairs)
        true_set = set(self.all_true_pairs)
        correct_pairs = predicted_set & true_set
        
        # Count unique documents
        all_doc_ids = set()
        for conv_id, _, _, _ in self.all_predicted_pairs + self.all_true_pairs:
            all_doc_ids.add(conv_id)
        
        return {
            'avg_loss': self.total_loss / self.total_samples,
            
            # Official CodaLab Subtask 2 metrics
            'pair_precision': micro_p,        # Micro precision
            'pair_recall': micro_r,           # Micro recall  
            'pair_f1': micro_f1,             # Micro F1
            'weighted_precision': weighted_p,  # W-avg precision (main metric)
            'weighted_recall': weighted_r,     # W-avg recall
            'weighted_f1': weighted_f1,       # W-avg F1 (main metric for ranking)
            
            # Debug/monitoring information
            'num_predicted_pairs': len(self.all_predicted_pairs),
            'num_true_pairs': len(self.all_true_pairs),
            'num_correct_pairs': len(correct_pairs),
            'num_documents': len(all_doc_ids)
        }