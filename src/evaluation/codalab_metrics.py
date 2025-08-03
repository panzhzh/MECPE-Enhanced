"""
Official CodaLab evaluation metrics for MECPE
Complete implementation following archive/original_tensorflow/CodaLab/evaluation/evaluate.py
"""
import numpy as np
from typing import Dict, List, Tuple, Any
import copy

# Official emotion categories mapping
EMOTION_IDX = {
    'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 
    'joy': 4, 'sadness': 5, 'surprise': 6
}

def cal_prf_pair_emocate(true_pairs: List[Tuple], pred_pairs: List[Tuple]) -> List[float]:
    """
    Calculate precision, recall, F1 for emotion-cause pairs (Subtask 2 evaluation)
    EXACT implementation of official CodaLab evaluation (evaluate.py:260-285)
    
    Args:
        true_pairs: List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
        pred_pairs: List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
    
    Returns:
        [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]
    """
    conf_mat = np.zeros([7, 7])  # 7 emotion categories
    
    # Process predicted pairs (line 262-266 in original)
    for p in pred_pairs:
        if p in true_pairs:
            conf_mat[p[3]][p[3]] += 1  # p[3] is emotion_category
        else:
            conf_mat[0][p[3]] += 1  # False positive
    
    # Process true pairs for false negatives (line 267-269 in original)
    for p in true_pairs:
        if p not in pred_pairs:
            conf_mat[p[3]][0] += 1  # False negative
    
    # Calculate precision, recall, F1 (line 270-277 in original)
    p = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis=0) + (1e-8), [1, 7]))
    r = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis=1) + (1e-8), [7, 1]))
    f = 2 * p * r / (p + r + (1e-8))
    
    # Weighted average (exclude neutral class - index 0)
    weight0 = np.sum(conf_mat, axis=1)
    weight = weight0[1:] / np.sum(weight0[1:])  # Original doesn't add 1e-8 here
    w_avg_p = np.sum(p[1:] * weight)
    w_avg_r = np.sum(r[1:] * weight)
    w_avg_f1 = np.sum(f[1:] * weight)
    
    # Micro average (line 279-282 in original)
    micro_acc = np.sum(np.diagonal(conf_mat)[1:])
    micro_p = micro_acc / (sum(np.sum(conf_mat, axis=0)[1:]) + (1e-8))
    micro_r = micro_acc / (sum(np.sum(conf_mat, axis=1)[1:]) + (1e-8))
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)
    
    # Return in official order (line 284)
    return [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]

def convert_predictions_to_pairs(batch_data: Dict) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Convert model predictions to emotion-cause pairs format
    
    Args:
        batch_data: Batch of conversation data with predictions
        
    Returns:
        (predicted_pairs, true_pairs) as lists of tuples
    """
    pred_pairs = []
    true_pairs = []
    
    conv_ids = batch_data.get('conv_ids', [])
    emotions_list = batch_data.get('emotions', [])
    pairs_list = batch_data.get('emotion_cause_pairs', [])
    
    # Extract emotion and cause predictions if available
    emotion_preds = batch_data.get('emotion_preds')  # [batch_size, max_utts]
    cause_preds = batch_data.get('cause_preds')      # [batch_size, max_utts]
    
    for i, conv_id in enumerate(conv_ids):
        conv_emotions = emotions_list[i] if i < len(emotions_list) else []
        conv_true_pairs = pairs_list[i] if i < len(pairs_list) else []
        
        # Convert true pairs to standard format
        for emo_utt, cause_utt in conv_true_pairs:
            if 0 < emo_utt <= len(conv_emotions):
                emotion_str = conv_emotions[emo_utt - 1]  # Convert to 0-indexed
                if emotion_str in EMOTION_IDX:
                    emotion_cat = EMOTION_IDX[emotion_str]
                    true_pairs.append((conv_id, emo_utt, cause_utt, emotion_cat))
        
        # Generate predicted pairs if predictions are available
        if emotion_preds is not None and cause_preds is not None:
            # Find predicted emotion and cause utterances
            emo_utts = []
            cause_utts = []
            
            max_utts = min(len(conv_emotions), emotion_preds.shape[1] if len(emotion_preds.shape) > 1 else len(emotion_preds))
            
            for j in range(max_utts):
                if emotion_preds[i, j] == 1:  # Predicted as emotion utterance
                    emotion_str = conv_emotions[j]
                    if emotion_str in EMOTION_IDX:
                        emotion_cat = EMOTION_IDX[emotion_str]
                        emo_utts.append((j + 1, emotion_cat))  # 1-indexed
                
                if cause_preds[i, j] == 1:  # Predicted as cause utterance
                    cause_utts.append(j + 1)  # 1-indexed
            
            # Generate all possible pairs (simple baseline strategy)
            for emo_utt, emotion_cat in emo_utts:
                for cause_utt in cause_utts:
                    pred_pairs.append((conv_id, emo_utt, cause_utt, emotion_cat))
    
    return pred_pairs, true_pairs

class CodaLabMetrics:
    """
    Official CodaLab metrics tracker
    Completely following the official evaluation standards
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_pred_pairs = []
        self.all_true_pairs = []
    
    def update(self, loss: float, batch_data: Dict):
        """
        Update metrics with batch results
        
        Args:
            loss: Training loss
            batch_data: Dictionary containing:
                - conv_ids: List of conversation IDs
                - emotions: List of emotion strings for each conversation
                - emotion_cause_pairs: List of true emotion-cause pairs
                - emotion_preds: Tensor of emotion predictions (optional)
                - cause_preds: Tensor of cause predictions (optional)
        """
        batch_size = len(batch_data.get('conv_ids', []))
        
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Extract pairs from current batch
        pred_pairs, true_pairs = convert_predictions_to_pairs(batch_data)
        
        self.all_pred_pairs.extend(pred_pairs)
        self.all_true_pairs.extend(true_pairs)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final official metrics
        
        Returns:
            Dictionary with official CodaLab metrics
        """
        if self.total_samples == 0:
            return {'avg_loss': 0.0}
        
        # Calculate official pair-level metrics
        results = cal_prf_pair_emocate(self.all_true_pairs, self.all_pred_pairs)
        
        metrics = {
            'avg_loss': self.total_loss / self.total_samples,
            'pair_precision': results[0],     # micro precision
            'pair_recall': results[1],        # micro recall  
            'pair_f1': results[2],           # micro F1
            'weighted_precision': results[3], # weighted precision
            'weighted_recall': results[4],    # weighted recall
            'weighted_f1': results[5],       # weighted F1 (main metric)
        }
        
        return metrics

def evaluate_official_format(pred_data: List[Dict], gold_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate in official CodaLab format
    
    Args:
        pred_data: List of prediction dictionaries
        gold_data: List of gold standard dictionaries
        
    Returns:
        Dictionary with official metrics
    """
    # Convert data to the format expected by official evaluation
    pred_pairs = []
    true_pairs = []
    
    # Create lookup dictionaries
    gold_dict = {item["conversation_ID"]: item for item in gold_data}
    pred_dict = {item["conversation_ID"]: item for item in pred_data}
    
    for conv_id, gold_item in gold_dict.items():
        if conv_id not in pred_dict:
            continue
            
        pred_item = pred_dict[conv_id]
        
        # Extract true pairs
        for pair in gold_item.get("emotion-cause_pairs", []):
            emo_info, cause_info = pair
            emo_id, emotion = emo_info.split('_')
            cause_id = cause_info.split('_')[0]
            
            if 'U' in emo_id:
                emo_id = emo_id.replace('U', '')
            if 'U' in cause_id:
                cause_id = cause_id.replace('U', '')
                
            if emotion in EMOTION_IDX:
                true_pairs.append((conv_id, int(emo_id), int(cause_id), EMOTION_IDX[emotion]))
        
        # Extract predicted pairs
        for pair in pred_item.get("emotion-cause_pairs", []):
            emo_info, cause_info = pair
            emo_id, emotion = emo_info.split('_')
            cause_id = cause_info.split('_')[0]
            
            if 'U' in emo_id:
                emo_id = emo_id.replace('U', '')
            if 'U' in cause_id:
                cause_id = cause_id.replace('U', '')
                
            if emotion in EMOTION_IDX:
                pred_pairs.append((conv_id, int(emo_id), int(cause_id), EMOTION_IDX[emotion]))
    
    # Calculate metrics
    results = cal_prf_pair_emocate(true_pairs, pred_pairs)
    
    return {
        'pair_precision': results[0],
        'pair_recall': results[1], 
        'pair_f1': results[2],
        'weighted_precision': results[3],
        'weighted_recall': results[4],
        'weighted_f1': results[5]
    }