"""
Step2 evaluation metrics for MECPE
Emotion-cause pair-level classification evaluation
Ported from archive/original_tensorflow/utils/pre_data_bert.py
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Import the Step1 pair evaluation functions
from .step1_metrics import cal_prf

def prf_2nd_step(pair_id_all: List[Tuple], pair_id: List[Tuple], pred_y: np.ndarray) -> List[float]:
    """
    Calculate precision, recall, F1 for Step2 pair classification (without emotion categories)
    Following the original TensorFlow implementation
    
    Args:
        pair_id_all: List of all true pairs [(doc_id, emo_utt, cause_utt, emotion_cat)]
        pair_id: List of candidate pairs [(doc_id, emo_utt, cause_utt, emotion_cat)]
        pred_y: [N] predicted labels (0/1) for candidate pairs
        
    Returns:
        [filtered_p, filtered_r, filtered_f1, original_p, original_r, original_f1, keep_rate]
    """
    # Filter pairs based on predictions
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    
    def cal_prf_pairs(pair_id_all, pair_id):
        """Calculate P/R/F1 for pairs"""
        acc_num, true_num, pred_num = 0, len(pair_id_all), len(pair_id)
        for p in pair_id:
            if p in pair_id_all:
                acc_num += 1
        p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
        return [p, r, f1]
    
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)
    return cal_prf_pairs(pair_id_all, pair_id_filtered) + cal_prf_pairs(pair_id_all, pair_id) + [keep_rate]

def prf_2nd_step_emocate(pair_id_all: List[Tuple], pair_id: List[Tuple], pred_y: np.ndarray) -> List[float]:
    """
    Calculate precision, recall, F1 for Step2 pair classification with emotion categories
    Following the original TensorFlow implementation
    
    Args:
        pair_id_all: List of all true pairs [(doc_id, emo_utt, cause_utt, emotion_cat)]
        pair_id: List of candidate pairs [(doc_id, emo_utt, cause_utt, emotion_cat)]
        pred_y: [N] predicted labels (0/1) for candidate pairs
        
    Returns:
        [f1_emo1-6, w_avg_f, o_f1_emo1-6, o_w_avg_f, keep_rate] (length 15)
    """
    # Filter pairs based on predictions
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)

    def cal_prf_emocate_pairs(pair_id_all, pair_id):
        """Calculate emotion-category-wise P/R/F1 for pairs"""
        conf_mat = np.zeros([7, 7])  # 7 emotion categories
        
        # Convert to sets for efficient lookup
        pair_id_all_set = set(pair_id_all)
        
        for p in pair_id:
            if p in pair_id_all_set:
                emotion_cat = p[3]
                conf_mat[emotion_cat][emotion_cat] += 1
                pair_id_all_set.remove(p)  # Remove matched pair
            else:
                emotion_cat = p[3]
                conf_mat[0][emotion_cat] += 1  # False positive
        
        # Add false negatives (remaining true pairs)
        for p in pair_id_all_set:
            emotion_cat = p[3]
            conf_mat[emotion_cat][0] += 1
        
        # Calculate precision, recall, F1
        p = np.diagonal(conf_mat / (np.reshape(np.sum(conf_mat, axis=0) + 1e-8, [1, 7])))
        r = np.diagonal(conf_mat / (np.reshape(np.sum(conf_mat, axis=1) + 1e-8, [7, 1])))
        f = 2 * p * r / (p + r + 1e-8)
        
        # Weighted average (exclude neutral class - index 0)
        weight0 = np.sum(conf_mat, axis=1)
        weight = weight0[1:] / (np.sum(weight0[1:]) + 1e-8)
        w_avg_p = np.sum(p[1:] * weight)
        w_avg_r = np.sum(r[1:] * weight)
        w_avg_f = np.sum(f[1:] * weight)

        # For 4-class evaluation (exclude disgust/fear as they have small counts)
        # ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        idx = [1, 4, 5, 6]  # anger, joy, sadness, surprise
        weight_part = weight0[idx] / (np.sum(weight0[idx]) + 1e-8)
        w_avg_p_part = np.sum(p[idx] * weight_part)
        w_avg_r_part = np.sum(r[idx] * weight_part)  
        w_avg_f_part = np.sum(f[idx] * weight_part)
        
        # Return format: [f1_emo1-6, w_avg_p, w_avg_r, w_avg_f, w_avg_p_part, w_avg_r_part, w_avg_f_part]
        results = list(f[1:]) + [w_avg_p, w_avg_r, w_avg_f, w_avg_p_part, w_avg_r_part, w_avg_f_part]
        return results
    
    return cal_prf_emocate_pairs(pair_id_all, pair_id_filtered) + cal_prf_emocate_pairs(pair_id_all, pair_id) + [keep_rate]

class Step2Metrics:
    """
    Step2 metrics tracker - pair-level emotion-cause relationship classification
    Following the original TensorFlow implementation
    """
    
    def __init__(self, use_emotion_categories: bool = False):
        self.use_emotion_categories = use_emotion_categories
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_pair_id_all = []
        self.all_pair_id = []
        self.all_pred_y = []
    
    def update(self, loss: float, 
               pair_id_all: List[Tuple],
               pair_id: List[Tuple], 
               pred_y: np.ndarray,
               **kwargs):
        """
        Update metrics with batch results
        
        Args:
            loss: Training loss
            pair_id_all: List of all true pairs for this batch
            pair_id: List of candidate pairs for this batch
            pred_y: [N] predicted labels (0/1) for candidate pairs
        """
        batch_size = len(pair_id)
        
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Store for later computation
        self.all_pair_id_all.extend(pair_id_all)
        self.all_pair_id.extend(pair_id)
        self.all_pred_y.extend(pred_y.tolist() if isinstance(pred_y, np.ndarray) else pred_y)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final Step2 metrics
        
        Returns:
            Dictionary with Step2 metrics
        """
        if self.total_samples == 0:
            return {'avg_loss': 0.0}
        
        pred_y = np.array(self.all_pred_y)
        
        if self.use_emotion_categories:
            # Use emotion-category-aware evaluation
            results = prf_2nd_step_emocate(self.all_pair_id_all, self.all_pair_id, pred_y)
            
            metrics = {
                'avg_loss': self.total_loss / self.total_samples,
                # Filtered results (predicted pairs only)
                'filtered_anger_f1': results[0],
                'filtered_disgust_f1': results[1], 
                'filtered_fear_f1': results[2],
                'filtered_joy_f1': results[3],
                'filtered_sadness_f1': results[4],
                'filtered_surprise_f1': results[5],
                'filtered_weighted_precision': results[6],
                'filtered_weighted_recall': results[7],
                'filtered_weighted_f1': results[8],  # Main metric for 6-class
                'filtered_weighted_precision_4class': results[9],
                'filtered_weighted_recall_4class': results[10],
                'filtered_weighted_f1_4class': results[11],  # Main metric for 4-class
                # Original results (all candidate pairs)
                'original_weighted_f1': results[20],  # Index 8 + 12
                'keep_rate': results[-1]
            }
        else:
            # Use simple pair evaluation
            results = prf_2nd_step(self.all_pair_id_all, self.all_pair_id, pred_y)
            
            metrics = {
                'avg_loss': self.total_loss / self.total_samples,
                'filtered_precision': results[0],
                'filtered_recall': results[1], 
                'filtered_f1': results[2],  # Main metric
                'original_precision': results[3],
                'original_recall': results[4],
                'original_f1': results[5],
                'keep_rate': results[6]
            }
        
        return metrics

def create_pair_dict(pair_list: List[Tuple], use_emotion_categories: bool = False) -> Dict[int, List]:
    """
    Create dictionary mapping doc_id to pairs
    
    Args:
        pair_list: List of pairs [(doc_id, emo_utt, cause_utt, emotion_cat)]
        use_emotion_categories: Whether to include emotion category strings
        
    Returns:
        Dictionary {doc_id: [pair_info, ...]}
    """
    emotion_idx_rev = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))
    pair_dict = defaultdict(list)
    
    for x in pair_list:
        if use_emotion_categories:
            # Include emotion category string
            tmp = x[1:3] + [emotion_idx_rev[x[3]]]
            pair_dict[x[0]].append(tmp)
        else:
            # Only utterance indices
            pair_dict[x[0]].append(x[1:-1])
    
    return pair_dict