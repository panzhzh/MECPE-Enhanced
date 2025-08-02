"""
Step1 evaluation metrics for MECPE
Emotion and cause utterance-level classification evaluation
Ported from archive/original_tensorflow/utils/pre_data_bert.py
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Dict, List, Tuple

def cal_prf(pred_y: np.ndarray, true_y: np.ndarray, doc_len: np.ndarray, average='binary') -> Tuple[float, float, float]:
    """
    Calculate precision, recall, F1 for binary classification (Step1 style)
    Original implementation from TensorFlow code
    
    Args:
        pred_y: [batch_size, max_doc_len] predicted labels (0/1)
        true_y: [batch_size, max_doc_len] true labels (0/1)
        doc_len: [batch_size] actual document lengths
        
    Returns:
        (precision, recall, f1)
    """
    pred_num, acc_num, true_num = 0, 0, 0
    
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            if pred_y[i][j]:
                pred_num += 1
            if true_y[i][j]:
                true_num += 1
            if pred_y[i][j] and true_y[i][j]:
                acc_num += 1
    
    p = acc_num / (pred_num + 1e-8)
    r = acc_num / (true_num + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    
    return p, r, f

def cal_prf_emocate(pred_y: np.ndarray, true_y: np.ndarray, doc_len: np.ndarray) -> np.ndarray:
    """
    Calculate precision, recall, F1 for emotion category classification
    Original implementation from TensorFlow code
    
    Args:
        pred_y: [batch_size, max_doc_len] predicted emotion categories (0-6)
        true_y: [batch_size, max_doc_len] true emotion categories (0-6)
        doc_len: [batch_size] actual document lengths
        
    Returns:
        Array of [f1_neutral, f1_anger, f1_disgust, f1_fear, f1_joy, f1_sadness, f1_surprise, weighted_avg_f1]
    """
    conf_mat = np.zeros([7, 7])
    
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            conf_mat[true_y[i][j]][pred_y[i][j]] += 1
    
    p = np.diagonal(conf_mat / (np.reshape(np.sum(conf_mat, axis=0) + 1e-8, [1, 7])))
    r = np.diagonal(conf_mat / (np.reshape(np.sum(conf_mat, axis=1) + 1e-8, [7, 1])))
    f = 2 * p * r / (p + r + 1e-8)
    
    weight = np.sum(conf_mat, axis=1) / np.sum(conf_mat)
    w_avg_f = np.sum(f * weight)
    
    return np.append(f, w_avg_f)

def get_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Get predictions from logits"""
    return torch.argmax(logits, dim=-1)

class Step1Metrics:
    """
    Step1 metrics tracker - utterance-level emotion and cause classification
    Following the original TensorFlow implementation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_emotion_preds = []
        self.all_cause_preds = []
        self.all_emotion_labels = []
        self.all_cause_labels = []
        self.all_doc_lens = []
    
    def update(self, loss: float,
               emotion_logits: torch.Tensor,
               cause_logits: torch.Tensor, 
               emotion_labels: torch.Tensor,
               cause_labels: torch.Tensor,
               attention_mask: torch.Tensor,
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
        """
        batch_size = emotion_labels.size(0)
        
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Get predictions
        emotion_preds = get_predictions(emotion_logits)  # [batch_size, max_utts]
        cause_preds = get_predictions(cause_logits)      # [batch_size, max_utts]
        
        # Calculate document lengths (number of non-padded utterances)
        if len(attention_mask.shape) == 3:
            # attention_mask: [batch_size, max_utts, max_tokens]
            doc_lens = (attention_mask.sum(dim=-1) > 0).sum(dim=-1)  # [batch_size]
        else:
            # attention_mask: [batch_size, max_utts]
            doc_lens = (attention_mask > 0).sum(dim=-1)  # [batch_size]
        
        # Store for later computation
        self.all_emotion_preds.append(emotion_preds.cpu().numpy())
        self.all_cause_preds.append(cause_preds.cpu().numpy())
        self.all_emotion_labels.append(emotion_labels.cpu().numpy())
        self.all_cause_labels.append(cause_labels.cpu().numpy())
        self.all_doc_lens.append(doc_lens.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final Step1 metrics
        
        Returns:
            Dictionary with Step1 metrics
        """
        if self.total_samples == 0:
            return {'avg_loss': 0.0}
        
        # Concatenate all batches - handle variable lengths by padding
        max_utts = max(arr.shape[1] for arr in self.all_emotion_preds)
        
        emotion_preds_padded = []
        cause_preds_padded = []
        emotion_labels_padded = []
        cause_labels_padded = []
        
        for i in range(len(self.all_emotion_preds)):
            ep, cp = self.all_emotion_preds[i], self.all_cause_preds[i]
            el, cl = self.all_emotion_labels[i], self.all_cause_labels[i]
            
            if ep.shape[1] < max_utts:
                pad_size = max_utts - ep.shape[1]
                ep = np.concatenate([ep, np.zeros((ep.shape[0], pad_size), dtype=ep.dtype)], axis=1)
                cp = np.concatenate([cp, np.zeros((cp.shape[0], pad_size), dtype=cp.dtype)], axis=1)
                el = np.concatenate([el, np.zeros((el.shape[0], pad_size), dtype=el.dtype)], axis=1)
                cl = np.concatenate([cl, np.zeros((cl.shape[0], pad_size), dtype=cl.dtype)], axis=1)
            
            emotion_preds_padded.append(ep)
            cause_preds_padded.append(cp)
            emotion_labels_padded.append(el)
            cause_labels_padded.append(cl)
        
        emotion_preds = np.concatenate(emotion_preds_padded, axis=0)
        cause_preds = np.concatenate(cause_preds_padded, axis=0)
        emotion_labels = np.concatenate(emotion_labels_padded, axis=0)
        cause_labels = np.concatenate(cause_labels_padded, axis=0)
        doc_lens = np.concatenate(self.all_doc_lens, axis=0)
        
        # Calculate Step1 metrics using original functions
        emo_p, emo_r, emo_f1 = cal_prf(emotion_preds, emotion_labels, doc_lens)
        cause_p, cause_r, cause_f1 = cal_prf(cause_preds, cause_labels, doc_lens)
        
        metrics = {
            'avg_loss': self.total_loss / self.total_samples,
            'emotion_precision': emo_p,
            'emotion_recall': emo_r,
            'emotion_f1': emo_f1,
            'cause_precision': cause_p,
            'cause_recall': cause_r,
            'cause_f1': cause_f1,
            'avg_f1': (emo_f1 + cause_f1) / 2.0
        }
        
        return metrics