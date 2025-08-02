"""
Step2 Dataset for MECPE: Emotion-Cause Pair Classification
Generates all possible emotion-cause pairs from Step1 predictions
"""
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import RobertaTokenizer
import json
import ast

class PairDataset(Dataset):
    """
    Step2 dataset that generates emotion-cause pairs from Step1 results
    Following the original TensorFlow data loading logic
    """
    
    def __init__(self, 
                 step1_result_file: str,
                 config,
                 tokenizer: Optional[RobertaTokenizer] = None,
                 split: str = "train"):
        """
        Args:
            step1_result_file: Path to Step1 output file with emotion/cause predictions
            config: Configuration object
            tokenizer: RoBERTa tokenizer for text encoding
            split: Dataset split (train/dev/test)
        """
        self.config = config
        self.split = split
        self.max_seq_len = config.data.max_seq_length
        self.pred_future_cause = config.data.pred_future_cause
        self.use_emotion_categories = config.model.use_emotion_categories
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
        else:
            self.tokenizer = tokenizer
        
        # Emotion category mapping
        self.emotion_idx = {
            'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 
            'joy': 4, 'sadness': 5, 'surprise': 6
        }
        
        # Load and process data
        self.pairs, self.pair_id_all, self.pair_id = self._load_step2_data(step1_result_file)
        
        print(f"Step2 {split} dataset loaded:")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Positive pairs: {sum(1 for p in self.pairs if p['label'] == 1)}")
        print(f"  Negative pairs: {sum(1 for p in self.pairs if p['label'] == 0)}")
    
    def _load_step2_data(self, step1_file: str) -> Tuple[List[Dict], List[Tuple], List[Tuple]]:
        """
        Load Step2 data from Step1 results following original logic
        """
        pairs = []
        pair_id_all = []  # All true emotion-cause pairs
        pair_id = []      # All candidate pairs (for evaluation)
        
        print(f"Loading Step2 data from: {step1_file}")
        
        with open(step1_file, 'r', encoding='utf-8') as f:
            while True:
                # Read document header
                line = f.readline()
                if not line:
                    break
                    
                doc_id, d_len = map(int, line.strip().split())
                
                # Read true emotion-cause pairs
                true_pairs_line = f.readline().strip()
                if true_pairs_line:
                    true_pairs = ast.literal_eval(true_pairs_line)
                    if true_pairs and len(true_pairs[0]) > 2:
                        # Remove duplicates if contains span indices
                        true_pairs = sorted(list(set([(p[0], p[1]) for p in true_pairs])))
                else:
                    true_pairs = []
                
                # Filter future causes if not allowed
                if not self.pred_future_cause:
                    true_pairs = [(p[0], p[1]) for p in true_pairs if p[1] <= p[0]]
                
                # Read utterances with predictions
                utterances = []
                emotion_preds = []
                cause_preds = []
                emotions = []
                
                for i in range(d_len):
                    line = f.readline().strip().split(' | ')
                    utt_id = int(line[0])
                    emotion_pred = int(line[1])
                    cause_pred = int(line[2])
                    speaker = line[3]
                    emotion_true = line[4]
                    text = line[5]
                    
                    utterances.append({
                        'utt_id': utt_id,
                        'text': text,
                        'speaker': speaker,
                        'emotion_true': emotion_true,
                        'emotion_pred': emotion_pred,
                        'cause_pred': cause_pred
                    })
                    
                    emotion_preds.append(emotion_pred)
                    cause_preds.append(cause_pred)
                    emotions.append(emotion_true)
                
                # Generate emotion-cause pairs following original logic
                emotion_utts = [i+1 for i, pred in enumerate(emotion_preds) if pred > 0]  # 1-indexed
                cause_utts = [i+1 for i, pred in enumerate(cause_preds) if pred > 0]     # 1-indexed
                
                # Add true pairs to pair_id_all
                for emo_utt, cause_utt in true_pairs:
                    if 1 <= emo_utt <= len(emotions) and 1 <= cause_utt <= len(emotions):
                        emotion_cat = self.emotion_idx.get(emotions[emo_utt-1], 0)
                        pair_id_all.append((doc_id, emo_utt, cause_utt, emotion_cat))
                
                # Generate all possible pairs from predicted emotion and cause utterances
                for emo_utt in emotion_utts:
                    for cause_utt in cause_utts:
                        # Check future cause constraint
                        if self.pred_future_cause or cause_utt <= emo_utt:
                            # Determine emotion category
                            if self.use_emotion_categories:
                                emotion_cat = self.emotion_idx.get(emotions[emo_utt-1], 0)
                            else:
                                emotion_cat = self.emotion_idx.get(emotions[emo_utt-1], 0)
                            
                            pair_tuple = (doc_id, emo_utt, cause_utt, emotion_cat)
                            pair_id.append(pair_tuple)
                            
                            # Check if this is a true pair
                            is_true_pair = (emo_utt, cause_utt) in true_pairs
                            
                            # Calculate distance (following original: j-i+100)
                            distance = cause_utt - emo_utt + 100
                            
                            # Create pair data
                            pair_data = {
                                'doc_id': doc_id,
                                'emotion_utt_id': emo_utt - 1,  # Convert to 0-indexed
                                'cause_utt_id': cause_utt - 1,   # Convert to 0-indexed
                                'emotion_text': utterances[emo_utt-1]['text'],
                                'cause_text': utterances[cause_utt-1]['text'],
                                'distance': distance,
                                'emotion_category': emotion_cat,
                                'label': 1 if is_true_pair else 0,
                                'pair_id': pair_tuple
                            }
                            
                            pairs.append(pair_data)
        
        return pairs, pair_id_all, pair_id
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single emotion-cause pair
        """
        pair = self.pairs[idx]
        
        # Tokenize both utterances
        emotion_encoding = self.tokenizer(
            pair['emotion_text'],
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        cause_encoding = self.tokenizer(
            pair['cause_text'],
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Stack input_ids and attention_mask for both utterances
        input_ids = torch.stack([
            emotion_encoding['input_ids'].squeeze(0),
            cause_encoding['input_ids'].squeeze(0)
        ])  # [2, max_seq_len]
        
        attention_mask = torch.stack([
            emotion_encoding['attention_mask'].squeeze(0),
            cause_encoding['attention_mask'].squeeze(0)
        ])  # [2, max_seq_len]
        
        # Prepare output
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'distance': torch.tensor(pair['distance'], dtype=torch.long),
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'doc_id': pair['doc_id'],
            'pair_id': pair['pair_id']
        }
        
        # Add emotion category if using emotion categories
        if self.use_emotion_categories:
            item['emotion_category'] = torch.tensor(pair['emotion_category'], dtype=torch.long)
        
        return item

def collate_pair_samples(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Step2 pair dataset
    """
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])          # [batch_size, 2, max_seq_len]
    attention_mask = torch.stack([item['attention_mask'] for item in batch]) # [batch_size, 2, max_seq_len]
    distance = torch.stack([item['distance'] for item in batch])             # [batch_size]
    labels = torch.stack([item['label'] for item in batch])                  # [batch_size]
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'distance': distance,
        'labels': labels,
        'doc_ids': [item['doc_id'] for item in batch],
        'pair_ids': [item['pair_id'] for item in batch]
    }
    
    # Add emotion category if present
    if 'emotion_category' in batch[0]:
        emotion_category = torch.stack([item['emotion_category'] for item in batch])
        result['emotion_category'] = emotion_category
    
    return result

def create_pair_datasets(config, tokenizer: Optional[RobertaTokenizer] = None):
    """
    Create Step2 datasets from Step1 results
    
    Args:
        config: Configuration object
        tokenizer: Optional RoBERTa tokenizer
        
    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    # Step1 result files (generated by Step1 training)
    step1_result_dir = os.path.join(config.training.save_dir, "step1_results")
    
    train_file = os.path.join(step1_result_dir, "train_predictions.txt")
    dev_file = os.path.join(step1_result_dir, "dev_predictions.txt") 
    test_file = os.path.join(step1_result_dir, "test_predictions.txt")
    
    # Check if Step1 results exist
    for file_path, split in [(train_file, "train"), (dev_file, "dev"), (test_file, "test")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Step1 result file not found: {file_path}. Please run Step1 training first.")
    
    # Create datasets
    train_dataset = PairDataset(train_file, config, tokenizer, "train")
    dev_dataset = PairDataset(dev_file, config, tokenizer, "dev")
    test_dataset = PairDataset(test_file, config, tokenizer, "test")
    
    return train_dataset, dev_dataset, test_dataset