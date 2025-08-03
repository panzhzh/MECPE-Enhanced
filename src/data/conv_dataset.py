"""
Modern ECF Dataset for multimodal emotion-cause pair extraction
"""
import os
import json
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from ..utils.config import Config

class ECFDataset(Dataset):
    """
    Modern ECF Dataset with multimodal support
    Handles text, audio, and video data end-to-end
    """
    
    def __init__(self, 
                 split: str = "train",
                 config: Config = None,
                 load_video: bool = True,
                 load_audio: bool = True):
        """
        Initialize ECF dataset
        
        Args:
            split: train/dev/test
            config: Configuration object
            load_video: Whether to load video features
            load_audio: Whether to load audio features
        """
        self.split = split
        self.config = config or Config()
        self.load_video = load_video
        self.load_audio = load_audio
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.text_model
        )
        
        # Load data
        self.data = self._load_ecf_data()
        self.meld_mapping = self._load_meld_mapping()
        
        print(f"Loaded {len(self.data)} conversations from {split} split")
        
        # Print summary statistics
        total_utterances = sum(len(conv['utterances']) for conv in self.data)
        conversations_with_pairs = sum(1 for conv in self.data if conv['emotion_cause_pairs'])
        total_pairs = sum(len(conv['emotion_cause_pairs']) for conv in self.data)
        
        print(f"  - Total utterances: {total_utterances}")
        print(f"  - Conversations with emotion-cause pairs: {conversations_with_pairs}/{len(self.data)}")
        print(f"  - Total emotion-cause pairs: {total_pairs}")
    
    def _load_ecf_data(self) -> List[Dict]:
        """Load ECF text data from files"""
        if self.split == "train":
            file_path = os.path.join(self.config.data.data_root, 
                                   self.config.data.ecf_train_file)
        elif self.split == "dev":
            file_path = os.path.join(self.config.data.data_root,
                                   self.config.data.ecf_dev_file)
        else:  # test
            file_path = os.path.join(self.config.data.data_root,
                                   self.config.data.ecf_test_file)
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            # Parse conversation header (format: "conv_id num_utterances")
            line = lines[i].strip()
            conv_info = line.split()
            if len(conv_info) < 2:
                i += 1
                continue
            
            try:
                conv_id = int(conv_info[0])
                num_utterances = int(conv_info[1])
            except ValueError:
                i += 1
                continue
            i += 1
            
            # Parse emotion-cause pairs (optional - some conversations might not have pairs)
            emotion_cause_pairs = []
            if i < len(lines) and lines[i].strip().startswith('('):
                pairs_line = lines[i].strip()
                emotion_cause_pairs = self._parse_pairs(pairs_line)
                i += 1
            
            # Parse utterances
            utterances = []
            utterances_parsed = 0
            while utterances_parsed < num_utterances and i < len(lines):
                utt_line = lines[i].strip() 
                if not utt_line:  # Skip empty lines
                    i += 1
                    continue
                if '|' in utt_line:
                    utterance = self._parse_utterance(utt_line)
                    if utterance:  # Only add valid utterances
                        utterances.append(utterance)
                        utterances_parsed += 1
                i += 1
            
            # Create conversation sample
            data.append({
                'conv_id': conv_id,
                'utterances': utterances,
                'emotion_cause_pairs': emotion_cause_pairs
            })
        
        return data
    
    def _parse_pairs(self, pairs_line: str) -> List[Tuple[int, int]]:
        """Parse emotion-cause pairs from string"""
        pairs = []
        # Remove outer parentheses and split by '),('
        pairs_str = pairs_line.strip('()')
        if not pairs_str:
            return pairs
            
        pair_strs = pairs_str.split('),(')
        for pair_str in pair_strs:
            if ',' in pair_str:
                try:
                    emo_utt, cause_utt = map(int, pair_str.split(','))
                    pairs.append((emo_utt, cause_utt))
                except ValueError:
                    continue
        return pairs
    
    def _parse_utterance(self, utt_line: str) -> Dict:
        """Parse single utterance from line"""
        parts = utt_line.split(' | ')
        if len(parts) >= 4:
            return {
                'utt_id': int(parts[0]),
                'speaker': parts[1],
                'emotion': parts[2],
                'text': parts[3],
                'timestamp': parts[4] if len(parts) > 4 else None
            }
        return {}
    
    def _load_meld_mapping(self) -> Dict:
        """Load ECF to MELD mapping"""
        mapping = {}
        mapping_file = self.config.data.ecf_meld_mapping
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            # Parse conversation header
            conv_info = lines[i].strip().split()
            if len(conv_info) >= 2:
                conv_id = int(conv_info[0])
                i += 1
                
                # Skip pairs line
                if i < len(lines) and lines[i].startswith('('):
                    i += 1
                
                # Parse utterances with MELD mapping
                mapping[conv_id] = {}
                for line_idx in range(i, len(lines)):
                    line = lines[line_idx].strip()
                    if not line or not line[0].isdigit():
                        i = line_idx
                        break
                        
                    parts = line.split(' | ')
                    if len(parts) >= 5:
                        utt_id = int(parts[0])
                        meld_ref = parts[-1]  # Last part is MELD reference
                        if 'dia' in meld_ref and 'utt' in meld_ref:
                            mapping[conv_id][utt_id] = meld_ref
                else:
                    break
            else:
                i += 1
        
        return mapping
    
    def _get_video_path(self, conv_id: int, utt_id: int) -> Optional[str]:
        """Get video file path for utterance"""
        if conv_id in self.meld_mapping and utt_id in self.meld_mapping[conv_id]:
            meld_ref = self.meld_mapping[conv_id][utt_id]
            
            # Extract split and video name
            if 'train_' in meld_ref:
                split_dir = 'train_splits_complete'
                video_name = meld_ref.replace('train_', '') + '.mp4'
            elif 'dev_' in meld_ref:
                split_dir = 'dev_splits_complete'  
                video_name = meld_ref.replace('dev_', '') + '.mp4'
            elif 'test_' in meld_ref:
                split_dir = 'test_splits_complete'
                video_name = meld_ref.replace('test_', '') + '.mp4'
            else:
                return None
                
            video_path = os.path.join(
                self.config.data.meld_root, split_dir, video_name
            )
            return video_path if os.path.exists(video_path) else None
        
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single conversation sample"""
        conv_data = self.data[idx]
        
        # Prepare text data
        utterances = conv_data['utterances']
        texts = [utt['text'] for utt in utterances]
        speakers = [utt['speaker'] for utt in utterances]
        emotions = [utt['emotion'] for utt in utterances]
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_seq_length,
            return_tensors="pt"
        )
        
        # Prepare labels for Conv (emotion/cause recognition)
        emotion_labels = []
        cause_labels = []
        
        for i, utt in enumerate(utterances):
            utt_id = utt['utt_id']
            
            # Check if this utterance is an emotion utterance
            is_emotion = any(pair[0] == utt_id for pair in conv_data['emotion_cause_pairs'])
            emotion_labels.append(1 if is_emotion else 0)
            
            # Check if this utterance is a cause utterance  
            is_cause = any(pair[1] == utt_id for pair in conv_data['emotion_cause_pairs'])
            cause_labels.append(1 if is_cause else 0)
        
        sample = {
            'conv_id': conv_data['conv_id'],
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'speakers': speakers,
            'emotions': emotions,
            'emotion_labels': torch.tensor(emotion_labels),
            'cause_labels': torch.tensor(cause_labels),
            'emotion_cause_pairs': conv_data['emotion_cause_pairs']
        }
        
        # Add video/audio paths for future processing
        if self.load_video or self.load_audio:
            video_paths = []
            for utt in utterances:
                video_path = self._get_video_path(conv_data['conv_id'], utt['utt_id'])
                video_paths.append(video_path)
            sample['video_paths'] = video_paths
        
        return sample

def collate_conversations(batch: List[Dict]) -> Dict:
    """
    Collate function for conversation-level batching
    """
    # This is a complex collate function that handles variable-length conversations
    # For now, we'll implement basic batching
    
    batch_size = len(batch)
    max_utterances = max(len(sample['input_ids']) for sample in batch)
    max_seq_length = max(sample['input_ids'].size(-1) for sample in batch)
    
    # Initialize batch tensors with proper padding
    input_ids = torch.zeros((batch_size, max_utterances, max_seq_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_utterances, max_seq_length), dtype=torch.long)
    emotion_labels = torch.zeros((batch_size, max_utterances), dtype=torch.long)
    cause_labels = torch.zeros((batch_size, max_utterances), dtype=torch.long)
    
    conv_ids = []
    all_speakers = []
    all_emotions = []
    all_pairs = []
    all_video_paths = []
    
    for i, sample in enumerate(batch):
        num_utts = len(sample['input_ids'])
        
        for j in range(num_utts):
            seq_len = sample['input_ids'][j].size(0)
            input_ids[i, j, :seq_len] = sample['input_ids'][j]
            attention_mask[i, j, :seq_len] = sample['attention_mask'][j]
        
        emotion_labels[i, :num_utts] = sample['emotion_labels']
        cause_labels[i, :num_utts] = sample['cause_labels']
        
        conv_ids.append(sample['conv_id'])
        all_speakers.append(sample['speakers'])
        all_emotions.append(sample['emotions'])
        all_pairs.append(sample['emotion_cause_pairs'])
        
        if 'video_paths' in sample:
            all_video_paths.append(sample['video_paths'])
    
    return {
        'conv_ids': conv_ids,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'emotion_labels': emotion_labels,
        'cause_labels': cause_labels,
        'speakers': all_speakers,
        'emotions': all_emotions,
        'emotion_cause_pairs': all_pairs,
        'video_paths': all_video_paths if all_video_paths else None
    }