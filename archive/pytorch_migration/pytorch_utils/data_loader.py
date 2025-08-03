"""
PyTorch版本的数据加载模块
替代原TensorFlow版本的数据处理
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import pickle
from typing import List, Dict, Tuple, Optional
from config import FLAGS


class ECFDataset(Dataset):
    """MECPE数据集类"""
    
    def __init__(self, data_file: str, tokenizer, word_idx: Dict, video_idx: Dict, 
                 spe_idx: Dict, is_conv: bool = True):
        self.is_conv = is_conv
        self.tokenizer = tokenizer
        self.word_idx = word_idx
        self.video_idx = video_idx
        self.spe_idx = spe_idx
        
        if is_conv:
            self.data = self._load_conv_data(data_file)
        else:
            self.data = self._load_step2_data(data_file)
    
    def _load_conv_data(self, data_file):
        """加载Conv数据（情感和原因识别）"""
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        emotion_map = {
            'neutral': 0, 'anger': 1, 'disgust': 2, 
            'fear': 3, 'joy': 4, 'sadness': 5, 'surprise': 6
        }
        
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            # 解析文档信息
            doc_info = lines[i].strip().split()
            if len(doc_info) < 2:
                i += 1
                continue
                
            try:
                doc_id, doc_len = doc_info[0], int(doc_info[1])
            except ValueError:
                i += 1
                continue
            
            # 解析情感-原因对
            i += 1
            if i >= len(lines):
                break
            pairs_line = lines[i].strip()
            true_pairs = []
            if pairs_line and pairs_line.startswith('('):
                # 解析形如 (3,1),(3,3),(4,1) 的格式
                try:
                    true_pairs = eval('[' + pairs_line.replace('(', '(').replace(')', ')') + ']')
                except:
                    true_pairs = []
            
            # 解析utterances
            utterances = []
            emotions = []
            causes = []
            speakers = []
            video_ids = []
            
            for j in range(doc_len):
                i += 1
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split(' | ')
                if len(parts) >= 4:
                    utt_id = parts[0]
                    speaker = parts[1]
                    emotion = parts[2]
                    text = parts[3]
                    
                    utterances.append(text.strip())
                    
                    # 情感标签处理
                    if FLAGS.choose_emocate:
                        emotions.append(emotion_map.get(emotion, 0))
                    else:
                        # 二元分类：是否为情感话语
                        is_emotion = any(pair[0] == int(utt_id) for pair in true_pairs)
                        emotions.append(1 if is_emotion else 0)
                    
                    # 原因标签处理
                    is_cause = any(pair[1] == int(utt_id) for pair in true_pairs)
                    causes.append(1 if is_cause else 0)
                    
                    speakers.append(self.spe_idx.get(speaker, 0))
                    
                    # 构建video_id
                    video_key = f"dia{doc_id}utt{utt_id}"
                    video_ids.append(self.video_idx.get(video_key, 0))
            
            if utterances:  # 只有有效的样本才添加
                # 转换为模型输入格式
                sample = self._process_conv_sample(
                    doc_id, utterances, emotions, causes, speakers, video_ids, true_pairs
                )
                data.append(sample)
            i += 1
        
        return data
    
    def _process_conv_sample(self, doc_id, utterances, emotions, causes, 
                             speakers, video_ids, true_pairs):
        """处理Conv样本"""
        # 文本编码
        if FLAGS.model_type == 'BiLSTM':
            # 使用词汇表编码
            encoded_text = self._encode_with_vocab(utterances)
        else:
            # 使用BERT tokenizer
            encoded_text = self._encode_with_bert(utterances)
        
        return {
            'doc_id': doc_id,
            'text': encoded_text,
            'emotions': torch.tensor(emotions, dtype=torch.long),
            'causes': torch.tensor(causes, dtype=torch.long),
            'speakers': torch.tensor(speakers, dtype=torch.long),
            'video_ids': torch.tensor(video_ids, dtype=torch.long),
            'true_pairs': true_pairs,
            'doc_len': len(utterances)
        }
    
    def _encode_with_vocab(self, utterances):
        """使用预训练词汇表编码"""
        encoded = []
        sen_lens = []
        
        for utt in utterances:
            words = utt.split()[:FLAGS.max_sen_len]
            word_ids = [self.word_idx.get(word, 0) for word in words]
            
            # Padding
            while len(word_ids) < FLAGS.max_sen_len:
                word_ids.append(0)
            
            encoded.append(word_ids)
            sen_lens.append(min(len(words), FLAGS.max_sen_len))
        
        # Doc level padding
        while len(encoded) < FLAGS.max_doc_len:
            encoded.append([0] * FLAGS.max_sen_len)
            sen_lens.append(0)
        
        return {
            'input_ids': torch.tensor(encoded[:FLAGS.max_doc_len], dtype=torch.long),
            'sen_lens': torch.tensor(sen_lens[:FLAGS.max_doc_len], dtype=torch.long)
        }
    
    def _encode_with_bert(self, utterances):
        """使用BERT tokenizer编码"""
        # 拼接所有utterances
        full_text = " [SEP] ".join(utterances)
        
        # BERT编码
        encoding = self.tokenizer(
            full_text,
            max_length=FLAGS.max_doc_len_bert,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 为每个utterance编码（用于sentence-level BERT）
        utt_encodings = []
        for utt in utterances[:FLAGS.max_doc_len]:
            utt_enc = self.tokenizer(
                utt,
                max_length=FLAGS.max_sen_len_bert,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            utt_encodings.append({
                'input_ids': utt_enc['input_ids'].squeeze(0),
                'attention_mask': utt_enc['attention_mask'].squeeze(0)
            })
        
        # Padding to max_doc_len
        while len(utt_encodings) < FLAGS.max_doc_len:
            utt_encodings.append({
                'input_ids': torch.zeros(FLAGS.max_sen_len_bert, dtype=torch.long),
                'attention_mask': torch.zeros(FLAGS.max_sen_len_bert, dtype=torch.long)
            })
        
        return {
            'doc_input_ids': encoding['input_ids'].squeeze(0),
            'doc_attention_mask': encoding['attention_mask'].squeeze(0),
            'utt_input_ids': torch.stack([enc['input_ids'] for enc in utt_encodings]),
            'utt_attention_mask': torch.stack([enc['attention_mask'] for enc in utt_encodings])
        }
    
    def _load_step2_data(self, data_file):
        """加载Step2数据（情感-原因配对）"""
        # 临时实现：基于Conv格式生成Step2数据
        # 在实际应用中，应该使用Conv的预测结果
        data = []
        
        # 首先加载原始数据（和Conv相同）
        conv_data = self._load_conv_data(data_file)
        
        # 转换为Step2格式
        for sample in conv_data:
            doc_id = sample['doc_id']
            # 模拟Conv预测结果（这里使用真实标签作为"预测"）
            pred_emotions = sample['emotions'].clone()
            pred_causes = sample['causes'].clone()
            
            # 获取其他信息
            utterances = []  # 需要从原始文件重新读取文本
            true_emotions = sample['emotions'].clone()
            video_ids = sample['video_ids']
            speakers = sample['speakers']
            true_pairs = sample['true_pairs']
            
            # 重新读取文本数据
            utterances = self._get_utterances_from_doc(data_file, doc_id)
            
            if utterances:
                # 生成候选配对
                pairs_data = self._generate_step2_pairs(
                    doc_id, utterances, pred_emotions.tolist(), pred_causes.tolist(),
                    speakers.tolist(), true_emotions.tolist(), video_ids.tolist(), true_pairs
                )
                data.extend(pairs_data)
        
        return data
    
    def _get_utterances_from_doc(self, data_file, target_doc_id):
        """从数据文件中提取指定文档的utterances"""
        utterances = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            doc_info = lines[i].strip().split()
            if len(doc_info) < 2:
                i += 1
                continue
                
            try:
                doc_id, doc_len = doc_info[0], int(doc_info[1])
            except ValueError:
                i += 1
                continue
            
            if doc_id == target_doc_id:
                # 跳过pairs行
                i += 2
                
                # 读取utterances
                for j in range(doc_len):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split(' | ')
                    if len(parts) >= 4:
                        text = parts[3]
                        utterances.append(text.strip())
                    i += 1
                break
            else:
                # 跳过这个文档
                i += 2 + doc_len
        
        return utterances
    
    def _generate_step2_pairs(self, doc_id, utterances, pred_emotions, pred_causes, 
                             speakers, true_emotions, video_ids, true_pairs):
        """生成Step2的候选配对数据"""
        pairs_data = []
        
        # 找到所有预测为情感的话语
        emotion_indices = [i for i, emo in enumerate(pred_emotions) if emo == 1]
        
        # 对每个情感话语，生成与所有可能原因话语的配对
        for emo_idx in emotion_indices:
            for cau_idx in range(len(utterances)):
                # Skip future causes if not allowed
                if not FLAGS.pred_future_cause and cau_idx > emo_idx:
                    continue
                
                # 计算距离特征 (相对位置)
                distance = cau_idx - emo_idx + 100  # 加100确保为正数，原版本使用66-134的范围
                
                # 判断是否为真实配对
                is_true_pair = any(pair[0] == emo_idx + 1 and pair[1] == cau_idx + 1 for pair in true_pairs)
                
                # 创建配对样本
                pair_sample = self._create_step2_sample(
                    doc_id, emo_idx, cau_idx, utterances, speakers, 
                    true_emotions, video_ids, distance, is_true_pair
                )
                pairs_data.append(pair_sample)
        
        return pairs_data
    
    def _create_step2_sample(self, doc_id, emo_idx, cau_idx, utterances, speakers, 
                            true_emotions, video_ids, distance, is_true_pair):
        """创建Step2的单个配对样本"""
        # 提取情感话语和原因话语
        emo_text = utterances[emo_idx]
        cau_text = utterances[cau_idx]
        
        # 文本编码 (两个话语)
        if FLAGS.model_type == 'BiLSTM':
            # 使用词汇表编码
            encoded_texts = self._encode_pair_with_vocab([emo_text, cau_text])
        else:
            # 使用BERT tokenizer  
            encoded_texts = self._encode_pair_with_bert([emo_text, cau_text])
        
        return {
            'doc_id': f"{doc_id}_{emo_idx}_{cau_idx}",
            'text': encoded_texts,
            'distance': torch.tensor(distance, dtype=torch.long),
            'emotion_category': torch.tensor(true_emotions[emo_idx], dtype=torch.long),
            'video_ids': torch.tensor([video_ids[emo_idx], video_ids[cau_idx]], dtype=torch.long),
            'label': torch.tensor(1 if is_true_pair else 0, dtype=torch.long),
            'pair_info': [doc_id, emo_idx + 1, cau_idx + 1, true_emotions[emo_idx]]  # 用于评估
        }
    
    def _encode_pair_with_vocab(self, texts):
        """使用预训练词汇表编码话语对"""
        encoded_pairs = []
        sen_lens = []
        
        for text in texts:
            words = text.split()[:FLAGS.max_sen_len]
            word_ids = [self.word_idx.get(word, 0) for word in words]
            
            # Padding
            while len(word_ids) < FLAGS.max_sen_len:
                word_ids.append(0)
            
            encoded_pairs.append(word_ids)
            sen_lens.append(min(len(words), FLAGS.max_sen_len))
        
        return {
            'input_ids': torch.tensor(encoded_pairs, dtype=torch.long),  # [2, max_sen_len]
            'sen_lens': torch.tensor(sen_lens, dtype=torch.long)  # [2]
        }
    
    def _encode_pair_with_bert(self, texts):
        """使用BERT tokenizer编码话语对"""
        # 为每个话语编码
        encodings = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                max_length=FLAGS.max_sen_len_bert,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
        
        return {
            'input_ids': torch.stack([enc['input_ids'] for enc in encodings]),
            'attention_mask': torch.stack([enc['attention_mask'] for enc in encodings])
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_embeddings():
    """加载预训练嵌入"""
    # 加载GloVe词嵌入
    word_idx = {}
    word_embedding = []
    
    with open(FLAGS.w2v_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header if exists
                continue
            parts = line.strip().split()
            if len(parts) > FLAGS.embedding_dim:
                word = ' '.join(parts[:-FLAGS.embedding_dim])
                embedding = [float(x) for x in parts[-FLAGS.embedding_dim:]]
                word_idx[word] = len(word_idx)
                word_embedding.append(embedding)
    
    word_embedding = np.array(word_embedding, dtype=np.float32)
    
    # 加载多模态嵌入
    video_embedding = np.load(FLAGS.video_emb_file)
    audio_embedding = np.load(FLAGS.audio_emb_file)
    video_idx = np.load(FLAGS.video_idx_file, allow_pickle=True).item()
    
    return {
        'word_idx': word_idx,
        'word_embedding': word_embedding,
        'video_idx': video_idx,
        'video_embedding': video_embedding,
        'audio_embedding': audio_embedding
    }


def load_speaker_dict(data_path: str):
    """加载说话人字典"""
    spe_idx = {}
    spe_idx_rev = {}
    
    # 从数据文件中提取说话人信息
    files = ['train.txt', 'dev.txt', 'test.txt']
    speakers = set()
    
    for file in files:
        try:
            with open(f"{data_path}/{file}", 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue
                
                doc_info = lines[i].strip().split()
                doc_len = int(doc_info[1])
                i += 2  # Skip pairs line
                
                for j in range(doc_len):
                    parts = lines[i].strip().split(' | ')
                    if len(parts) >= 4:
                        speaker = parts[3]
                        speakers.add(speaker)
                    i += 1
        except FileNotFoundError:
            continue
    
    # 构建映射
    for i, speaker in enumerate(sorted(speakers)):
        spe_idx[speaker] = i
        spe_idx_rev[i] = speaker
    
    return spe_idx, spe_idx_rev


def create_dataloader(dataset, batch_size=None, shuffle=True):
    """创建DataLoader"""
    if batch_size is None:
        batch_size = FLAGS.batch_size
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # 可以根据需要调整
    )


def collate_fn(batch):
    """批处理函数 - 处理变长序列"""
    batch_data = {}
    
    # 检查是否是Conv数据（有doc_len字段）
    is_conv = 'doc_len' in batch[0]
    
    if is_conv:
        # Conv数据处理 - 使用配置的最大文档长度确保一致性
        max_doc_len = FLAGS.max_doc_len
        
        for key in batch[0].keys():
            if key in ['doc_id', 'true_pairs']:
                # 这些字段直接作为列表保存
                batch_data[key] = [item[key] for item in batch]
            elif key == 'text':
                # text字段是字典，需要特殊处理
                batch_data[key] = {}
                for text_key in batch[0][key].keys():
                    if text_key == 'input_ids':
                        # 对input_ids进行padding
                        padded_texts = []
                        for item in batch:
                            text_tensor = item[key][text_key]
                            current_len = text_tensor.size(0)
                            if current_len < max_doc_len:
                                # Padding with zeros
                                pad_size = max_doc_len - current_len
                                padding = torch.zeros(pad_size, text_tensor.size(1), dtype=text_tensor.dtype)
                                text_tensor = torch.cat([text_tensor, padding], dim=0)
                            padded_texts.append(text_tensor)
                        batch_data[key][text_key] = torch.stack(padded_texts)
                    elif text_key == 'sen_lens':
                        # 对sen_lens进行padding
                        padded_lens = []
                        for item in batch:
                            len_tensor = item[key][text_key]
                            current_len = len_tensor.size(0)
                            if current_len < max_doc_len:
                                # Padding with zeros
                                pad_size = max_doc_len - current_len
                                padding = torch.zeros(pad_size, dtype=len_tensor.dtype)
                                len_tensor = torch.cat([len_tensor, padding], dim=0)
                            padded_lens.append(len_tensor)
                        batch_data[key][text_key] = torch.stack(padded_lens)
                    else:
                        # 其他text相关字段
                        batch_data[key][text_key] = torch.stack([item[key][text_key] for item in batch])
            elif key == 'doc_len':
                # doc_len是标量，转换为tensor
                batch_data[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
            elif key in ['emotions', 'causes', 'speakers', 'video_ids']:
                # 这些字段需要padding到相同长度
                padded_tensors = []
                for item in batch:
                    tensor = item[key]
                    current_len = tensor.size(0)
                    if current_len < max_doc_len:
                        # Padding with zeros
                        pad_size = max_doc_len - current_len
                        padding = torch.zeros(pad_size, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(tensor)
                batch_data[key] = torch.stack(padded_tensors)
            else:
                # 其他tensor字段直接stack（如果维度匹配）
                try:
                    batch_data[key] = torch.stack([item[key] for item in batch])
                except:
                    # 如果stack失败，作为列表保存
                    batch_data[key] = [item[key] for item in batch]
    else:
        # Step2数据处理 - 固定维度，直接stack
        for key in batch[0].keys():
            if key in ['doc_id', 'pair_info']:
                # 这些字段直接作为列表保存
                batch_data[key] = [item[key] for item in batch]
            elif key == 'text':
                # Step2的text字段处理
                batch_data[key] = {}
                for text_key in batch[0][key].keys():
                    batch_data[key][text_key] = torch.stack([item[key][text_key] for item in batch])
            else:
                # 其他tensor字段直接stack
                try:
                    batch_data[key] = torch.stack([item[key] for item in batch])
                except:
                    # 如果stack失败，作为列表保存
                    batch_data[key] = [item[key] for item in batch]
    
    return batch_data


def get_tokenizer():
    """获取tokenizer"""
    if FLAGS.model_type == 'BiLSTM':
        return None  # BiLSTM不需要特殊tokenizer
    else:
        return AutoTokenizer.from_pretrained(FLAGS.bert_model_name)