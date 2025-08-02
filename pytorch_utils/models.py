"""
PyTorch版本的模型实现
包含Step1的情感和原因识别模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple, Optional
from config import FLAGS


class BiLSTMEncoder(nn.Module):
    """BiLSTM编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if dropout > 0 and num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) 实际序列长度
        Returns:
            output: (batch_size, seq_len, 2*hidden_dim)
        """
        if lengths is not None:
            # 确保长度至少为1，避免pack_padded_sequence错误
            lengths = torch.clamp(lengths, min=1)
            # 打包序列以处理不同长度
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, _ = self.lstm(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return self.dropout(output)


class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) 掩码，1表示有效位置
        Returns:
            attended: (batch_size, input_dim)
            attention_weights: (batch_size, seq_len)
        """
        # 计算注意力分数
        scores = self.w2(self.tanh(self.w1(x)))  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        attended = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return attended, attention_weights


class MultimodalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, text_dim: int, video_dim: int, audio_dim: int, 
                 output_dim: int, use_video: bool = True, use_audio: bool = True):
        super().__init__()
        self.use_video = use_video
        self.use_audio = use_audio
        
        # 维度变换
        input_dim = text_dim
        if use_video:
            self.video_proj = nn.Sequential(
                nn.Linear(video_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            )
            input_dim += output_dim
            
        if use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, output_dim),
                nn.ReLU(), 
                nn.LayerNorm(output_dim)
            )
            input_dim += output_dim
        
        self.fusion = nn.Linear(input_dim, output_dim)
        
    def forward(self, text_features, video_features=None, audio_features=None):
        """
        Args:
            text_features: (batch_size, seq_len, text_dim)
            video_features: (batch_size, seq_len, video_dim)
            audio_features: (batch_size, seq_len, audio_dim)
        """
        features = [text_features]
        
        if self.use_video and video_features is not None:
            video_proj = self.video_proj(video_features)
            features.append(video_proj)
            
        if self.use_audio and audio_features is not None:
            audio_proj = self.audio_proj(audio_features)
            features.append(audio_proj)
        
        # 拼接特征
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)


class BERTEncoder(nn.Module):
    """BERT编码器"""
    
    def __init__(self, model_name: str, hidden_dropout: float = 0.1, 
                 attention_dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout
        config.attention_probs_dropout_prob = attention_dropout
        
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state, outputs.pooler_output


class EmotionCauseModel(nn.Module):
    """Step1: 情感和原因识别模型"""
    
    def __init__(self, embeddings_dict: Dict, config=None):
        super().__init__()
        if config is None:
            config = FLAGS
        
        self.config = config
        self.use_video = config.use_x_v
        self.use_audio = config.use_x_a
        
        # 嵌入层
        word_embedding = embeddings_dict['word_embedding']
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embedding), freeze=False
        )
        
        if self.use_video:
            video_embedding = embeddings_dict['video_embedding']
            self.video_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(video_embedding), freeze=False
            )
            
        if self.use_audio:
            audio_embedding = embeddings_dict['audio_embedding']
            self.audio_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(audio_embedding), freeze=False
            )
        
        # 文本编码器
        if config.model_type == 'BiLSTM':
            self.text_encoder = self._build_bilstm_encoder()
        else:
            self.text_encoder = self._build_bert_encoder()
        
        # 多模态融合
        text_dim = 2 * config.n_hidden if config.model_type == 'BiLSTM' else 768
        video_dim = video_embedding.shape[1] if self.use_video else 0
        audio_dim = audio_embedding.shape[1] if self.use_audio else 0
        
        self.fusion = MultimodalFusion(
            text_dim=text_dim,
            video_dim=video_dim,
            audio_dim=audio_dim,
            output_dim=2 * config.n_hidden,
            use_video=self.use_video,
            use_audio=self.use_audio
        )
        
        # 预测头
        pred_dim = 7 if config.choose_emocate else config.n_class
        
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(config.dropout2),
            nn.Linear(2 * config.n_hidden, pred_dim)
        )
        
        self.cause_classifier = nn.Sequential(
            nn.Dropout(config.dropout2),
            nn.Linear(2 * config.n_hidden, config.n_class)
        )
        
        # 多模态预测（用于辅助训练）
        if self.use_video:
            self.video_emotion_classifier = nn.Sequential(
                nn.Dropout(config.dropout2),
                nn.Linear(2 * config.n_hidden, pred_dim)
            )
            
        if self.use_audio:
            self.audio_emotion_classifier = nn.Sequential(
                nn.Dropout(config.dropout2),
                nn.Linear(2 * config.n_hidden, pred_dim)
            )
    
    def _build_bilstm_encoder(self):
        """构建BiLSTM编码器"""
        word_encoder = BiLSTMEncoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.n_hidden,
            dropout=self.config.dropout1
        )
        
        word_attention = AttentionLayer(2 * self.config.n_hidden)
        
        sentence_encoder = BiLSTMEncoder(
            input_dim=2 * self.config.n_hidden,
            hidden_dim=self.config.n_hidden,
            dropout=0.0
        )
        
        return nn.ModuleDict({
            'word_encoder': word_encoder,
            'word_attention': word_attention,
            'sentence_encoder': sentence_encoder
        })
    
    def _build_bert_encoder(self):
        """构建BERT编码器"""
        return BERTEncoder(
            model_name=self.config.bert_model_name,
            hidden_dropout=self.config.bert_hidden_dropout,
            attention_dropout=self.config.bert_attention_dropout
        )
    
    def forward(self, batch):
        """前向传播"""
        if self.config.model_type == 'BiLSTM':
            return self._forward_bilstm(batch)
        else:
            return self._forward_bert(batch)
    
    def _forward_bilstm(self, batch):
        """BiLSTM前向传播"""
        input_ids = batch['text']['input_ids']  # (batch_size, max_doc_len, max_sen_len)
        sen_lens = batch['text']['sen_lens']    # (batch_size, max_doc_len)
        doc_lens = batch['doc_len']             # (batch_size,)
        video_ids = batch['video_ids']          # (batch_size, max_doc_len)
        
        batch_size, max_doc_len, max_sen_len = input_ids.shape
        
        # 词嵌入
        word_embeds = self.word_embedding(input_ids)  # (batch_size, max_doc_len, max_sen_len, embed_dim)
        word_embeds = F.dropout(word_embeds, p=self.config.dropout1, training=self.training)
        
        # 重塑为句子级处理
        word_embeds = word_embeds.view(-1, max_sen_len, self.config.embedding_dim)
        sen_lens_flat = sen_lens.view(-1)
        
        # 词级编码
        word_encoded = self.text_encoder['word_encoder'](word_embeds, sen_lens_flat)
        
        # 词级注意力
        actual_max_len = word_encoded.size(1)  # 实际的序列长度
        sen_mask = self._create_mask(sen_lens_flat, actual_max_len)
        sentence_repr, _ = self.text_encoder['word_attention'](word_encoded, sen_mask)
        
        # 重塑回文档级
        sentence_repr = sentence_repr.view(batch_size, max_doc_len, -1)
        
        # 获取多模态特征
        video_features = None
        audio_features = None
        
        if self.use_video:
            video_features = self.video_embedding(video_ids)
            video_features = F.dropout(video_features, p=self.config.dropout_v, training=self.training)
            
        if self.use_audio:
            audio_features = self.audio_embedding(video_ids)  # 使用相同的索引
            audio_features = F.dropout(audio_features, p=self.config.dropout_a, training=self.training)
        
        # 多模态融合
        fused_features = self.fusion(sentence_repr, video_features, audio_features)
        
        # 句子级编码
        doc_mask = self._create_mask(doc_lens, max_doc_len)
        sentence_encoded = self.text_encoder['sentence_encoder'](fused_features, doc_lens)
        
        # 预测
        emotion_logits = self.emotion_classifier(sentence_encoded)
        cause_logits = self.cause_classifier(sentence_encoded)
        
        results = {
            'emotion_logits': emotion_logits,
            'cause_logits': cause_logits
        }
        
        # 辅助预测 - 使用独立的多模态特征，并通过sentence_encoder保持一致性
        if self.use_video and hasattr(self, 'video_emotion_classifier'):
            # 对视频特征进行投影，然后通过sentence encoder
            video_proj = self.fusion.video_proj(video_features)
            video_encoded = self.text_encoder['sentence_encoder'](video_proj, doc_lens)
            video_emotion_logits = self.video_emotion_classifier(video_encoded)
            results['video_emotion_logits'] = video_emotion_logits
            
        if self.use_audio and hasattr(self, 'audio_emotion_classifier'):
            # 对音频特征进行投影，然后通过sentence encoder
            audio_proj = self.fusion.audio_proj(audio_features)
            audio_encoded = self.text_encoder['sentence_encoder'](audio_proj, doc_lens)
            audio_emotion_logits = self.audio_emotion_classifier(audio_encoded)
            results['audio_emotion_logits'] = audio_emotion_logits
        
        return results
    
    def _forward_bert(self, batch):
        """BERT前向传播"""
        # 实现BERT版本的前向传播
        # 这里先提供基础框架
        doc_input_ids = batch['text']['doc_input_ids']
        doc_attention_mask = batch['text']['doc_attention_mask']
        
        # BERT编码
        sequence_output, pooled_output = self.text_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask
        )
        
        # 后续处理...
        # 这里需要根据具体需求实现句子级提取和多模态融合
        
        return {
            'emotion_logits': torch.zeros(1),  # 占位符
            'cause_logits': torch.zeros(1)
        }
    
    def _create_mask(self, lengths, max_len):
        """创建mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device)
        mask = mask < lengths.unsqueeze(1)
        return mask.float()


class EmotionCauseLoss(nn.Module):
    """Step1的损失函数"""
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = FLAGS
        
        self.config = config
        self.emotion_weight = config.emo_weight
        self.cause_weight = config.cause_weight
        self.l2_reg = config.l2_reg
        
        if config.choose_emocate:
            self.emotion_criterion = nn.CrossEntropyLoss()
        else:
            self.emotion_criterion = nn.CrossEntropyLoss()
        
        self.cause_criterion = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets, model):
        """
        计算损失
        Args:
            predictions: 模型预测结果字典
            targets: 目标标签字典 (includes doc_len for masking)
            model: 模型实例（用于L2正则化）
        """
        # 获取有效的序列长度信息
        if 'doc_len' in targets:
            doc_lens = targets['doc_len']
            max_len = predictions['emotion_logits'].size(1)
            
            # 创建mask来只计算有效位置的损失
            batch_size = doc_lens.size(0)
            mask = torch.arange(max_len).expand(batch_size, max_len).to(doc_lens.device)
            mask = mask < doc_lens.unsqueeze(1)  # (batch_size, max_len)
            
            # 应用mask到预测和目标
            emotion_pred_masked = predictions['emotion_logits'][mask]  # (valid_positions, n_class)
            cause_pred_masked = predictions['cause_logits'][mask]     # (valid_positions, n_class)
            emotion_target_masked = targets['emotions'][:, :max_len][mask]  # (valid_positions,)
            cause_target_masked = targets['causes'][:, :max_len][mask]      # (valid_positions,)
            
            # 主要损失
            emotion_loss = self.emotion_criterion(emotion_pred_masked, emotion_target_masked)
            cause_loss = self.cause_criterion(cause_pred_masked, cause_target_masked)
            
            # 辅助损失
            if 'video_emotion_logits' in predictions:
                video_pred_masked = predictions['video_emotion_logits'][mask]
                video_emotion_loss = self.emotion_criterion(video_pred_masked, emotion_target_masked)
                emotion_loss += video_emotion_loss
            
            if 'audio_emotion_logits' in predictions:
                audio_pred_masked = predictions['audio_emotion_logits'][mask]
                audio_emotion_loss = self.emotion_criterion(audio_pred_masked, emotion_target_masked)
                emotion_loss += audio_emotion_loss
                
        else:
            # 如果没有doc_len信息，使用原来的方式
            emotion_loss = self.emotion_criterion(
                predictions['emotion_logits'].view(-1, predictions['emotion_logits'].size(-1)),
                targets['emotions'].view(-1)
            )
            
            cause_loss = self.cause_criterion(
                predictions['cause_logits'].view(-1, predictions['cause_logits'].size(-1)),
                targets['causes'].view(-1)
            )
        
        total_loss = (self.emotion_weight * emotion_loss + 
                     self.cause_weight * cause_loss)
        
        # L2正则化
        if self.l2_reg > 0:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            total_loss += self.l2_reg * l2_loss
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'cause_loss': cause_loss
        }


class EmotionCausePairModel(nn.Module):
    """Step2: 情感-原因配对模型"""
    
    def __init__(self, embeddings_dict: Dict, config=None):
        super().__init__()
        if config is None:
            config = FLAGS
        
        self.config = config
        self.use_video = config.use_x_v
        self.use_audio = config.use_x_a
        
        # 嵌入层
        word_embedding = embeddings_dict['word_embedding']
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embedding), freeze=False
        )
        
        # 位置嵌入 (用于距离和情感类别)
        self.pos_embedding = nn.Embedding(200, config.embedding_dim_pos)  # 距离范围: 0-200
        self.emotion_embedding = nn.Embedding(7, config.embedding_dim_pos)  # 7种情感
        
        if self.use_video:
            video_embedding = embeddings_dict['video_embedding']
            self.video_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(video_embedding), freeze=False
            )
            
        if self.use_audio:
            audio_embedding = embeddings_dict['audio_embedding']
            self.audio_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(audio_embedding), freeze=False
            )
        
        # 文本编码器 (处理话语对)
        self.text_encoder = self._build_text_encoder()
        
        # 多模态特征变换
        h2 = 2 * config.n_hidden
        if self.use_video:
            self.video_proj = nn.Sequential(
                nn.Linear(video_embedding.shape[1], h2),
                nn.ReLU(),
                nn.LayerNorm(h2)
            )
            
        if self.use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_embedding.shape[1], h2),
                nn.ReLU(),
                nn.LayerNorm(h2)
            )
        
        # 特征融合和分类
        self._build_classifier()
    
    def _build_text_encoder(self):
        """构建文本编码器"""
        return nn.ModuleDict({
            'word_encoder': BiLSTMEncoder(
                input_dim=self.config.embedding_dim,
                hidden_dim=self.config.n_hidden,
                dropout=self.config.dropout1
            ),
            'word_attention': AttentionLayer(2 * self.config.n_hidden)
        })
    
    def _build_classifier(self):
        """构建分类器"""
        h2 = 2 * self.config.n_hidden
        
        # 计算每个话语的特征维度 - BiLSTM输出
        utterance_feature_dim = h2  # 每个话语的BiLSTM特征维度
        
        # 计算多模态特征维度（每个话语）
        if self.use_video:
            utterance_feature_dim += h2  # 视频特征投影后的维度
        if self.use_audio:
            utterance_feature_dim += h2  # 音频特征投影后的维度
        
        # 两个话语的总特征维度
        pair_feature_dim = 2 * utterance_feature_dim
        
        # 添加距离和情感特征
        total_feature_dim = pair_feature_dim + self.config.embedding_dim_pos  # 距离特征
        if self.config.choose_emocate:
            total_feature_dim += self.config.embedding_dim_pos  # 情感类别特征
        
        # 分类器 - 去掉Softmax，在损失函数中使用CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout2),
            nn.Linear(total_feature_dim, self.config.n_class)
        )
    
    def forward(self, batch):
        """前向传播"""
        input_ids = batch['text']['input_ids']  # (batch_size, 2, max_sen_len)
        sen_lens = batch['text']['sen_lens']    # (batch_size, 2)
        distance = batch['distance']            # (batch_size,)
        emotion_category = batch['emotion_category']  # (batch_size,)
        video_ids = batch['video_ids']          # (batch_size, 2)
        
        batch_size = input_ids.size(0)
        
        # 文本编码 - 处理话语对
        # 重塑为 (batch_size * 2, max_sen_len) 进行并行处理
        input_ids_flat = input_ids.view(-1, input_ids.size(-1))
        sen_lens_flat = sen_lens.view(-1)
        
        # 词嵌入
        word_embeds = self.word_embedding(input_ids_flat)
        word_embeds = F.dropout(word_embeds, p=self.config.dropout1, training=self.training)
        
        # 词级编码
        word_encoded = self.text_encoder['word_encoder'](word_embeds, sen_lens_flat)
        
        # 词级注意力
        actual_max_len = word_encoded.size(1)
        sen_mask = self._create_mask(sen_lens_flat, actual_max_len)
        utterance_repr, _ = self.text_encoder['word_attention'](word_encoded, sen_mask)
        
        # 重塑回 (batch_size, 2, hidden_dim)
        utterance_repr = utterance_repr.view(batch_size, 2, -1)
        
        # 多模态特征
        multimodal_features = []
        
        if self.use_video:
            video_features = self.video_embedding(video_ids)  # (batch_size, 2, video_dim)
            video_features = F.dropout(video_features, p=self.config.dropout_v, training=self.training)
            video_features = self.video_proj(video_features)  # (batch_size, 2, h2)
            multimodal_features.append(video_features)
            
        if self.use_audio:
            audio_features = self.audio_embedding(video_ids)  # (batch_size, 2, audio_dim)
            audio_features = F.dropout(audio_features, p=self.config.dropout_a, training=self.training)
            audio_features = self.audio_proj(audio_features)  # (batch_size, 2, h2)
            multimodal_features.append(audio_features)
        
        # 融合文本和多模态特征
        if multimodal_features:
            # 将多模态特征与文本特征拼接
            all_features = [utterance_repr] + multimodal_features
            fused_repr = torch.cat(all_features, dim=-1)  # (batch_size, 2, combined_dim)
        else:
            fused_repr = utterance_repr
        
        # 将两个话语的表示拼接
        pair_repr = fused_repr.view(batch_size, -1)  # (batch_size, 2 * combined_dim)
        
        # 添加距离特征
        distance_feat = self.pos_embedding(distance)  # (batch_size, pos_dim)
        features = [pair_repr, distance_feat]
        
        # 添加情感类别特征
        if self.config.choose_emocate:
            emotion_feat = self.emotion_embedding(emotion_category)  # (batch_size, pos_dim)
            features.append(emotion_feat)
        
        # 最终特征拼接
        final_features = torch.cat(features, dim=1)  # (batch_size, total_dim)
        
        # 分类
        logits = self.classifier(final_features)  # (batch_size, n_class)
        
        return {
            'pair_logits': logits
        }
    
    def _create_mask(self, lengths, max_len):
        """创建mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device)
        mask = mask < lengths.unsqueeze(1)
        return mask.float()


class EmotionCausePairLoss(nn.Module):
    """Step2的损失函数"""
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = FLAGS
        
        self.config = config
        self.l2_reg = config.l2_reg
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets, model):
        """
        计算损失
        Args:
            predictions: 模型预测结果字典
            targets: 目标标签
            model: 模型实例（用于L2正则化）
        """
        # 主要损失
        pair_loss = self.criterion(predictions['pair_logits'], targets)
        
        total_loss = pair_loss
        
        # L2正则化
        if self.l2_reg > 0:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            total_loss += self.l2_reg * l2_loss
        
        return {
            'total_loss': total_loss,
            'pair_loss': pair_loss
        }