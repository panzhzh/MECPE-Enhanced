# encoding: utf-8

"""
PyTorch版本的BERT模型，使用HuggingFace transformers替代原版TensorFlow实现
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np

class BertEncoder(nn.Module):
    """BERT编码器包装类，兼容原版接口"""
    
    def __init__(self, bert_model_name='bert-base-cased', dropout_rate=0.1, attention_dropout_rate=0.1):
        super(BertEncoder, self).__init__()
        
        # 加载预训练BERT模型和配置
        self.config = BertConfig.from_pretrained(bert_model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = attention_dropout_rate
        
        self.bert = BertModel.from_pretrained(bert_model_name, config=self.config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
        Returns:
            sequence_output: [batch_size, seq_len, hidden_size]
            pooled_output: [batch_size, hidden_size]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        return outputs.last_hidden_state, outputs.pooler_output

class BertSentenceEncoder(nn.Module):
    """BERT句子级编码器 (对应原版BERT_sen模式)"""
    
    def __init__(self, bert_model_name='bert-base-cased', dropout_rate=0.1, attention_dropout_rate=0.1):
        super(BertSentenceEncoder, self).__init__()
        self.bert_encoder = BertEncoder(bert_model_name, dropout_rate, attention_dropout_rate)
        
    def forward(self, input_ids, attention_mask=None):
        """
        处理句子级输入
        Args:
            input_ids: [batch_size, max_doc_len, max_sen_len]
            attention_mask: [batch_size, max_doc_len, max_sen_len]
        Returns:
            sentence_embeddings: [batch_size, max_doc_len, hidden_size]
        """
        batch_size, max_doc_len, max_sen_len = input_ids.shape
        
        # 重塑为 [batch_size * max_doc_len, max_sen_len]
        input_ids_flat = input_ids.view(-1, max_sen_len)
        attention_mask_flat = attention_mask.view(-1, max_sen_len) if attention_mask is not None else None
        
        # BERT编码
        _, pooled_output = self.bert_encoder(input_ids_flat, attention_mask_flat)
        
        # 重塑回 [batch_size, max_doc_len, hidden_size]
        sentence_embeddings = pooled_output.view(batch_size, max_doc_len, -1)
        
        return sentence_embeddings

class BertDocumentEncoder(nn.Module):
    """BERT文档级编码器 (对应原版BERT_doc模式)"""
    
    def __init__(self, bert_model_name='bert-base-cased', dropout_rate=0.1, attention_dropout_rate=0.1):
        super(BertDocumentEncoder, self).__init__()
        self.bert_encoder = BertEncoder(bert_model_name, dropout_rate, attention_dropout_rate)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sentence_indices=None):
        """
        处理文档级输入
        Args:
            input_ids: [batch_size, max_doc_len_bert]
            attention_mask: [batch_size, max_doc_len_bert]
            token_type_ids: [batch_size, max_doc_len_bert]
            sentence_indices: [batch_size, max_doc_len] - 每个句子在序列中的起始位置
        Returns:
            sentence_embeddings: [batch_size, max_doc_len, hidden_size]
        """
        # BERT编码整个文档
        sequence_output, _ = self.bert_encoder(input_ids, attention_mask, token_type_ids)
        
        if sentence_indices is not None:
            # 根据句子索引提取每个句子的[CLS]表示
            batch_size, max_doc_len = sentence_indices.shape
            hidden_size = sequence_output.size(-1)
            
            # 构建索引
            batch_indices = torch.arange(batch_size, device=sequence_output.device).unsqueeze(1).expand(-1, max_doc_len)
            flat_indices = (batch_indices.flatten(), sentence_indices.flatten())
            
            # 提取句子表示
            sentence_embeddings = sequence_output[flat_indices].view(batch_size, max_doc_len, hidden_size)
        else:
            # 如果没有提供句子索引，使用整个序列的表示
            sentence_embeddings = sequence_output
            
        return sentence_embeddings

def get_bert_tokenizer(model_name='bert-base-cased'):
    """获取BERT tokenizer，兼容原版接口"""
    do_lower_case = 'uncased' in model_name.lower()
    return BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint, bert_scope='bert'):
    """
    兼容原版的checkpoint映射函数 (在PyTorch中主要用于参数初始化)
    由于使用HuggingFace预训练模型，这个函数主要返回空映射
    """
    assignment_map = {}
    initialized_variable_names = {}
    
    # 在PyTorch版本中，HuggingFace会自动处理预训练权重的加载
    # 这里返回空映射以保持接口兼容性
    for var in tvars:
        var_name = var.name if hasattr(var, 'name') else str(var)
        if bert_scope in var_name:
            initialized_variable_names[var_name] = True
            
    return assignment_map, initialized_variable_names

# 为了兼容原版代码，创建一些别名和包装函数
class BertModel_TF_Compatible:
    """兼容原版TensorFlow BertModel接口的包装类"""
    
    def __init__(self, config, is_training=True, input_ids=None, input_mask=None, 
                 token_type_ids=None, scope=None):
        self.config = config
        self.is_training = is_training
        self.scope = scope
        
        # 确定BERT模型名称
        if 'uncased' in scope.lower():
            model_name = 'bert-base-uncased'
        else:
            model_name = 'bert-base-cased'
            
        # 创建BERT编码器
        self.bert_encoder = BertEncoder(
            model_name, 
            dropout_rate=1-config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1,
            attention_dropout_rate=1-config.attention_probs_dropout_prob if hasattr(config, 'attention_probs_dropout_prob') else 0.1
        )
        
        # 处理输入
        if input_ids is not None:
            self.sequence_output, self.pooled_output = self.bert_encoder(
                input_ids, input_mask, token_type_ids)
    
    def get_pooled_output(self):
        return self.pooled_output
    
    def get_sequence_output(self):
        return self.sequence_output

# 兼容性配置类
class BertConfig_Compatible:
    """兼容原版BertConfig的配置类"""
    
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
    
    @classmethod
    def from_json_file(cls, json_file):
        """从JSON文件加载配置 (在PyTorch版本中使用默认配置)"""
        config = cls()
        # 在实际使用中，HuggingFace会自动处理配置加载
        return config