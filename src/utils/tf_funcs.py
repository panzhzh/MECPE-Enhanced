# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

# PyTorch版本的TensorFlow函数

class Saver(object):
    """模型保存器，PyTorch版本"""
    def __init__(self, save_dir, max_to_keep=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep

    def save(self, model, optimizer, step):
        save_path = os.path.join(self.save_dir, f'model_step_{step}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, save_path)

    def restore(self, model, optimizer, idx=''):
        if idx:
            model_path = os.path.join(self.save_dir, f'model_step_{idx}.pt')
        else:
            # 找到最新的checkpoint
            checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('model_step_')]
            if not checkpoints:
                raise FileNotFoundError("No checkpoint found")
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(self.save_dir, latest_checkpoint)
        
        print("Reading model parameters from %s" % model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']

def get_weight_variable(name, shape, device='cpu'):
    """获取权重变量，PyTorch版本"""
    return torch.nn.Parameter(torch.uniform_(torch.empty(shape, device=device), -0.01, 0.01))

def getmask(length, max_len, device='cpu'):
    """
    生成mask矩阵
    length: [batch_size] 
    返回: [batch_size, max_len, 1]
    """
    batch_size = length.size(0)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < length.unsqueeze(1)
    return mask.float().unsqueeze(-1)

class BiLSTM(nn.Module):
    """双向LSTM，对应原版biLSTM函数"""
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
    
    def forward(self, inputs, length):
        """
        inputs: [batch_size, max_len, input_size]
        length: [batch_size]
        返回: [batch_size, max_len, hidden_size*2]
        """
        # 确保length至少为1，避免pack_padded_sequence错误
        length_clamped = torch.clamp(length, min=1)
        
        # Pack padded sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, length_clamped.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        packed_outputs, (hidden, cell) = self.lstm(packed_inputs)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs

class LSTM(nn.Module):
    """单向LSTM，对应原版LSTM函数"""
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
    
    def forward(self, inputs, length):
        """
        inputs: [batch_size, max_len, input_size]
        length: [batch_size]
        返回: [batch_size, max_len, hidden_size]
        """
        # 确保length至少为1，避免pack_padded_sequence错误
        length_clamped = torch.clamp(length, min=1)
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, length_clamped.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs

def softmax_by_length(inputs, length):
    """
    按长度进行softmax
    inputs: [batch_size, 1, max_len]
    length: [batch_size]
    返回: [batch_size, 1, max_len]
    """
    batch_size, _, max_len = inputs.shape
    device = inputs.device
    
    # 确保长度至少为1
    length_clamped = torch.clamp(length, min=1)
    
    # 创建mask
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < length_clamped.unsqueeze(1)
    mask = mask.unsqueeze(1).float()  # [batch_size, 1, max_len]
    
    # 对未mask的位置应用softmax
    inputs_exp = torch.exp(inputs)
    inputs_masked = inputs_exp * mask
    inputs_sum = inputs_masked.sum(dim=2, keepdim=True) + 1e-9
    
    return inputs_masked / inputs_sum

class AttentionLayer(nn.Module):
    """注意力层，对应原版att_var函数"""
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs, length):
        """
        inputs: [batch_size, max_len, input_size]
        length: [batch_size]
        返回: [batch_size, input_size]
        """
        batch_size, max_len, input_size = inputs.shape
        
        # 计算注意力权重
        u = torch.tanh(self.w1(inputs.view(-1, input_size)))  # [batch_size*max_len, hidden_size]
        alpha = self.w2(u).view(batch_size, 1, max_len)  # [batch_size, 1, max_len]
        alpha = softmax_by_length(alpha, length)  # [batch_size, 1, max_len]
        
        # 加权求和
        output = torch.bmm(alpha, inputs)  # [batch_size, 1, input_size]
        return output.squeeze(1)  # [batch_size, input_size]

class LayerNorm(nn.Module):
    """层归一化，对应原版layer_normalize函数"""
    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadAttention(nn.Module):
    """多头注意力机制，对应原版multihead_attention函数"""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return context

class MultiHeadAttentionRealtime(MultiHeadAttention):
    """实时多头注意力机制，对应原版multihead_attention_realtime函数"""
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 创建实时mask (下三角矩阵)
        realtime_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
        
        if mask is not None:
            mask = mask * realtime_mask
        else:
            mask = realtime_mask
        
        return super().forward(query, key, value, mask)

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络，对应原版pw_feedforward函数"""
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    """Transformer块，对应原版standard_trans函数"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TransformerBlockRealtime(nn.Module):
    """实时Transformer块，对应原版standard_trans_realtime函数"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(TransformerBlockRealtime, self).__init__()
        self.attention = MultiHeadAttentionRealtime(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class CrossModalTransformer(nn.Module):
    """跨模态Transformer，对应原版cross_modal_trans函数"""
    def __init__(self, d_model, num_heads, d_ff, num_layers=4, dropout=0.0):
        super(CrossModalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, query, key_value, mask=None):
        x = query
        for layer in self.layers:
            attn_output = layer.attention(self.norm1(x), self.norm2(key_value), self.norm2(key_value), mask)
            x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(self.norm1(x))
        x = self.norm1(x + ff_output)
        
        return x

# 全局函数接口 (保持与原版TensorFlow函数相同的接口)
def biLSTM(inputs, length, n_hidden, scope, device='cpu'):
    """BiLSTM的函数式接口"""
    lstm = BiLSTM(inputs.size(-1), n_hidden).to(device)
    return lstm(inputs, length)

def standard_trans(inputs, n_hidden, n_head=1, scope="standard_trans", device='cpu'):
    """Transformer的函数式接口"""
    transformer = TransformerBlock(n_hidden, n_head, n_hidden).to(device)
    return transformer(inputs)

def standard_trans_realtime(inputs, n_hidden, n_head=1, scope="standard_trans", device='cpu'):
    """实时Transformer的函数式接口"""
    transformer = TransformerBlockRealtime(n_hidden, n_head, n_hidden).to(device)
    return transformer(inputs)

def cross_modal_trans(Q, KV, n_hidden, cmt_num_layer=4, n_head=1, cmt_dropout=0, scope="cross_modal_trans", device='cpu'):
    """跨模态Transformer的函数式接口"""
    transformer = CrossModalTransformer(n_hidden, n_head, n_hidden, cmt_num_layer, cmt_dropout).to(device)
    return transformer(Q, KV)

def layer_normalize(inputs, epsilon=1e-8, scope="ln"):
    """层归一化的函数式接口"""
    # 直接实现layer normalization而不需要创建LayerNorm对象
    mean = inputs.mean(-1, keepdim=True)
    std = inputs.std(-1, keepdim=True)
    return (inputs - mean) / (std + epsilon)

def att_var(inputs, length, w1, b1, w2, device='cpu'):
    """注意力机制的函数式接口"""
    attention = AttentionLayer(inputs.size(-1), w1.size(0)).to(device)
    # 需要设置权重
    attention.w1.weight.data = w1.T
    attention.w1.bias.data = b1
    attention.w2.weight.data = w2.T
    return attention(inputs, length)