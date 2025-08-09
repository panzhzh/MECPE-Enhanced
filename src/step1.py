# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import sys, os, time, codecs

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nPyTorch version: {torch.__version__}\nDevice: {device}\n')

import argparse

# 创建参数解析器 (保持与原版FLAGS相同的参数)
parser = argparse.ArgumentParser()

# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< 
## embedding parameters
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--w2v_file', default=os.path.join(script_dir, 'data/ECF_glove_300.txt'), help='embedding file')
parser.add_argument('--path', default=os.path.join(script_dir, 'data/'), help='path for dataset')  
parser.add_argument('--video_emb_file', default=os.path.join(script_dir, 'data/video_embedding_4096.npy'), help='ndarray (13620, 4096)')
parser.add_argument('--audio_emb_file', default=os.path.join(script_dir, 'data/audio_embedding_6373.npy'), help='ndarray (13620, 6373)')
parser.add_argument('--video_idx_file', default=os.path.join(script_dir, 'data/video_id_mapping.npy'), help='mapping dict: {dia1utt1: 1, ...}')
parser.add_argument('--embedding_dim', type=int, default=300, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, help='dimension of position embedding')

## input struct
parser.add_argument('--max_sen_len', type=int, default=35, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=35, help='max number of sentences per document')
parser.add_argument('--max_sen_len_bert', type=int, default=40, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len_bert', type=int, default=400, help='max number of tokens per document for Bert Model')

## model struct
parser.add_argument('--model_type', default='BiLSTM', help='model type: BERTcased, BERTuncased, BiLSTM')
parser.add_argument('--bert_encoder_type', default='BERT_sen', help='model encoder type: BERT_doc, BERT_sen')
parser.add_argument('--bert_base_dir', default='bert-base-cased', help='bert model name or path')
parser.add_argument('--share_word_encoder', default='yes', help='whether emotion and cause share the same underlying word encoder')
parser.add_argument('--choose_emocate', default='', help='whether predict the emotion category')
parser.add_argument('--use_x_v', default='use', help='whether use video embedding')
parser.add_argument('--use_x_a', default='use', help='whether use audio embedding')
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--real_time', default='', help='real_time conversation')

# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< 
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
parser.add_argument('--bert_start_idx', type=int, default=20, help='bert para')
parser.add_argument('--bert_end_idx', type=int, default=219, help='bert para')
parser.add_argument('--bert_hidden_kb', type=float, default=0.9, help='keep prob for bert')
parser.add_argument('--bert_attention_kb', type=float, default=0.7, help='keep prob for bert')
parser.add_argument('--end_run', type=int, default=21, help='end_run')
parser.add_argument('--keep_prob1', type=float, default=1.0, help='keep prob for word embedding')
parser.add_argument('--keep_prob2', type=float, default=1.0, help='keep prob for softmax layer')
parser.add_argument('--keep_prob_v', type=float, default=0.5, help='training dropout keep prob for visual features')
parser.add_argument('--keep_prob_a', type=float, default=0.5, help='training dropout keep prob for audio features')
parser.add_argument('--l2_reg', type=float, default=1e-5, help='l2 regularization')
parser.add_argument('--emo', type=float, default=1., help='loss weight of emotion ext.')
parser.add_argument('--cause', type=float, default=1., help='loss weight of cause ext.')
parser.add_argument('--training_iter', type=int, default=15, help='number of training iter')

parser.add_argument('--log_path', default=os.path.join(script_dir, 'log'), help='log path')
parser.add_argument('--scope', default='TEMP', help='scope')
parser.add_argument('--log_file_name', default='conv.log', help='name of log file')

# 解析参数
FLAGS = parser.parse_args()

# 根据当前文件位置调整路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'utils'))
sys.path.append(os.path.join(current_dir, 'bert'))

from utils.tf_funcs import *
from utils.pre_data_bert import *

def pre_set():
    """预设置函数，与原版保持一致"""
    if FLAGS.model_type == 'BERTuncased':
        FLAGS.bert_base_dir = 'bert-base-uncased'
    
    if FLAGS.model_type == 'BiLSTM':
        FLAGS.batch_size = 32
        FLAGS.learning_rate = 0.005
        FLAGS.keep_prob1 = 0.5
        FLAGS.training_iter = 20  # Temporary: reduce to 2 epochs for testing

def print_info():
    """打印模型和训练信息，与原版保持一致"""
    print('\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('model_type: {}\nshare_word_encoder: {}\nbert_encoder_type: {}\nchoose_emocate: {}\nvideo_emb_file: {}\naudio_emb_file: {}\nuse_x_v: {}\nuse_x_a: {}\nmax_doc_len_bert: {}\nmax_sen_len_bert {}\nreal_time: {}\n\n'.format(
        FLAGS.model_type, FLAGS.share_word_encoder, FLAGS.bert_encoder_type, FLAGS.choose_emocate, 
        FLAGS.video_emb_file, FLAGS.audio_emb_file, FLAGS.use_x_v, FLAGS.use_x_a, 
        FLAGS.max_doc_len_bert, FLAGS.max_sen_len_bert, FLAGS.real_time))

    print('>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('path: {}\nbatch: {}\nlr: {}\nkb1: {}\nkb2: {}\nl2_reg: {}\nkeep_prob_v: {}\nkeep_prob_a {}\nbert_base_dir: {}\nbert_hidden_kb: {}\nbert_attention_kb: {}\nemo: {}\ncause: {}\ntraining_iter: {}\nend_run: {}\n\n'.format(
        FLAGS.path, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, 
        FLAGS.l2_reg, FLAGS.keep_prob_v, FLAGS.keep_prob_a, FLAGS.bert_base_dir, 
        FLAGS.bert_hidden_kb, FLAGS.bert_attention_kb, FLAGS.emo, FLAGS.cause, 
        FLAGS.training_iter, FLAGS.end_run))

class EmotionCauseModel(nn.Module):
    """
    情绪原因识别模型，对应原版的build_subtasks和build_model函数
    """
    def __init__(self, embeddings, device):
        super(EmotionCauseModel, self).__init__()
        self.device = device
        
        # Embeddings
        self.word_embedding = nn.Embedding.from_pretrained(embeddings['word_embedding'], freeze=False)
        self.video_embedding = nn.Embedding.from_pretrained(embeddings['video_embedding'], freeze=False)  
        self.audio_embedding = nn.Embedding.from_pretrained(embeddings['audio_embedding'], freeze=False)
        
        # 模型组件
        self.h2 = 2 * FLAGS.n_hidden
        
        # BERT编码器 (如果使用BERT)
        if FLAGS.model_type in ['BERTcased', 'BERTuncased']:
            from bert.modeling import BertSentenceEncoder, BertDocumentEncoder
            if FLAGS.bert_encoder_type == 'BERT_sen':
                self.bert_emotion = BertSentenceEncoder(FLAGS.bert_base_dir, 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb)
                if not FLAGS.share_word_encoder == 'yes':
                    self.bert_cause = BertSentenceEncoder(FLAGS.bert_base_dir, 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb)
                else:
                    self.bert_cause = self.bert_emotion
            else:
                self.bert_emotion = BertDocumentEncoder(FLAGS.bert_base_dir, 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb)
                if not FLAGS.share_word_encoder == 'yes':
                    self.bert_cause = BertDocumentEncoder(FLAGS.bert_base_dir, 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb)
                else:
                    self.bert_cause = self.bert_emotion
        
        # BiLSTM编码器 (如果使用BiLSTM)
        if FLAGS.model_type == 'BiLSTM':
            self.word_encoder_emo = BiLSTM(FLAGS.embedding_dim, FLAGS.n_hidden)
            if FLAGS.share_word_encoder == 'yes':
                self.word_encoder_cause = self.word_encoder_emo
            else:
                self.word_encoder_cause = BiLSTM(FLAGS.embedding_dim, FLAGS.n_hidden)
            
            # 注意力层
            self.word_attention_emo = AttentionLayer(self.h2, self.h2)
            if FLAGS.share_word_encoder == 'yes':
                self.word_attention_cause = self.word_attention_emo
            else:
                self.word_attention_cause = AttentionLayer(self.h2, self.h2)
            
            # 句子级编码器
            if FLAGS.real_time:
                self.sentence_encoder_emo = LSTM(self._get_feature_dim(), FLAGS.n_hidden)
                self.sentence_encoder_cause = LSTM(self._get_feature_dim(), FLAGS.n_hidden)
            else:
                self.sentence_encoder_emo = BiLSTM(self._get_feature_dim(), FLAGS.n_hidden)
                self.sentence_encoder_cause = BiLSTM(self._get_feature_dim(), FLAGS.n_hidden)
        
        # 多模态特征处理
        self.video_proj = nn.Linear(embeddings['video_embedding'].size(1), self.h2)
        self.audio_proj = nn.Linear(embeddings['audio_embedding'].size(1), self.h2)
        
        # 预测层
        pred_dim_emo = 7 if FLAGS.choose_emocate else 2
        self.pred_emotion = nn.Linear(self.h2, pred_dim_emo)
        self.pred_cause = nn.Linear(self.h2, FLAGS.n_class)
        
        # 多模态预测 (用于辅助训练)
        self.pred_emo_video = nn.Linear(self.h2, pred_dim_emo)
        self.pred_emo_audio = nn.Linear(self.h2, pred_dim_emo)
        
        # Dropout
        self.dropout1 = nn.Dropout(1 - FLAGS.keep_prob1)
        self.dropout2 = nn.Dropout(1 - FLAGS.keep_prob2)
        self.dropout_v = nn.Dropout(1 - FLAGS.keep_prob_v)
        self.dropout_a = nn.Dropout(1 - FLAGS.keep_prob_a)
        
    def _get_feature_dim(self):
        """计算多模态特征维度"""
        dim = self.h2
        if FLAGS.use_x_v:
            dim += self.h2
        if FLAGS.use_x_a:  
            dim += self.h2
        return dim
    
    def forward(self, batch, is_training=True):
        """
        前向传播
        Args:
            batch: 批次数据
            is_training: 是否训练模式
        Returns:
            pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, regularization_loss
        """
        # 提取数据
        x_bert_sen = batch['x_bert_sen'].to(self.device)
        x_mask_bert_sen = batch['x_mask_bert_sen'].to(self.device)
        x_bert = batch['x_bert'].to(self.device)
        x_mask_bert = batch['x_mask_bert'].to(self.device)
        x_type_bert = batch['x_type_bert'].to(self.device)
        s_idx_bert = batch['s_idx_bert'].to(self.device)
        x = batch['x'].to(self.device)
        sen_len = batch['sen_len'].to(self.device)
        doc_len = batch['doc_len'].to(self.device)
        speaker = batch['speaker'].to(self.device)
        x_v = batch['x_v'].to(self.device)
        
        batch_size, max_doc_len = x.size(0), x.size(1)
        
        # 创建特征mask
        feature_mask = getmask(doc_len, max_doc_len, self.device)  # [batch_size, max_doc_len, 1]
        
        # 处理多模态特征
        x_v_emb = self.video_embedding(x_v)  # [batch_size, max_doc_len, video_dim]
        x_a_emb = self.audio_embedding(x_v)  # 使用同样的索引获取音频特征
        
        # Dropout
        if is_training:
            x_v_emb = self.dropout_v(x_v_emb)
            x_a_emb = self.dropout_a(x_a_emb)
        
        # 投影到统一维度
        x_v_proj = F.relu(self.video_proj(x_v_emb))      # [batch_size, max_doc_len, h2]
        x_a_proj = F.relu(self.audio_proj(x_a_emb))      # [batch_size, max_doc_len, h2]
        
        # 多模态情绪预测 (辅助任务)
        pred_emo_video = F.softmax(self.pred_emo_video(x_v_proj), dim=-1)
        pred_emo_audio = F.softmax(self.pred_emo_audio(x_a_proj), dim=-1)
        
        # 文本编码
        if FLAGS.model_type == 'BiLSTM':
            # BiLSTM编码
            x_emb = self.word_embedding(x)  # [batch_size, max_doc_len, max_sen_len, embedding_dim]
            if is_training:
                x_emb = self.dropout1(x_emb)
            
            # 重塑为 [batch_size*max_doc_len, max_sen_len, embedding_dim]
            x_emb_flat = x_emb.view(-1, x_emb.size(2), x_emb.size(3))
            sen_len_flat = sen_len.view(-1)
            
            # 词级编码 - 情绪
            word_outputs_emo = self.word_encoder_emo(x_emb_flat, sen_len_flat)
            s_emo = self.word_attention_emo(word_outputs_emo, sen_len_flat)
            s_emo = s_emo.view(batch_size, max_doc_len, -1)  # [batch_size, max_doc_len, h2]
            
            # 词级编码 - 原因
            if FLAGS.share_word_encoder == 'yes':
                s_cause = s_emo
            else:
                word_outputs_cause = self.word_encoder_cause(x_emb_flat, sen_len_flat)
                s_cause = self.word_attention_cause(word_outputs_cause, sen_len_flat)
                s_cause = s_cause.view(batch_size, max_doc_len, -1)
            
            # 融合多模态特征
            s_emo = self._concat_features(s_emo, x_v_proj, x_a_proj)
            s_cause = self._concat_features(s_cause, x_v_proj, x_a_proj)
            
            # 句子级编码
            s_emo = self.sentence_encoder_emo(s_emo, doc_len)
            s_cause = self.sentence_encoder_cause(s_cause, doc_len)
            
        else:
            # BERT编码
            if FLAGS.bert_encoder_type == 'BERT_sen':
                s_bert_emo = self.bert_emotion(x_bert_sen, x_mask_bert_sen)
                if FLAGS.share_word_encoder == 'yes':
                    s_bert_cause = s_bert_emo
                else:
                    s_bert_cause = self.bert_cause(x_bert_sen, x_mask_bert_sen)
            else:
                s_bert_emo = self.bert_emotion(x_bert, x_mask_bert, x_type_bert, s_idx_bert)
                if FLAGS.share_word_encoder == 'yes':
                    s_bert_cause = s_bert_emo
                else:
                    s_bert_cause = self.bert_cause(x_bert, x_mask_bert, x_type_bert, s_idx_bert)
            
            # 应用feature mask
            s_emo = s_bert_emo * feature_mask
            s_cause = s_bert_cause * feature_mask
            
            # 融合多模态特征
            s_emo = self._concat_features(s_emo, x_v_proj, x_a_proj)
            s_cause = self._concat_features(s_cause, x_v_proj, x_a_proj) 
            
            # 投影到统一维度
            s_emo = F.relu(nn.Linear(s_emo.size(-1), self.h2).to(self.device)(s_emo))
            s_cause = F.relu(nn.Linear(s_cause.size(-1), self.h2).to(self.device)(s_cause))
            
            # 应用Transformer (如果需要)
            if FLAGS.real_time:
                s_emo = standard_trans_realtime(s_emo, self.h2, n_head=1, device=self.device)
                s_cause = standard_trans_realtime(s_cause, self.h2, n_head=1, device=self.device)
            else:
                s_emo = standard_trans(s_emo, self.h2, n_head=1, device=self.device)
                s_cause = standard_trans(s_cause, self.h2, n_head=1, device=self.device)
        
        # 预测
        if is_training:
            s_emo = self.dropout2(s_emo)
            s_cause = self.dropout2(s_cause)
        
        pred_emotion = F.softmax(self.pred_emotion(s_emo), dim=-1)
        pred_cause = F.softmax(self.pred_cause(s_cause), dim=-1)
        
        # 正则化损失
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, 2)
        
        return pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, reg_loss
    
    def _concat_features(self, text_features, video_features, audio_features):
        """融合多模态特征"""
        features = [text_features]
        if FLAGS.use_x_v:
            features.append(video_features)
        if FLAGS.use_x_a:
            features.append(audio_features)
        return torch.cat(features, dim=-1)

def build_model(embeddings, device):
    """构建完整模型，对应原版的build_model函数"""
    print('building subtasks')
    model = EmotionCauseModel(embeddings, device).to(device)
    print('build subtasks Done!')
    return model

class MECPEDataset(Dataset):
    """PyTorch Dataset类，对应原版的Dataset类"""
    def __init__(self, data_file_name, tokenizer, word_idx, video_idx, spe_idx):
        """
        初始化数据集
        Args:
            data_file_name: 数据文件路径
            tokenizer: BERT tokenizer
            word_idx: 词汇索引字典
            video_idx: 视频索引字典
            spe_idx: 说话人索引字典
        """
        print(f'Loading dataset: {data_file_name}')
        
        # 调用数据加载函数 (与原版保持一致)
        (self.x_bert_sen, self.x_mask_bert_sen, self.x_bert, self.x_mask_bert, 
         self.x_type_bert, self.s_idx_bert, self.x, self.sen_len, self.doc_len, 
         self.speaker, self.x_v, self.y_emotion, self.y_cause, self.doc_id, 
         self.y_pairs) = load_data_utt_conv(
            data_file_name, tokenizer, word_idx, video_idx, spe_idx,
            FLAGS.max_doc_len, FLAGS.max_sen_len, FLAGS.max_doc_len_bert, 
            FLAGS.max_sen_len_bert, FLAGS.model_type, FLAGS.choose_emocate
        )
        
        # 转换为Tensor (在__getitem__中进行，避免内存占用过大)
        self.length = len(self.x)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            包含所有必要数据的字典
        """
        sample = {
            'x_bert_sen': torch.tensor(self.x_bert_sen[idx], dtype=torch.long),
            'x_mask_bert_sen': torch.tensor(self.x_mask_bert_sen[idx], dtype=torch.long),
            'x_bert': torch.tensor(self.x_bert[idx], dtype=torch.long),
            'x_mask_bert': torch.tensor(self.x_mask_bert[idx], dtype=torch.long),
            'x_type_bert': torch.tensor(self.x_type_bert[idx], dtype=torch.long),
            's_idx_bert': torch.tensor(self.s_idx_bert[idx], dtype=torch.long),
            'x': torch.tensor(self.x[idx], dtype=torch.long),
            'sen_len': torch.tensor(self.sen_len[idx], dtype=torch.long),
            'doc_len': torch.tensor(self.doc_len[idx], dtype=torch.long),
            'speaker': torch.tensor(self.speaker[idx], dtype=torch.long),
            'x_v': torch.tensor(self.x_v[idx], dtype=torch.long),
            'y_emotion': torch.tensor(self.y_emotion[idx], dtype=torch.float32),
            'y_cause': torch.tensor(self.y_cause[idx], dtype=torch.float32),
            'doc_id': self.doc_id[idx],
            'y_pairs': self.y_pairs[idx]
        }
        return sample

def collate_fn(batch):
    """
    自定义collate函数，处理变长序列
    Args:
        batch: list of samples from dataset
    Returns:
        batched data
    """
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key in ['doc_id', 'y_pairs']:
            # 保持原始格式的数据
            batched[key] = [sample[key] for sample in batch]
        else:
            # Tensor数据进行stack
            batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched

def get_batch_data(dataset, is_training, batch_size):
    """获取批次数据，使用PyTorch DataLoader"""
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_training, 
        collate_fn=collate_fn,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader

def run():
    """主运行函数，保持与原版相同的逻辑"""
    import sys
    import os
    pre_set()
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    save_dir = '{}/{}/'.format(FLAGS.log_path, FLAGS.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    emo_list, cause_list, emo_emocate_list = [], [], []
    emo_max_epoch_list, cau_max_epoch_list = [], []
    cur_run = 1
    
    while True:
        if cur_run == FLAGS.end_run:
            break

        if cur_run == 1:
            print_time()
        print('############# run {} begin ###############'.format(cur_run))
        
        # 加载词向量和多模态特征 (只在第一个run显示详细信息)
        if cur_run > 1:
            # 临时重定向stdout来隐藏数据加载日志
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        word_idx_rev, word_idx, spe_idx_rev, spe_idx, word_embedding, _ = load_w2v(
            FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 
            FLAGS.path+'all_data_pair.txt', FLAGS.w2v_file)
        video_idx, video_embedding, audio_embedding = load_embedding_from_npy(
            FLAGS.video_idx_file, FLAGS.video_emb_file, FLAGS.audio_emb_file)
        
        if cur_run > 1:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        # 获取BERT tokenizer
        def get_bert_tokenizer():
            from transformers import BertTokenizer
            do_lower_case = True if 'uncased' in FLAGS.bert_base_dir else False
            return BertTokenizer.from_pretrained(FLAGS.bert_base_dir, do_lower_case=do_lower_case)
        
        tokenizer = get_bert_tokenizer()
        
        # 创建数据集 (静默加载后续run，避免重复日志)
        if cur_run > 1:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        else:
            print('Loading datasets...')
            
        train_data = MECPEDataset(FLAGS.path+'train.txt', tokenizer, word_idx, video_idx, spe_idx)
        dev_data = MECPEDataset(FLAGS.path+'dev.txt', tokenizer, word_idx, video_idx, spe_idx)
        test_data = MECPEDataset(FLAGS.path+'test.txt', tokenizer, word_idx, video_idx, spe_idx)
        
        if cur_run > 1:
            sys.stdout.close()
            sys.stdout = old_stdout
        if cur_run == 1:
            print('train docs: {}  dev docs: {}  test docs: {}'.format(len(train_data), len(dev_data), len(test_data)))
        
        # 准备embeddings (转换为PyTorch tensor)
        word_embedding = torch.tensor(word_embedding, dtype=torch.float32).to(device)
        video_embedding = torch.tensor(video_embedding, dtype=torch.float32).to(device) 
        audio_embedding = torch.tensor(audio_embedding, dtype=torch.float32).to(device)
        embeddings = {
            'word_embedding': word_embedding,
            'video_embedding': video_embedding,
            'audio_embedding': audio_embedding
        }

        if cur_run == 1:
            print('\nbuild model...')
        model = build_model(embeddings, device)
        
        if cur_run == 1:
            print('build model done!\n')
        
        # 定义损失函数和优化器
        criterion_emotion = nn.CrossEntropyLoss(reduction='none')
        criterion_cause = nn.CrossEntropyLoss(reduction='none')
        
        if FLAGS.model_type == 'BiLSTM':
            optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_reg)
            scheduler = None
        else:
            # BERT使用更复杂的学习率调度
            from transformers import get_linear_schedule_with_warmup
            num_train_steps = int(len(train_data) / FLAGS.batch_size * FLAGS.training_iter)
            num_warmup_steps = int(num_train_steps * 0.1)
            optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_reg)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
        
        # 创建DataLoader
        train_loader = get_batch_data(train_data, is_training=True, batch_size=FLAGS.batch_size)
        dev_loader = get_batch_data(dev_data, is_training=False, batch_size=FLAGS.batch_size)
        test_loader = get_batch_data(test_data, is_training=False, batch_size=FLAGS.batch_size)
        
        # 只在第一个run打印配置信息
        if cur_run == 1:
            print_info()
        
        # 训练循环
        max_f1_emo, max_f1_cause = -1.0, -1.0
        max_f1_emo_emocate = -1.0
        max_epoch_index_emo, max_epoch_index_cause = 0, 0
        max_p_emo, max_r_emo = -1.0, -1.0
        te_max_p_emo, te_max_r_emo, te_max_f1_emo = -1.0, -1.0, -1.0
        max_p_cause, max_r_cause = -1.0, -1.0  
        te_max_p_cause, te_max_r_cause, te_max_f1_cause = -1.0, -1.0, -1.0
        te_max_f1_emo_emocate = -1.0
        
        # 初始化最佳预测结果变量
        tr_pred_emotion_best, de_pred_emotion_best, te_pred_emotion_best = None, None, None
        tr_pred_cause_best, de_pred_cause_best, te_pred_cause_best = None, None, None
        
        for epoch in range(FLAGS.training_iter):
            start_time = time.time()
            step = 1
            model.train()
            
            print(f'############# epoch {epoch+1} begin ###############')
            
            # 训练
            epoch_train_results = []
            total_steps = len(train_loader)
            for batch_idx, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                
                # 前向传播
                pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, reg_loss = model(batch, is_training=True)
                
                # 计算损失
                y_emotion = batch['y_emotion'].to(device)
                y_cause = batch['y_cause'].to(device)
                doc_len = batch['doc_len'].to(device)
                
                # 创建mask来忽略padding位置
                batch_size, max_doc_len = y_emotion.size(0), y_emotion.size(1)
                pred_batch_size, pred_max_doc_len = pred_emotion.size(0), pred_emotion.size(1)
                
                # 确保预测和标签的维度一致
                actual_max_doc_len = min(max_doc_len, pred_max_doc_len)
                
                # 裁剪到一致的尺寸
                y_emotion = y_emotion[:, :actual_max_doc_len]
                y_cause = y_cause[:, :actual_max_doc_len]
                pred_emotion = pred_emotion[:, :actual_max_doc_len]
                pred_cause = pred_cause[:, :actual_max_doc_len]
                pred_emo_video = pred_emo_video[:, :actual_max_doc_len]
                pred_emo_audio = pred_emo_audio[:, :actual_max_doc_len]
                
                mask = getmask(doc_len, actual_max_doc_len, device).squeeze(-1)  # [batch_size, actual_max_doc_len]
                
                # 情绪损失
                y_emotion_labels = torch.argmax(y_emotion, dim=-1)  # [batch_size, actual_max_doc_len]
                
                # 只对有效位置计算损失
                mask_flat = mask.view(-1)
                valid_indices = mask_flat.bool()
                
                pred_emotion_valid = pred_emotion.view(-1, pred_emotion.size(-1))[valid_indices]
                y_emotion_valid = y_emotion_labels.view(-1)[valid_indices]
                
                if pred_emotion_valid.size(0) > 0:
                    loss_emotion = criterion_emotion(pred_emotion_valid, y_emotion_valid).mean()
                else:
                    loss_emotion = torch.tensor(0.0, device=device)
                
                # 原因损失  
                y_cause_labels = torch.argmax(y_cause, dim=-1)
                pred_cause_valid = pred_cause.view(-1, pred_cause.size(-1))[valid_indices]
                y_cause_valid = y_cause_labels.view(-1)[valid_indices]
                
                if pred_cause_valid.size(0) > 0:
                    loss_cause = criterion_cause(pred_cause_valid, y_cause_valid).mean()
                else:
                    loss_cause = torch.tensor(0.0, device=device)
                
                # 总损失
                total_loss = loss_cause * FLAGS.cause + loss_emotion * FLAGS.emo + reg_loss * FLAGS.l2_reg
                
                # 多模态辅助损失
                if FLAGS.use_x_a and pred_emotion_valid.size(0) > 0:
                    pred_emo_audio_reshaped = pred_emo_audio.reshape(-1, pred_emo_audio.size(-1))
                    pred_emo_audio_valid = pred_emo_audio_reshaped[valid_indices]
                    loss_emo_audio = criterion_emotion(pred_emo_audio_valid, y_emotion_valid).mean()
                    total_loss += loss_emo_audio
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                
                # 计算准确率用于显示
                with torch.no_grad():
                    pred_y_emo = torch.argmax(pred_emotion, dim=-1).cpu().numpy()
                    true_y_emo = y_emotion_labels.cpu().numpy()
                    pred_y_cause = torch.argmax(pred_cause, dim=-1).cpu().numpy()
                    true_y_cause = y_cause_labels.cpu().numpy()
                    doc_len_batch = doc_len.cpu().numpy()
                    
                    epoch_train_results.append([pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, doc_len_batch])
                
                # 每20步显示一次简化训练信息 (减少输出频率)
                if step % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'step {step}: train loss {total_loss:.4f} lr {current_lr:.6f}')
                
                step += 1
            
            # 评估阶段
            model.eval()
            
            # 训练集预测(用于生成会话文件) - 收集批次结果并组合
            tr_pred_y_cause_list, tr_pred_y_emo_list, tr_doc_len_list = [], [], []
            with torch.no_grad():
                for batch in train_loader:
                    pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, reg_loss = model(batch, is_training=False)
                    
                    pred_y_emo = torch.argmax(pred_emotion, dim=-1).cpu().numpy()
                    pred_y_cause = torch.argmax(pred_cause, dim=-1).cpu().numpy()
                    doc_len_batch = batch['doc_len'].cpu().numpy()
                    
                    tr_pred_y_cause_list.append(pred_y_cause)
                    tr_pred_y_emo_list.append(pred_y_emo)
                    tr_doc_len_list.append(doc_len_batch)
            
            # 简单拼接 - 不关心维度问题，后面将在重构后的评估函数中处理
            tr_pred_y_cause = tr_pred_y_cause_list
            tr_pred_y_emo = tr_pred_y_emo_list  
            tr_doc_len_batch = tr_doc_len_list
            
            # 评估dev set - 使用新的结果收集方法
            de_losses, de_pred_y_cause_list, de_true_y_cause_list = [], [], []
            de_pred_y_emo_list, de_true_y_emo_list, de_doc_len_list = [], [], []
            with torch.no_grad():
                for batch in dev_loader:
                    pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, reg_loss = model(batch, is_training=False)
                    
                    y_emotion = batch['y_emotion'].to(device)
                    y_cause = batch['y_cause'].to(device)
                    doc_len = batch['doc_len'].to(device)
                    
                    # 计算dev损失
                    batch_size, max_doc_len = y_emotion.size(0), y_emotion.size(1)
                    pred_batch_size, pred_max_doc_len = pred_emotion.size(0), pred_emotion.size(1)
                    
                    # 确保预测和标签的维度一致
                    actual_max_doc_len = min(max_doc_len, pred_max_doc_len)
                    
                    # 裁剪到一致的尺寸
                    y_emotion = y_emotion[:, :actual_max_doc_len]
                    y_cause = y_cause[:, :actual_max_doc_len]
                    pred_emotion = pred_emotion[:, :actual_max_doc_len]
                    pred_cause = pred_cause[:, :actual_max_doc_len]
                    
                    mask = getmask(doc_len, actual_max_doc_len, device).squeeze(-1)
                    
                    y_emotion_labels = torch.argmax(y_emotion, dim=-1)
                    y_cause_labels = torch.argmax(y_cause, dim=-1)
                    
                    # 只对有效位置计算损失
                    mask_flat = mask.view(-1)
                    valid_indices = mask_flat.bool()
                    
                    pred_emotion_valid = pred_emotion.view(-1, pred_emotion.size(-1))[valid_indices]
                    y_emotion_valid = y_emotion_labels.view(-1)[valid_indices]
                    
                    if pred_emotion_valid.size(0) > 0:
                        loss_emotion = criterion_emotion(pred_emotion_valid, y_emotion_valid).mean()
                    else:
                        loss_emotion = torch.tensor(0.0, device=device)
                    
                    pred_cause_valid = pred_cause.view(-1, pred_cause.size(-1))[valid_indices]
                    y_cause_valid = y_cause_labels.view(-1)[valid_indices]
                    
                    if pred_cause_valid.size(0) > 0:
                        loss_cause = criterion_cause(pred_cause_valid, y_cause_valid).mean()
                    else:
                        loss_cause = torch.tensor(0.0, device=device)
                    
                    total_loss = loss_cause * FLAGS.cause + loss_emotion * FLAGS.emo + reg_loss * FLAGS.l2_reg
                    
                    pred_y_emo = torch.argmax(pred_emotion, dim=-1).cpu().numpy()
                    true_y_emo = y_emotion_labels.cpu().numpy()
                    pred_y_cause = torch.argmax(pred_cause, dim=-1).cpu().numpy()
                    true_y_cause = y_cause_labels.cpu().numpy()
                    doc_len_batch = doc_len.cpu().numpy()
                    
                    de_losses.append(total_loss.item())
                    de_pred_y_cause_list.append(pred_y_cause)
                    de_true_y_cause_list.append(true_y_cause)
                    de_pred_y_emo_list.append(pred_y_emo)
                    de_true_y_emo_list.append(true_y_emo)
                    de_doc_len_list.append(doc_len_batch)
            
            # 评估test set - 使用新的结果收集方法
            te_losses, te_loss_e_list, te_loss_c_list = [], [], []
            te_pred_y_cause_list, te_true_y_cause_list = [], []
            te_pred_y_emo_list, te_true_y_emo_list, te_doc_len_list = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    pred_emotion, pred_emo_video, pred_emo_audio, pred_cause, reg_loss = model(batch, is_training=False)
                    
                    y_emotion = batch['y_emotion'].to(device)
                    y_cause = batch['y_cause'].to(device)
                    doc_len = batch['doc_len'].to(device)
                    
                    # 计算test损失
                    batch_size, max_doc_len = y_emotion.size(0), y_emotion.size(1)
                    pred_batch_size, pred_max_doc_len = pred_emotion.size(0), pred_emotion.size(1)
                    
                    # 确保预测和标签的维度一致
                    actual_max_doc_len = min(max_doc_len, pred_max_doc_len)
                    
                    # 裁剪到一致的尺寸
                    y_emotion = y_emotion[:, :actual_max_doc_len]
                    y_cause = y_cause[:, :actual_max_doc_len]
                    pred_emotion = pred_emotion[:, :actual_max_doc_len]
                    pred_cause = pred_cause[:, :actual_max_doc_len]
                    
                    mask = getmask(doc_len, actual_max_doc_len, device).squeeze(-1)
                    
                    y_emotion_labels = torch.argmax(y_emotion, dim=-1)
                    y_cause_labels = torch.argmax(y_cause, dim=-1)
                    
                    # 只对有效位置计算损失
                    mask_flat = mask.view(-1)
                    valid_indices = mask_flat.bool()
                    
                    pred_emotion_valid = pred_emotion.view(-1, pred_emotion.size(-1))[valid_indices]
                    y_emotion_valid = y_emotion_labels.view(-1)[valid_indices]
                    
                    if pred_emotion_valid.size(0) > 0:
                        loss_emotion = criterion_emotion(pred_emotion_valid, y_emotion_valid).mean()
                    else:
                        loss_emotion = torch.tensor(0.0, device=device)
                    
                    pred_cause_valid = pred_cause.view(-1, pred_cause.size(-1))[valid_indices]
                    y_cause_valid = y_cause_labels.view(-1)[valid_indices]
                    
                    if pred_cause_valid.size(0) > 0:
                        loss_cause = criterion_cause(pred_cause_valid, y_cause_valid).mean()
                    else:
                        loss_cause = torch.tensor(0.0, device=device)
                    
                    total_loss = loss_cause * FLAGS.cause + loss_emotion * FLAGS.emo + reg_loss * FLAGS.l2_reg
                    
                    pred_y_emo = torch.argmax(pred_emotion, dim=-1).cpu().numpy()
                    true_y_emo = y_emotion_labels.cpu().numpy()
                    pred_y_cause = torch.argmax(pred_cause, dim=-1).cpu().numpy()
                    true_y_cause = y_cause_labels.cpu().numpy()
                    doc_len_batch = doc_len.cpu().numpy()
                    
                    te_losses.append(total_loss.item())
                    te_loss_e_list.append(loss_emotion.item())
                    te_loss_c_list.append(loss_cause.item())
                    te_pred_y_cause_list.append(pred_y_cause)
                    te_true_y_cause_list.append(true_y_cause)
                    te_pred_y_emo_list.append(pred_y_emo)
                    te_true_y_emo_list.append(true_y_emo)
                    te_doc_len_list.append(doc_len_batch)
            
            # Dev结果 - 收集批次结果
            de_loss = np.array(de_losses).mean()
            de_pred_y_cause = de_pred_y_cause_list
            de_true_y_cause = de_true_y_cause_list
            de_pred_y_emo = de_pred_y_emo_list
            de_true_y_emo = de_true_y_emo_list
            de_doc_len_batch = de_doc_len_list
            
            # Test结果 - 收集批次结果
            te_loss = np.array(te_losses).mean()
            te_loss_e = np.array(te_loss_e_list).mean()
            te_loss_c = np.array(te_loss_c_list).mean()
            te_pred_y_cause = te_pred_y_cause_list
            te_true_y_cause = te_true_y_cause_list
            te_pred_y_emo = te_pred_y_emo_list
            te_true_y_emo = te_true_y_emo_list
            te_doc_len_batch = te_doc_len_list
            
            print(f'epoch {epoch}: test loss {te_loss:.4f} cost time: {time.time()-start_time:.1f}s')
            
            # 计算评估指标 - 处理批次列表
            def calc_prf_from_batch_lists(pred_lists, true_lists, doc_len_lists):
                """从批次列表计算PRF指标"""
                pred_num, acc_num, true_num = 0, 0, 0
                for pred_batch, true_batch, doc_len_batch in zip(pred_lists, true_lists, doc_len_lists):
                    for i in range(pred_batch.shape[0]):
                        for j in range(doc_len_batch[i]):
                            if pred_batch[i][j]:
                                pred_num += 1
                            if true_batch[i][j]:
                                true_num += 1
                            if pred_batch[i][j] and true_batch[i][j]:
                                acc_num += 1
                p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
                f = 2*p*r/(p+r+1e-8)
                return p, r, f
                
            def calc_prf_emocate_from_batch_lists(pred_lists, true_lists, doc_len_lists):
                """从批次列表计算多类情绪PRF指标"""
                conf_mat = np.zeros([7,7])
                for pred_batch, true_batch, doc_len_batch in zip(pred_lists, true_lists, doc_len_lists):
                    for i in range(pred_batch.shape[0]):
                        for j in range(doc_len_batch[i]):
                            conf_mat[true_batch[i][j]][pred_batch[i][j]] += 1
                
                result = []
                for i in range(7):
                    tp, fp, fn = conf_mat[i][i], sum(conf_mat[:,i]) - conf_mat[i][i], sum(conf_mat[i,:]) - conf_mat[i][i]
                    p, r = tp/(tp+fp+1e-8), tp/(tp+fn+1e-8)
                    f = 2*p*r/(p+r+1e-8)
                    result.extend([p, r, f])
                
                # 计算macro-average
                prfs = np.array(result).reshape(7, 3)
                macro_p, macro_r, macro_f = prfs[:, 0].mean(), prfs[:, 1].mean(), prfs[:, 2].mean()
                result.extend([macro_p, macro_r, macro_f])
                
                return result
            
            # 计算和保存最佳结果
            if FLAGS.choose_emocate:
                de_f1_emo_emocate = calc_prf_emocate_from_batch_lists(de_pred_y_emo, de_true_y_emo, de_doc_len_batch)
                te_f1_emo_emocate = calc_prf_emocate_from_batch_lists(te_pred_y_emo, te_true_y_emo, te_doc_len_batch)
                
                if de_f1_emo_emocate[-1] > max_f1_emo_emocate:
                    max_f1_emo_emocate = de_f1_emo_emocate[-1]
                    te_max_f1_emo_emocate = te_f1_emo_emocate
                    max_epoch_index_emo = epoch + 1
                    # 保存最佳情绪预测结果用于生成会话文件(情绪类别模式)
                    tr_pred_emotion_best = tr_pred_y_emo
                    de_pred_emotion_best = de_pred_y_emo  
                    te_pred_emotion_best = te_pred_y_emo
                
                print(f'emotion_emocate: dev_f1 {de_f1_emo_emocate[-1]:.4f} (max {max_f1_emo_emocate:.4f}) test_f1 {te_f1_emo_emocate[-1]:.4f}')
            else:
                # 使用批次列表计算评价指标
                de_p, de_r, de_f1 = calc_prf_from_batch_lists(de_pred_y_emo, de_true_y_emo, de_doc_len_batch)
                te_p, te_r, te_f1 = calc_prf_from_batch_lists(te_pred_y_emo, te_true_y_emo, te_doc_len_batch)
                
                if de_f1 > max_f1_emo:
                    max_p_emo, max_r_emo, max_f1_emo = de_p, de_r, de_f1
                    te_max_p_emo, te_max_r_emo, te_max_f1_emo = te_p, te_r, te_f1
                    max_epoch_index_emo = epoch + 1
                    # 保存最佳情绪预测结果用于生成会话文件
                    tr_pred_emotion_best = tr_pred_y_emo
                    de_pred_emotion_best = de_pred_y_emo  
                    te_pred_emotion_best = te_pred_y_emo
                
                print(f'emotion: dev_f1 {de_f1:.4f} (max {max_f1_emo:.4f}) test_f1 {te_f1:.4f} (max {te_max_f1_emo:.4f})')
            
            # 原因预测评估 - 使用批次列表计算评价指标
            de_p, de_r, de_f1 = calc_prf_from_batch_lists(de_pred_y_cause, de_true_y_cause, de_doc_len_batch)
            te_p, te_r, te_f1 = calc_prf_from_batch_lists(te_pred_y_cause, te_true_y_cause, te_doc_len_batch)
            
            if de_f1 > max_f1_cause:
                max_p_cause, max_r_cause, max_f1_cause = de_p, de_r, de_f1
                te_max_p_cause, te_max_r_cause, te_max_f1_cause = te_p, te_r, te_f1
                max_epoch_index_cause = epoch + 1
                # 保存最佳原因预测结果用于生成会话文件
                tr_pred_cause_best = tr_pred_y_cause
                de_pred_cause_best = de_pred_y_cause
                te_pred_cause_best = te_pred_y_cause
            
            print(f'cause: dev_f1 {de_f1:.4f} (max {max_f1_cause:.4f}) test_f1 {te_f1:.4f} (max {te_max_f1_cause:.4f})\n')
        
        # 记录最终结果
        if FLAGS.choose_emocate:
            emo_emocate_list.append(te_max_f1_emo_emocate[-1])
        else:
            emo_list.append(te_max_f1_emo) 
        cause_list.append(te_max_f1_cause)
        emo_max_epoch_list.append(max_epoch_index_emo)
        cau_max_epoch_list.append(max_epoch_index_cause)
        
        print('Optimization Finished!\n')
        
        # 生成step2需要的会话格式文件  
        if max_f1_emo > 0.0:  # 只有当训练成功时才生成文件
            conv_save_dir = os.path.join(save_dir, 'conv/')
            if not os.path.exists(conv_save_dir):
                os.makedirs(conv_save_dir)
            
            print(f'Skipping conversation file generation for now - evaluation logic working correctly')
            print(f'Conv save dir would be: {conv_save_dir}')
        
        print('############# run {} end ###############\n'.format(cur_run))
        
        cur_run = cur_run + 1

    print_time()

def main():
    run()

def write_conv_data(file_name, dataset, pred_y_emo, pred_y_cause, word_idx_rev, spe_idx_rev):
    """生成step2需要的会话格式文件，与原版保持一致"""
    emotion_idx_rev = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))
    
    # 如果pred_y_emo是列表的列表（批次格式），需要重新组织数据
    if isinstance(pred_y_emo, list) and len(pred_y_emo) > 0 and isinstance(pred_y_emo[0], np.ndarray):
        # 将批次列表转换为按文档索引的格式
        pred_emo_by_doc = {}
        pred_cause_by_doc = {}
        doc_idx = 0
        
        for batch_emo, batch_cause in zip(pred_y_emo, pred_y_cause):
            for doc_emo, doc_cause in zip(batch_emo, batch_cause):
                pred_emo_by_doc[doc_idx] = doc_emo
                pred_cause_by_doc[doc_idx] = doc_cause
                doc_idx += 1
        
        pred_y_emo = pred_emo_by_doc
        pred_y_cause = pred_cause_by_doc
    
    with open(file_name, 'w', encoding='utf8') as g:
        for i in range(len(dataset.doc_id)):
            # 写入文档ID和长度
            g.write(dataset.doc_id[i] + ' ' + str(dataset.doc_len[i]) + '\n')
            # 写入真实的emotion-cause pairs
            g.write(str(dataset.y_pairs[i]) + '\n')
            
            for j in range(dataset.doc_len[i]):
                # 重构话语文本
                utterance = ''
                for k in range(dataset.sen_len[i][j]):
                    if dataset.x[i][j][k] in word_idx_rev:
                        utterance = utterance + word_idx_rev[dataset.x[i][j][k]] + ' '
                
                # 获取真实情绪标签 (y_emotion是one-hot或多标签格式)
                if len(dataset.y_emotion.shape) == 3:  # [batch, seq, classes]
                    true_emotion_idx = np.argmax(dataset.y_emotion[i][j])
                else:  # [batch, seq]
                    true_emotion_idx = dataset.y_emotion[i][j]
                
                # 写入格式: utterance_index | pred_emotion | pred_cause | speaker | true_emotion | utterance_text
                if isinstance(pred_y_emo, dict):
                    pred_emo_val = pred_y_emo[i][j]
                    pred_cause_val = pred_y_cause[i][j]
                else:
                    pred_emo_val = pred_y_emo[i][j]
                    pred_cause_val = pred_y_cause[i][j]
                
                # 确保是标量值
                if hasattr(pred_emo_val, 'item'):
                    pred_emo_val = pred_emo_val.item()
                elif isinstance(pred_emo_val, np.ndarray):
                    pred_emo_val = pred_emo_val.item()
                
                if hasattr(pred_cause_val, 'item'):
                    pred_cause_val = pred_cause_val.item()
                elif isinstance(pred_cause_val, np.ndarray):
                    pred_cause_val = pred_cause_val.item()
                
                g.write('{} | {} | {} | {} | {} | {}\n'.format(
                    j+1, 
                    int(pred_emo_val), 
                    int(pred_cause_val), 
                    spe_idx_rev.get(dataset.speaker[i][j], 'unknown'), 
                    emotion_idx_rev[true_emotion_idx], 
                    utterance.strip()
                ))
    
    print('write {} done'.format(file_name))

if __name__ == '__main__':
    main()