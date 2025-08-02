"""
PyTorch版本的配置文件，替代原TensorFlow FLAGS系统
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # 数据相关
    w2v_file: str = './data/ECF_glove_300.txt'
    data_path: str = './data/'
    video_emb_file: str = './data/features/video_embedding_4096.npy'
    audio_emb_file: str = './data/features/audio_embedding_6373.npy'
    video_idx_file: str = './data/video_id_mapping.npy'
    
    # 嵌入维度
    embedding_dim: int = 300
    embedding_dim_pos: int = 50
    
    # 输入结构参数
    max_sen_len: int = 35
    max_doc_len: int = 35
    max_sen_len_bert: int = 40
    max_doc_len_bert: int = 400
    
    # 模型结构
    model_type: str = 'BiLSTM'  # 'BiLSTM', 'BERT'
    bert_model_name: str = 'bert-base-cased'  # HuggingFace模型名
    share_word_encoder: bool = True
    choose_emocate: bool = False
    use_x_v: bool = False  # 是否使用视频特征
    use_x_a: bool = False  # 是否使用音频特征
    n_hidden: int = 100
    n_class: int = 2
    real_time: bool = False
    
    # 训练参数
    batch_size: int = 8
    learning_rate: float = 1e-5
    epochs: int = 20
    keep_prob1: float = 1.0  # word embedding dropout (1-dropout_rate)
    keep_prob2: float = 1.0  # softmax dropout
    keep_prob_v: float = 0.5  # video dropout
    keep_prob_a: float = 0.5  # audio dropout
    l2_reg: float = 1e-5
    emo_weight: float = 1.0  # emotion loss weight
    cause_weight: float = 1.0  # cause loss weight
    training_epochs: int = 15
    
    # BERT特定参数
    bert_hidden_dropout: float = 0.1
    bert_attention_dropout: float = 0.3
    
    # 日志和保存
    log_path: str = './log'
    scope: str = 'PyTorch_TEMP'
    log_file_name: str = 'step1_pytorch.log'
    
    # Step2特定参数
    pred_future_cause: bool = True
    emocate_eval: int = 6
    step1_file_dir: str = 'step1/'
    save_pair: bool = True
    
    # 设备
    device: str = 'cpu'  # 默认CPU，在运行时动态设置
    
    def __post_init__(self):
        """根据模型类型调整参数"""
        import torch
        
        # 动态设置设备
        if self.device == 'cpu':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.model_type == 'BiLSTM':
            self.batch_size = 32
            self.learning_rate = 0.005
            self.keep_prob1 = 0.5
            self.training_epochs = 30
        
        # 转换keep_prob为dropout_rate (PyTorch惯例)
        self.dropout1 = 1.0 - self.keep_prob1
        self.dropout2 = 1.0 - self.keep_prob2
        self.dropout_v = 1.0 - self.keep_prob_v
        self.dropout_a = 1.0 - self.keep_prob_a


# 为了向后兼容，保持FLAGS接口
FLAGS = Config()


def update_config(**kwargs):
    """更新配置参数"""
    global FLAGS
    for key, value in kwargs.items():
        if hasattr(FLAGS, key):
            setattr(FLAGS, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    # 重新运行post_init
    FLAGS.__post_init__()


def print_config():
    """打印配置信息"""
    print('\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print(f'model_type: {FLAGS.model_type}')
    print(f'share_word_encoder: {FLAGS.share_word_encoder}')
    print(f'choose_emocate: {FLAGS.choose_emocate}')
    print(f'use_x_v: {FLAGS.use_x_v}')
    print(f'use_x_a: {FLAGS.use_x_a}')
    print(f'device: {FLAGS.device}')
    
    print('\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print(f'batch_size: {FLAGS.batch_size}')
    print(f'learning_rate: {FLAGS.learning_rate}')
    print(f'training_epochs: {FLAGS.training_epochs}')
    print(f'l2_reg: {FLAGS.l2_reg}')
    print()