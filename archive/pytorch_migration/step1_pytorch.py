"""
PyTorch版本的Step1: 情感和原因识别
替代原TensorFlow版本的step1.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm

# 添加项目路径
sys.path.append('.')
sys.path.append('./pytorch_utils')

from config import FLAGS, update_config, print_config
from pytorch_utils.data_loader import (
    ECFDataset, load_embeddings, load_speaker_dict, 
    create_dataloader, get_tokenizer
)
from pytorch_utils.models import EmotionCauseModel, EmotionCauseLoss
from pytorch_utils.metrics import calculate_prf, calculate_prf_emocate


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MECPE Step1 - PyTorch Version')
    
    # 模型参数
    parser.add_argument('--model_type', default='BiLSTM', choices=['BiLSTM', 'BERT'],
                       help='Model type')
    parser.add_argument('--use_x_v', default='no', choices=['yes', 'no'],
                       help='Whether to use video features')
    parser.add_argument('--use_x_a', default='no', choices=['yes', 'no'], 
                       help='Whether to use audio features')
    parser.add_argument('--scope', default='PyTorch_TEMP',
                       help='Experiment scope name')
    parser.add_argument('--choose_emocate', default='no', choices=['yes', 'no'],
                       help='Whether to predict emotion categories')
    parser.add_argument('--share_word_encoder', default='yes', choices=['yes', 'no'],
                       help='Whether emotion and cause share word encoder')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--training_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--l2_reg', type=float, default=1e-5,
                       help='L2 regularization')
    
    # 其他参数
    parser.add_argument('--device', default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_device(device_arg):
    """设置设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU Name: {torch.cuda.get_device_name()}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    return device


def create_log_dir(scope):
    """创建日志目录"""
    log_dir = os.path.join(FLAGS.log_path, scope)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def train_epoch(model, dataloader, criterion, optimizer, device, desc="Training"):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = {'emotion': [], 'cause': []}
    all_targets = {'emotion': [], 'cause': []}
    all_doc_lens = []
    
    progress_bar = tqdm(dataloader, desc=desc)
    
    for batch in progress_bar:
        # 移动到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        predictions = model(batch)
        
        # 计算损失
        targets = {
            'emotions': batch['emotions'],
            'causes': batch['causes'],
            'doc_len': batch['doc_len']
        }
        
        loss_dict = criterion(predictions, targets, model)
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 收集预测结果用于评估
        with torch.no_grad():
            pred_emotions = torch.argmax(predictions['emotion_logits'], dim=-1)
            pred_causes = torch.argmax(predictions['cause_logits'], dim=-1)
            
            all_predictions['emotion'].append(pred_emotions.cpu().numpy())
            all_predictions['cause'].append(pred_causes.cpu().numpy())
            all_targets['emotion'].append(batch['emotions'].cpu().numpy())
            all_targets['cause'].append(batch['causes'].cpu().numpy())
            all_doc_lens.append(batch['doc_len'].cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
        })
    
    # 计算整体指标 - 处理变长序列
    pred_emotions = []
    pred_causes = []
    true_emotions = []
    true_causes = []
    doc_lens = []
    
    for i, batch_pred_emo in enumerate(all_predictions['emotion']):
        batch_pred_cause = all_predictions['cause'][i]
        batch_true_emo = all_targets['emotion'][i]
        batch_true_cause = all_targets['cause'][i]
        batch_doc_len = all_doc_lens[i]
        
        # 对每个样本，只取有效长度
        for j in range(len(batch_doc_len)):
            valid_len = batch_doc_len[j]
            pred_emotions.append(batch_pred_emo[j][:valid_len])
            pred_causes.append(batch_pred_cause[j][:valid_len])
            true_emotions.append(batch_true_emo[j][:valid_len])
            true_causes.append(batch_true_cause[j][:valid_len])
            doc_lens.append(valid_len)
    
    # 现在拼接
    pred_emotions = np.concatenate(pred_emotions) if pred_emotions else np.array([])
    pred_causes = np.concatenate(pred_causes) if pred_causes else np.array([])
    true_emotions = np.concatenate(true_emotions) if true_emotions else np.array([])
    true_causes = np.concatenate(true_causes) if true_causes else np.array([])
    doc_lens = np.array(doc_lens)
    
    if FLAGS.choose_emocate:
        emotion_metrics = calculate_prf_emocate(pred_emotions, true_emotions, doc_lens)
    else:
        emotion_metrics = calculate_prf(pred_emotions, true_emotions, doc_lens)
    
    cause_metrics = calculate_prf(pred_causes, true_causes, doc_lens)
    
    return {
        'loss': total_loss / len(dataloader),
        'emotion_metrics': emotion_metrics,
        'cause_metrics': cause_metrics
    }


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = {'emotion': [], 'cause': []}
    all_targets = {'emotion': [], 'cause': []}
    all_doc_lens = []
    
    progress_bar = tqdm(dataloader, desc=desc)
    
    with torch.no_grad():
        for batch in progress_bar:
            # 移动到设备
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            batch[key][sub_key] = sub_value.to(device)
            
            # 前向传播
            predictions = model(batch)
            
            # 计算损失
            targets = {
                'emotions': batch['emotions'],
                'causes': batch['causes'],
                'doc_len': batch['doc_len']
            }
            
            loss_dict = criterion(predictions, targets, model)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
            
            # 收集预测结果
            pred_emotions = torch.argmax(predictions['emotion_logits'], dim=-1)
            pred_causes = torch.argmax(predictions['cause_logits'], dim=-1)
            
            all_predictions['emotion'].append(pred_emotions.cpu().numpy())
            all_predictions['cause'].append(pred_causes.cpu().numpy())
            all_targets['emotion'].append(batch['emotions'].cpu().numpy())
            all_targets['cause'].append(batch['causes'].cpu().numpy())
            all_doc_lens.append(batch['doc_len'].cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算整体指标 - 处理变长序列
    pred_emotions = []
    pred_causes = []
    true_emotions = []
    true_causes = []
    doc_lens = []
    
    for i, batch_pred_emo in enumerate(all_predictions['emotion']):
        batch_pred_cause = all_predictions['cause'][i]
        batch_true_emo = all_targets['emotion'][i]
        batch_true_cause = all_targets['cause'][i]
        batch_doc_len = all_doc_lens[i]
        
        # 对每个样本，只取有效长度
        for j in range(len(batch_doc_len)):
            valid_len = batch_doc_len[j]
            pred_emotions.append(batch_pred_emo[j][:valid_len])
            pred_causes.append(batch_pred_cause[j][:valid_len])
            true_emotions.append(batch_true_emo[j][:valid_len])
            true_causes.append(batch_true_cause[j][:valid_len])
            doc_lens.append(valid_len)
    
    # 现在拼接
    pred_emotions = np.concatenate(pred_emotions) if pred_emotions else np.array([])
    pred_causes = np.concatenate(pred_causes) if pred_causes else np.array([])
    true_emotions = np.concatenate(true_emotions) if true_emotions else np.array([])
    true_causes = np.concatenate(true_causes) if true_causes else np.array([])
    doc_lens = np.array(doc_lens)
    
    if FLAGS.choose_emocate:
        emotion_metrics = calculate_prf_emocate(pred_emotions, true_emotions, doc_lens)
    else:
        emotion_metrics = calculate_prf(pred_emotions, true_emotions, doc_lens)
    
    cause_metrics = calculate_prf(pred_causes, true_causes, doc_lens)
    
    return {
        'loss': total_loss / len(dataloader),
        'emotion_metrics': emotion_metrics,
        'cause_metrics': cause_metrics,
        'predictions': {
            'emotions': pred_emotions,
            'causes': pred_causes
        }
    }


def save_results(log_dir, run_id, predictions, datasets, embeddings_dict):
    """保存预测结果"""
    # 创建结果目录
    results_dir = os.path.join(log_dir, 'step1')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预测结果文件
    # 这里可以实现具体的文件保存逻辑
    # 格式应该与原版本兼容，以便Step2使用
    
    print(f"Results saved to {results_dir}")


def main():
    # 解析参数
    args = parse_args()
    
    # 更新配置
    config_updates = {
        'model_type': args.model_type,
        'use_x_v': args.use_x_v == 'yes',
        'use_x_a': args.use_x_a == 'yes',
        'scope': args.scope,
        'choose_emocate': args.choose_emocate == 'yes',
        'share_word_encoder': args.share_word_encoder == 'yes',
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'training_epochs': args.training_epochs,
        'l2_reg': args.l2_reg
    }
    update_config(**config_updates)
    
    # 设置环境
    set_seed(args.seed)
    device = setup_device(args.device)
    FLAGS.device = device
    
    # 创建日志目录
    log_dir = create_log_dir(FLAGS.scope)
    
    # 打印配置
    print_config()
    
    # 加载数据
    print("Loading embeddings and data...")
    embeddings_dict = load_embeddings()
    spe_idx, spe_idx_rev = load_speaker_dict(FLAGS.data_path)
    tokenizer = get_tokenizer()
    
    # 创建数据集
    train_dataset = ECFDataset(
        os.path.join(FLAGS.data_path, 'train.txt'),
        tokenizer, embeddings_dict['word_idx'], 
        embeddings_dict['video_idx'], spe_idx, is_step1=True
    )
    
    dev_dataset = ECFDataset(
        os.path.join(FLAGS.data_path, 'dev.txt'),
        tokenizer, embeddings_dict['word_idx'],
        embeddings_dict['video_idx'], spe_idx, is_step1=True
    )
    
    test_dataset = ECFDataset(
        os.path.join(FLAGS.data_path, 'test.txt'),
        tokenizer, embeddings_dict['word_idx'],
        embeddings_dict['video_idx'], spe_idx, is_step1=True
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = create_dataloader(train_dataset, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)
    
    # 创建模型
    print("Building model...")
    model = EmotionCauseModel(embeddings_dict, FLAGS)
    model = model.to(device)
    
    # 创建损失函数和优化器
    criterion = EmotionCauseLoss(FLAGS)
    
    if FLAGS.model_type == 'BiLSTM':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    else:
        # BERT使用不同的优化器配置
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)
    
    # 训练循环
    print("Starting training...")
    best_dev_f1 = -1
    best_test_results = None
    
    for epoch in range(FLAGS.training_epochs):
        print(f"\n=== Epoch {epoch + 1}/{FLAGS.training_epochs} ===")
        
        # 训练
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            desc=f"Training Epoch {epoch + 1}"
        )
        
        # 验证
        dev_results = evaluate(
            model, dev_loader, criterion, device,
            desc=f"Dev Epoch {epoch + 1}"
        )
        
        # 测试
        test_results = evaluate(
            model, test_loader, criterion, device,
            desc=f"Test Epoch {epoch + 1}"
        )
        
        # 打印结果
        print(f"Train - Loss: {train_results['loss']:.4f}")
        print(f"Train - Emotion F1: {train_results['emotion_metrics'][2]:.4f}")
        print(f"Train - Cause F1: {train_results['cause_metrics'][2]:.4f}")
        
        print(f"Dev - Loss: {dev_results['loss']:.4f}")
        print(f"Dev - Emotion F1: {dev_results['emotion_metrics'][2]:.4f}")
        print(f"Dev - Cause F1: {dev_results['cause_metrics'][2]:.4f}")
        
        print(f"Test - Loss: {test_results['loss']:.4f}")
        print(f"Test - Emotion F1: {test_results['emotion_metrics'][2]:.4f}")
        print(f"Test - Cause F1: {test_results['cause_metrics'][2]:.4f}")
        
        # 保存最佳模型
        dev_f1 = dev_results['emotion_metrics'][2]  # 使用emotion F1作为主要指标
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_results = test_results
            
            # 保存模型
            model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': FLAGS,
                'epoch': epoch,
                'dev_f1': dev_f1
            }, model_path)
            print(f"Best model saved to {model_path}")
    
    # 打印最终结果
    print("\n=== Final Results ===")
    print(f"Best Dev Emotion F1: {best_dev_f1:.4f}")
    if best_test_results:
        print(f"Best Test Emotion F1: {best_test_results['emotion_metrics'][2]:.4f}")
        print(f"Best Test Cause F1: {best_test_results['cause_metrics'][2]:.4f}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()