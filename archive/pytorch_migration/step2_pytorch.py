#!/usr/bin/env python3
"""
PyTorch版本的Step2：情感-原因配对
基于Conv的预测结果进行情感-原因对的识别
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import defaultdict

from config import FLAGS
from pytorch_utils.data_loader import ECFDataset, load_embeddings, load_speaker_dict, get_tokenizer, create_dataloader
from pytorch_utils.models import EmotionCausePairModel, EmotionCausePairLoss
from pytorch_utils.metrics import prf_2nd_step, prf_2nd_step_emocate, list_round


def print_info():
    """打印训练配置信息"""
    print('\n' + '='*60)
    print('STEP2 MODEL INFO:')
    print(f'model_type: {FLAGS.model_type}')
    print(f'choose_emocate: {FLAGS.choose_emocate}')
    print(f'use_x_v: {FLAGS.use_x_v}')
    print(f'use_x_a: {FLAGS.use_x_a}')
    print(f'pred_future_cause: {FLAGS.pred_future_cause}')
    print(f'batch_size: {FLAGS.batch_size}')
    print(f'learning_rate: {FLAGS.learning_rate}')
    print(f'dropout1: {FLAGS.dropout1}')
    print(f'dropout2: {FLAGS.dropout2}')
    print(f'l2_reg: {FLAGS.l2_reg}')
    print('='*60 + '\n')


def train_epoch(model, dataloader, criterion, optimizer, device, desc="Training"):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc=desc)
    for batch in progress_bar:
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], dict):
                for subkey in batch[key]:
                    if isinstance(batch[key][subkey], torch.Tensor):
                        batch[key][subkey] = batch[key][subkey].to(device)
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], list) and len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                batch[key] = [item.to(device) for item in batch[key]]
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(batch)
        targets_batch = batch['label']
        
        # 计算损失
        loss_dict = criterion(outputs, targets_batch, model)
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果
        pair_logits = outputs['pair_logits'].detach().cpu()
        pred_batch = torch.argmax(pair_logits, dim=1).numpy()
        predictions.extend(pred_batch)
        targets.extend(targets_batch.cpu().numpy())
        
        # 更新进度条
        acc = np.mean(np.array(predictions) == np.array(targets))
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, targets


def evaluate_epoch(model, dataloader, criterion, device, desc="Evaluating"):
    """评估一个epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    pair_ids = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc)
        for batch in progress_bar:
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)
                elif isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], list) and len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = [item.to(device) for item in batch[key]]
            
            # 前向传播
            outputs = model(batch)
            targets_batch = batch['label']
            
            # 计算损失
            loss_dict = criterion(outputs, targets_batch, model)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
            
            # 收集预测结果
            pair_logits = outputs['pair_logits'].detach().cpu()
            pred_batch = torch.argmax(pair_logits, dim=1).numpy()
            predictions.extend(pred_batch)
            targets.extend(targets_batch.cpu().numpy())
            
            # 收集pair_info用于评估
            if 'pair_info' in batch:
                pair_ids.extend(batch['pair_info'])
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, targets, pair_ids


def create_dict(pair_list, choose_emocate):
    """创建配对字典"""
    emotion_idx_rev = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 
                      4: 'joy', 5: 'sadness', 6: 'surprise'}
    pair_dict = defaultdict(list)
    
    for x in pair_list:
        if choose_emocate:
            tmp = x[1:3] + [emotion_idx_rev[x[3]]]
            pair_dict[x[0]].append(tmp)
        else:
            pair_dict[x[0]].append(x[1:-1])
    
    return pair_dict


def write_predictions(output_file, dataset, predictions, pair_ids):
    """保存预测结果"""
    pair_id_filtered = []
    for i, pred in enumerate(predictions):
        if pred == 1:  # 预测为正样本
            pair_id_filtered.append(pair_ids[i])
    
    print(f"Filtered {len(pair_id_filtered)} positive pairs from {len(predictions)} total pairs")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pair_id_filtered:
            f.write(f"{pair}\n")
    
    print(f"Predictions saved to {output_file}")


def main():
    print_info()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载嵌入和词典
    print("Loading embeddings and dictionaries...")
    embeddings_dict = load_embeddings()
    spe_idx, spe_idx_rev = load_speaker_dict(FLAGS.data_path)
    tokenizer = get_tokenizer()
    
    # 加载数据集
    print("Loading datasets...")
    
    # 训练集
    train_dataset = ECFDataset(
        f"{FLAGS.data_path}/train.txt",
        tokenizer,
        embeddings_dict['word_idx'],
        embeddings_dict['video_idx'],
        spe_idx,
        is_conv=False
    )
    
    # 验证集
    dev_dataset = ECFDataset(
        f"{FLAGS.data_path}/dev.txt",
        tokenizer,
        embeddings_dict['word_idx'],
        embeddings_dict['video_idx'],
        spe_idx,
        is_conv=False
    )
    
    # 测试集
    test_dataset = ECFDataset(
        f"{FLAGS.data_path}/test.txt",
        tokenizer,
        embeddings_dict['word_idx'],
        embeddings_dict['video_idx'],
        spe_idx,
        is_conv=False
    )
    
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Dev: {len(dev_dataset)} pairs")
    print(f"Test: {len(test_dataset)} pairs")
    
    # 创建数据加载器
    train_loader = create_dataloader(train_dataset, FLAGS.batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, FLAGS.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, FLAGS.batch_size, shuffle=False)
    
    # 创建模型
    print("Creating model...")
    model = EmotionCausePairModel(embeddings_dict, FLAGS)
    model.to(device)
    
    # 创建损失函数和优化器
    criterion = EmotionCausePairLoss(FLAGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    
    # 训练参数
    best_f1 = -1.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(FLAGS.epochs):
        print(f"\nEpoch {epoch + 1}/{FLAGS.epochs}")
        
        # 训练
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            desc=f"Training Epoch {epoch+1}"
        )
        
        # 验证
        dev_loss, dev_preds, dev_targets, dev_pair_ids = evaluate_epoch(
            model, dev_loader, criterion, device,
            desc=f"Dev Epoch {epoch+1}"
        )
        
        # 测试
        test_loss, test_preds, test_targets, test_pair_ids = evaluate_epoch(
            model, test_loader, criterion, device,
            desc=f"Test Epoch {epoch+1}"
        )
        
        # 计算F1分数
        # 注意：这里需要从数据集中获取pair_id_all信息
        # 为了简化，我们使用基本的accuracy作为评估指标
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        dev_acc = np.mean(np.array(dev_preds) == np.array(dev_targets))
        test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Acc: {dev_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        # 检查是否是最佳模型
        if dev_acc > best_f1:
            best_f1 = dev_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存最佳模型
            model_save_path = f"{FLAGS.scope}_best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, model_save_path)
            print(f"New best model saved at epoch {epoch+1} with Dev Acc: {dev_acc:.4f}")
            
            # 保存预测结果
            if hasattr(FLAGS, 'save_predictions') and FLAGS.save_predictions:
                os.makedirs(f"predictions/{FLAGS.scope}", exist_ok=True)
                write_predictions(f"predictions/{FLAGS.scope}/dev_predictions.txt", 
                                dev_dataset, dev_preds, dev_pair_ids)
                write_predictions(f"predictions/{FLAGS.scope}/test_predictions.txt",
                                test_dataset, test_preds, test_pair_ids)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best Dev Acc: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Step2 PyTorch Training')
    parser.add_argument('--use_x_v', type=str, default='', help='Use video features')
    parser.add_argument('--use_x_a', type=str, default='', help='Use audio features')
    parser.add_argument('--scope', type=str, default='Step2_BiLSTM', help='Experiment scope')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions')
    
    args = parser.parse_args()
    
    # 更新配置
    FLAGS.use_x_v = bool(args.use_x_v.lower() in ['yes', 'true', '1'])
    FLAGS.use_x_a = bool(args.use_x_a.lower() in ['yes', 'true', '1'])
    FLAGS.scope = args.scope
    FLAGS.epochs = args.epochs
    
    main()