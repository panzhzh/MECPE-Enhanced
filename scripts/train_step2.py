"""
Step2 Training Script for MECPE
Emotion-Cause Pair Classification using modern encoders
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import RobertaTokenizer
import time

from src.utils.config import Config
from src.data.step2_dataset import create_step2_datasets, collate_step2_pairs
from src.models.step2_model import create_step2_model
from src.evaluation.step2_metrics import Step2Metrics

def run_epoch(model, dataloader, optimizer=None, device='cpu', is_train=True, config=None):
    """Run one epoch (train or eval)"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    # Initialize metrics
    use_emotion_categories = config.model.use_emotion_categories if config else False
    metrics = Step2Metrics(use_emotion_categories=use_emotion_categories)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Training" if is_train else "Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)              # [batch_size, 2, max_seq_len]
            attention_mask = batch['attention_mask'].to(device)    # [batch_size, 2, max_seq_len]
            distance = batch['distance'].to(device)               # [batch_size]
            labels = batch['labels'].to(device)                   # [batch_size]
            
            # Optional emotion category
            emotion_category = None
            if 'emotion_category' in batch:
                emotion_category = batch['emotion_category'].to(device)
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                distance=distance,
                emotion_category=emotion_category
            )
            
            # Convert labels to one-hot for loss computation (following original)
            batch_size = labels.size(0)
            labels_onehot = torch.zeros(batch_size, 2, device=device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            # Compute loss
            loss_dict = model.compute_loss(logits, labels_onehot)
            loss = loss_dict['total_loss']
            
            # Backward pass (only in training)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Update Step2 metrics
            metrics.update(
                loss=loss.item(),
                pair_id_all=batch.get('true_pairs', []),  # Will be added in evaluation
                pair_id=batch['pair_ids'],
                pred_y=predictions
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    # For proper Step2 evaluation, we need to collect all predictions first
    # This is a simplified version - full evaluation requires collecting all pairs
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'accuracy': np.mean([1 if p == l else 0 for p, l in zip(predictions, labels.cpu().numpy())]) if num_batches > 0 else 0.0
    }

def evaluate_step2_full(model, dataloader, device, config):
    """
    Full Step2 evaluation following original methodology
    Collect all predictions and compute Step2 metrics properly
    """
    model.eval()
    
    all_predictions = []
    all_pair_ids = []
    all_true_pairs = set()  # This should come from the original dataset
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Full Evaluation"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            distance = batch['distance'].to(device)
            labels = batch['labels'].to(device)
            
            # Optional emotion category
            emotion_category = None
            if 'emotion_category' in batch:
                emotion_category = batch['emotion_category'].to(device)
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                distance=distance,
                emotion_category=emotion_category
            )
            
            # Compute loss
            batch_size = labels.size(0)
            labels_onehot = torch.zeros(batch_size, 2, device=device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            loss_dict = model.compute_loss(logits, labels_onehot)
            loss = loss_dict['total_loss']
            
            # Collect predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_pair_ids.extend(batch['pair_ids'])
            
            # Collect true pairs
            for i, label in enumerate(labels.cpu().numpy()):
                if label == 1:  # This is a true pair
                    all_true_pairs.add(batch['pair_ids'][i])
            
            total_loss += loss.item()
            num_batches += 1
    
    # Compute Step2 metrics
    use_emotion_categories = config.model.use_emotion_categories
    metrics = Step2Metrics(use_emotion_categories=use_emotion_categories)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Update metrics with all collected data
    metrics.update(
        loss=avg_loss,
        pair_id_all=list(all_true_pairs),
        pair_id=all_pair_ids,
        pred_y=np.array(all_predictions)
    )
    
    return metrics.compute()

def main():
    """Main Step2 training function"""
    print("🚀 Starting Step2 Training (Emotion-Cause Pair Classification)...")
    
    # Load config
    config = Config("configs/base_config.yaml")
    print(f"Config: {config.experiment.name}")
    print(f"Text Model: {config.model.text_model}")
    print(f"Device: {config.training.device}")
    print(f"Use Emotion Categories: {config.model.use_emotion_categories}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    # Create datasets
    print("\n📊 Loading Step2 datasets...")
    try:
        train_dataset, dev_dataset, test_dataset = create_step2_datasets(config, tokenizer)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Please run Step1 training first to generate the required prediction files.")
        return
    
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Dev: {len(dev_dataset)} pairs")
    print(f"Test: {len(test_dataset)} pairs")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.step2_batch_size,
        shuffle=True, 
        collate_fn=collate_step2_pairs
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.step2_batch_size,
        shuffle=False,
        collate_fn=collate_step2_pairs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.step2_batch_size,
        shuffle=False,
        collate_fn=collate_step2_pairs
    )
    
    # Create model
    print("\n🧠 Creating Step2 model...")
    model = create_step2_model(config)
    device = torch.device(config.training.device)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer (following original Step2 settings)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.step2_learning_rate,
        weight_decay=config.training.l2_reg
    )
    
    # Training loop
    print(f"\n🎯 Training for {config.training.step2_num_epochs} epochs...")
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(config.training.step2_num_epochs):
        start_time = time.time()
        print(f"\n=== Epoch {epoch + 1}/{config.training.step2_num_epochs} ===")
        
        # Training
        train_metrics = run_epoch(model, train_loader, optimizer, device, is_train=True, config=config)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluation on dev set
        print("Evaluating on dev set...")
        dev_metrics = evaluate_step2_full(model, dev_loader, device, config)
        
        # Print main metrics
        if config.model.use_emotion_categories:
            main_f1 = dev_metrics.get('filtered_weighted_f1', 0.0)
            print(f"Dev Weighted F1: {main_f1:.4f}")
            print(f"Dev 4-class F1: {dev_metrics.get('filtered_weighted_f1_4class', 0.0):.4f}")
        else:
            main_f1 = dev_metrics.get('filtered_f1', 0.0)
            print(f"Dev F1: {main_f1:.4f}")
            print(f"Dev Precision: {dev_metrics.get('filtered_precision', 0.0):.4f}")
            print(f"Dev Recall: {dev_metrics.get('filtered_recall', 0.0):.4f}")
        
        print(f"Keep Rate: {dev_metrics.get('keep_rate', 0.0):.4f}")
        
        # Save best model
        if main_f1 > best_f1:
            best_f1 = main_f1
            best_epoch = epoch + 1
            
            # Save model
            model_save_path = os.path.join(config.training.save_dir, 'best_step2_model.pt')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved new best model (F1: {best_f1:.4f})")
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.1f}s")
    
    print(f"\n🎉 Training completed!")
    print(f"Best Dev F1: {best_f1:.4f} (Epoch {best_epoch})")
    
    # Final evaluation on test set
    if best_f1 > 0:
        print("\n📊 Final evaluation on test set...")
        # Load best model
        model_save_path = os.path.join(config.training.save_dir, 'best_step2_model.pt')
        model.load_state_dict(torch.load(model_save_path))
        
        test_metrics = evaluate_step2_full(model, test_loader, device, config)
        
        if config.model.use_emotion_categories:
            test_f1 = test_metrics.get('filtered_weighted_f1', 0.0)
            print(f"Test Weighted F1: {test_f1:.4f}")
            print(f"Test 4-class F1: {test_metrics.get('filtered_weighted_f1_4class', 0.0):.4f}")
        else:
            test_f1 = test_metrics.get('filtered_f1', 0.0)
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_metrics.get('filtered_precision', 0.0):.4f}")
            print(f"Test Recall: {test_metrics.get('filtered_recall', 0.0):.4f}")

if __name__ == "__main__":
    main()