"""
Pair-level Training Script for MECPE
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
from src.data.pair_dataset import create_pair_datasets, collate_pair_samples
from src.models.pair_model import create_pair_model
from src.evaluation.pair_level_metrics import PairLevelMetrics

def run_epoch(model, dataloader, optimizer=None, device='cpu', is_train=True, config=None):
    """Run one epoch (train or eval)"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    # For training epochs, use simple accuracy tracking
    use_emotion_categories = config.model.use_emotion_categories if config else False  
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
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
            
            # Update simple training metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += np.sum(predictions == labels.cpu().numpy())
            total_samples += batch_size
    
    # Return simple metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy
    }

def evaluate_pair_full(model, dataloader, device, config):
    """
    Full Pair evaluation using proper pair-level metrics
    """
    model.eval()
    
    # Use proper pair-level metrics
    metrics = PairLevelMetrics()
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Full Evaluation"):
            batch_count += 1
            
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
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            
            # Group by document and update metrics for THIS batch
            doc_data = {}  # Reset for each batch - this is correct per-batch processing
            
            for i, pair_id in enumerate(batch['pair_ids']):
                doc_id, emo_utt, cause_utt, emotion_cat = pair_id
                
                if doc_id not in doc_data:
                    doc_data[doc_id] = {'predicted_pairs': [], 'true_pairs': []}
                
                # Add prediction if model predicts this is a true pair
                if predictions[i] == 1:
                    doc_data[doc_id]['predicted_pairs'].append((emo_utt, cause_utt, emotion_cat))
                
                # Add true pair if label is 1
                if labels[i].item() == 1:
                    doc_data[doc_id]['true_pairs'].append((emo_utt, cause_utt, emotion_cat))
            
            # Update metrics for each document in this batch
            for doc_id, data in doc_data.items():
                metrics.update(
                    loss=loss.item() / len(doc_data),  # Distribute loss across documents in batch
                    doc_id=doc_id,
                    pair_predictions=data['predicted_pairs'],
                    true_pairs=data['true_pairs']
                )
    
    # Compute and return final metrics
    return metrics.compute()

def main():
    """Main Pair training function"""
    print("🚀 Starting Pair Training (Emotion-Cause Pair Classification)...")
    
    # Load config
    config = Config("configs/base_config.yaml")
    print(f"Config: {config.experiment.name}")
    print(f"Text Model: {config.model.text_model}")
    print(f"Device: {config.training.device}")
    print(f"Use Emotion Categories: {config.model.use_emotion_categories}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    # Create datasets
    print("\n📊 Loading Pair datasets...")
    try:
        train_dataset, dev_dataset, test_dataset = create_pair_datasets(config, tokenizer)
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
        batch_size=config.training.pair_batch_size,
        shuffle=True, 
        collate_fn=collate_pair_samples
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.pair_batch_size,
        shuffle=False,
        collate_fn=collate_pair_samples
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.pair_batch_size,
        shuffle=False,
        collate_fn=collate_pair_samples
    )
    
    # Create model
    print("\n🧠 Creating Pair model...")
    model = create_pair_model(config)
    device = torch.device(config.training.device)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer (following original Pair settings)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.pair_learning_rate,
        weight_decay=config.training.l2_reg
    )
    
    # Training loop
    print(f"\n🎯 Training for {config.training.pair_num_epochs} epochs...")
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(config.training.pair_num_epochs):
        start_time = time.time()
        print(f"\n=== Epoch {epoch + 1}/{config.training.pair_num_epochs} ===")
        
        # Training
        train_metrics = run_epoch(model, train_loader, optimizer, device, is_train=True, config=config)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluation on dev set
        print("Evaluating on dev set...")
        dev_metrics = evaluate_pair_full(model, dev_loader, device, config)
        
        # Print main metrics using correct keys from PairLevelMetrics
        # Use weighted F1 as main metric (following CodaLab standards)
        main_f1 = dev_metrics.get('weighted_f1', 0.0)
        micro_f1 = dev_metrics.get('pair_f1', 0.0)
        
        print(f"Dev W-Avg F1: {main_f1:.4f} (Main Metric)")
        print(f"Dev W-Avg Precision: {dev_metrics.get('weighted_precision', 0.0):.4f}")
        print(f"Dev W-Avg Recall: {dev_metrics.get('weighted_recall', 0.0):.4f}")
        print(f"Dev Micro F1: {micro_f1:.4f}")
        print(f"Dev Micro Precision: {dev_metrics.get('pair_precision', 0.0):.4f}")
        print(f"Dev Micro Recall: {dev_metrics.get('pair_recall', 0.0):.4f}")
        
        # Debug information
        print(f"Predicted/True/Correct pairs: {dev_metrics.get('num_predicted_pairs', 0)}/{dev_metrics.get('num_true_pairs', 0)}/{dev_metrics.get('num_correct_pairs', 0)}")
        print(f"Documents processed: {dev_metrics.get('num_documents', 0)}")
        
        # Also evaluate on test set for monitoring (but don't use for model selection)
        print("Evaluating on test set...")
        test_metrics = evaluate_pair_full(model, test_loader, device, config)
        test_main_f1 = test_metrics.get('weighted_f1', 0.0)
        test_micro_f1 = test_metrics.get('pair_f1', 0.0)
        
        print(f"Test W-Avg F1: {test_main_f1:.4f}")
        print(f"Test W-Avg Precision: {test_metrics.get('weighted_precision', 0.0):.4f}")
        print(f"Test W-Avg Recall: {test_metrics.get('weighted_recall', 0.0):.4f}")
        print(f"Test Micro F1: {test_micro_f1:.4f}")
        print(f"Test Micro Precision: {test_metrics.get('pair_precision', 0.0):.4f}")
        print(f"Test Micro Recall: {test_metrics.get('pair_recall', 0.0):.4f}")
        
        # Save best model
        if main_f1 > best_f1:
            best_f1 = main_f1
            best_epoch = epoch + 1
            
            # Save model
            model_save_path = os.path.join(config.training.save_dir, 'best_pair_model.pt')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved new best model (W-Avg F1: {best_f1:.4f})")
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.1f}s")
    
    print(f"\n🎉 Training completed!")
    print(f"Best Dev W-Avg F1: {best_f1:.4f} (Epoch {best_epoch})")
    
    # Final evaluation on test set with best model
    if best_f1 > 0:
        print("\n📊 Final evaluation on test set with best model...")
        # Load best model
        model_save_path = os.path.join(config.training.save_dir, 'best_pair_model.pt')
        model.load_state_dict(torch.load(model_save_path))
        
        final_test_metrics = evaluate_pair_full(model, test_loader, device, config)
        
        final_weighted_f1 = final_test_metrics.get('weighted_f1', 0.0)
        final_micro_f1 = final_test_metrics.get('pair_f1', 0.0)
        
        print(f"Final Test W-Avg F1: {final_weighted_f1:.4f}")
        print(f"Final Test W-Avg Precision: {final_test_metrics.get('weighted_precision', 0.0):.4f}")
        print(f"Final Test W-Avg Recall: {final_test_metrics.get('weighted_recall', 0.0):.4f}")
        print(f"Final Test Micro F1: {final_micro_f1:.4f}")
        print(f"Final Test Micro Precision: {final_test_metrics.get('pair_precision', 0.0):.4f}")
        print(f"Final Test Micro Recall: {final_test_metrics.get('pair_recall', 0.0):.4f}")
        print(f"Final Test Predicted/True/Correct pairs: {final_test_metrics.get('num_predicted_pairs', 0)}/{final_test_metrics.get('num_true_pairs', 0)}/{final_test_metrics.get('num_correct_pairs', 0)}")

if __name__ == "__main__":
    main()