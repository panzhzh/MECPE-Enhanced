"""
Simple baseline training script for MECPE
Clean and minimal implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import Config
from src.data.conv_dataset import ECFDataset, collate_conversations
from src.models.conv_model import create_baseline_model
from src.evaluation.conv_metrics import Step1Metrics

def run_epoch(model, dataloader, optimizer=None, device='cpu', is_train=True):
    """Run one epoch (train or eval)"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    metrics = Step1Metrics()
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Training" if is_train else "Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            cause_labels = batch['cause_labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss_dict = model.compute_loss(
                emotion_logits=outputs['emotion_logits'],
                cause_logits=outputs['cause_logits'],
                emotion_labels=emotion_labels,
                cause_labels=cause_labels,
                attention_mask=attention_mask
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass (only in training)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update metrics
            metrics.update(
                loss=loss.item(),
                emotion_logits=outputs['emotion_logits'],
                cause_logits=outputs['cause_logits'],
                emotion_labels=emotion_labels,
                cause_labels=cause_labels,
                attention_mask=attention_mask
            )
    
    return metrics.compute()

def main():
    """Main training function"""
    print("🚀 Starting Simple Baseline Training...")
    
    # Load config
    config = Config("configs/base_config.yaml")
    print(f"Config: {config.experiment.name}")
    print(f"Model: {config.model.text_model} (dim: {config.model.text_hidden_size})")
    print(f"Device: {config.training.device}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = ECFDataset(split="train", config=config, load_video=False, load_audio=False)
    test_dataset = ECFDataset(split="test", config=config, load_video=False, load_audio=False)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, 
        shuffle=True, collate_fn=collate_conversations
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_conversations
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_baseline_model(config)
    device = torch.device(config.training.device)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Training loop
    print(f"\n🎯 Training for {config.training.num_epochs} epochs...")
    
    best_f1 = 0.0
    
    for epoch in range(config.training.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ===")
        
        # Training
        train_metrics = run_epoch(model, train_loader, optimizer, device, is_train=True)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Print conversation-level metrics
        print(f"Train Emotion F1: {train_metrics['emotion_f1']:.4f}")
        print(f"Train Cause F1: {train_metrics['cause_f1']:.4f}")
        print(f"Train Avg F1: {train_metrics['avg_f1']:.4f}")
        
        # Test evaluation
        test_metrics = run_epoch(model, test_loader, device=device, is_train=False)
        print(f"Test Loss: {test_metrics['avg_loss']:.4f}")
        
        # Print conversation-level metrics
        print(f"Test Emotion F1: {test_metrics['emotion_f1']:.4f}")
        print(f"Test Cause F1: {test_metrics['cause_f1']:.4f}")
        print(f"Test Avg F1: {test_metrics['avg_f1']:.4f}")
        
        # Save best model based on conversation-level avg F1
        main_metric = test_metrics['avg_f1']
        if main_metric > best_f1:
            best_f1 = main_metric
            torch.save(model.state_dict(), 
                      os.path.join(config.training.save_dir, 'best_model.pt'))
            print(f"💾 Saved new best model (F1: {best_f1:.4f})")
    
    print(f"\n🎉 Training completed! Best Test F1: {best_f1:.4f}")
    
    # Generate Step2 input files from best model predictions
    print("\n📝 Generating Step2 input files...")
    generate_step2_input_files(model, train_loader, dev_loader, test_loader, config, device)

def generate_step2_input_files(model, train_loader, dev_loader, test_loader, config, device):
    """Generate Step2 input files from Step1 predictions"""
    model.eval()
    
    # Create output directory
    step2_input_dir = os.path.join(config.training.save_dir, "step1_results")
    os.makedirs(step2_input_dir, exist_ok=True)
    
    # Process each dataset
    datasets = [
        (train_loader, "train_predictions.txt", "train"),
        (dev_loader, "dev_predictions.txt", "dev"), 
        (test_loader, "test_predictions.txt", "test")
    ]
    
    emotion_idx_rev = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'sadness', 6: 'surprise'}
    
    for dataloader, filename, split_name in datasets:
        output_path = os.path.join(step2_input_dir, filename)
        print(f"Generating {split_name} predictions -> {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Processing {split_name}"):
                    # Move to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    emotion_labels = batch['emotion_labels'].to(device)
                    cause_labels = batch['cause_labels'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Get predictions
                    emotion_preds = torch.argmax(outputs['emotion_logits'], dim=-1)  # [batch_size, max_utts]
                    cause_preds = torch.argmax(outputs['cause_logits'], dim=-1)     # [batch_size, max_utts]
                    
                    # Process each conversation in batch
                    for i in range(input_ids.size(0)):
                        conv_id = batch.get('conv_ids', [f"conv_{i}"])[i] if 'conv_ids' in batch else f"conv_{i}"
                        emotions = batch.get('emotions', [[]])[i] if 'emotions' in batch else []
                        emotion_cause_pairs = batch.get('emotion_cause_pairs', [[]])[i] if 'emotion_cause_pairs' in batch else []
                        
                        # Calculate document length (number of non-padded utterances)
                        if len(attention_mask.shape) == 3:
                            doc_len = (attention_mask[i].sum(dim=-1) > 0).sum().item()
                        else:
                            doc_len = (attention_mask[i] > 0).sum().item()
                        
                        # Write document header
                        f.write(f"{conv_id} {doc_len}\n")
                        
                        # Write true emotion-cause pairs
                        f.write(f"{emotion_cause_pairs}\n")
                        
                        # Write utterances with predictions
                        for j in range(doc_len):
                            emotion_pred = emotion_preds[i, j].item()
                            cause_pred = cause_preds[i, j].item()
                            
                            # Get true emotion category
                            emotion_true_idx = emotion_labels[i, j].item() if j < emotion_labels.size(1) else 0
                            emotion_true = emotion_idx_rev.get(emotion_true_idx, 'neutral')
                            
                            # Get utterance text (placeholder - in real implementation this would come from dataset)
                            utt_text = f"utterance_{j+1}_text"
                            speaker = f"Speaker_{j%3}"  # Placeholder
                            
                            # Write utterance line: utt_id | emotion_pred | cause_pred | speaker | emotion_true | text
                            f.write(f"{j+1} | {emotion_pred} | {cause_pred} | {speaker} | {emotion_true} | {utt_text}\n")
        
        print(f"✅ Generated {filename}")

if __name__ == "__main__":
    main()