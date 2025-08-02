"""
Unified Training Script for MECPE
Supports Step1, Step2, or both sequentially
"""
import sys
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import RobertaTokenizer

from src.utils.config import Config
from src.data.dataset import ECFDataset, collate_conversations
from src.data.step2_dataset import create_step2_datasets, collate_step2_pairs
from src.models.baseline_model import create_baseline_model
from src.models.step2_model import create_step2_model
from src.evaluation.metrics import UnifiedMetrics
from src.evaluation.step2_metrics import Step2Metrics

def run_step1_epoch(model, dataloader, optimizer=None, device='cpu', is_train=True):
    """Run one epoch for Step1 (emotion/cause utterance classification)"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    metrics = UnifiedMetrics()
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Step1 Training" if is_train else "Step1 Evaluating"):
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
                attention_mask=attention_mask,
                conv_ids=batch.get('conv_ids'),
                emotions=batch.get('emotions'),
                emotion_cause_pairs=batch.get('emotion_cause_pairs')
            )
    
    return metrics.compute()

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

def train_step1(config):
    """Train Step1: Emotion/Cause Utterance Classification"""
    print("🚀 Starting Step1 Training (Emotion/Cause Utterance Classification)...")
    
    print(f"Config: {config.experiment.name}")
    print(f"Model: {config.model.text_model}")
    print(f"Device: {config.training.device}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = ECFDataset(split="train", config=config, load_video=False, load_audio=False)
    dev_dataset = ECFDataset(split="dev", config=config, load_video=False, load_audio=False)
    test_dataset = ECFDataset(split="test", config=config, load_video=False, load_audio=False)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Dev: {len(dev_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, 
        shuffle=True, collate_fn=collate_conversations
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_conversations
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_conversations
    )
    
    # Create model
    print("\n🧠 Creating Step1 model...")
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
        print(f"\n=== Step1 Epoch {epoch + 1}/{config.training.num_epochs} ===")
        
        # Training
        train_metrics = run_step1_epoch(model, train_loader, optimizer, device, is_train=True)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Print Step1 metrics
        if 'step1_emotion_f1' in train_metrics:
            print(f"Train Step1 Emotion F1: {train_metrics['step1_emotion_f1']:.4f}")
            print(f"Train Step1 Cause F1: {train_metrics['step1_cause_f1']:.4f}")
            print(f"Train Step1 Avg F1: {train_metrics['step1_avg_f1']:.4f}")
        
        # Print CodaLab metrics
        if 'codalab_weighted_f1' in train_metrics:
            print(f"Train CodaLab Weighted F1: {train_metrics['codalab_weighted_f1']:.4f}")
            print(f"Train CodaLab Pair F1: {train_metrics['codalab_pair_f1']:.4f}")
        
        # Test evaluation
        test_metrics = run_step1_epoch(model, test_loader, device=device, is_train=False)
        print(f"Test Loss: {test_metrics['avg_loss']:.4f}")
        
        # Print Step1 metrics
        if 'step1_emotion_f1' in test_metrics:
            print(f"Test Step1 Emotion F1: {test_metrics['step1_emotion_f1']:.4f}")
            print(f"Test Step1 Cause F1: {test_metrics['step1_cause_f1']:.4f}")
            print(f"Test Step1 Avg F1: {test_metrics['step1_avg_f1']:.4f}")
        
        # Print CodaLab metrics
        if 'codalab_weighted_f1' in test_metrics:
            print(f"Test CodaLab Weighted F1: {test_metrics['codalab_weighted_f1']:.4f}")
            print(f"Test CodaLab Pair F1: {test_metrics['codalab_pair_f1']:.4f}")
        
        # Save best model based on Step1 avg F1 or CodaLab weighted F1
        main_metric = test_metrics.get('step1_avg_f1', test_metrics.get('codalab_weighted_f1', 0.0))
        if main_metric > best_f1:
            best_f1 = main_metric
            torch.save(model.state_dict(), 
                      os.path.join(config.training.save_dir, 'best_step1_model.pt'))
            print(f"💾 Saved new best Step1 model (F1: {best_f1:.4f})")
    
    print(f"\n🎉 Step1 Training completed! Best Test F1: {best_f1:.4f}")
    
    # Generate Step2 input files from best model predictions
    print("\n📝 Generating Step2 input files...")
    generate_step2_input_files(model, train_loader, dev_loader, test_loader, config, device)
    
    return model, best_f1

def run_step2_epoch(model, dataloader, optimizer=None, device='cpu', is_train=True, config=None):
    """Run one epoch for Step2 (emotion-cause pair classification) with full evaluation"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    # Initialize Step2 metrics
    from src.evaluation.step2_metrics import Step2Metrics
    use_emotion_categories = config.model.use_emotion_categories if config else False
    metrics = Step2Metrics(use_emotion_categories=use_emotion_categories)
    
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_pair_ids = []
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Step2 Training" if is_train else "Step2 Evaluating"):
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
            
            # Collect predictions and data for Step2 evaluation
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_pair_ids.extend(batch['pair_ids'])
            
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate Step2 metrics using the complete evaluation
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if not is_train:
        # For evaluation, get ground truth from dataset
        dataset = dataloader.dataset
        dataset_true_pairs = dataset.pair_id_all  # All true pairs from ground truth
        
        # IMPORTANT: all_pair_ids must match the order of dataset.pair_id since we collected them in batch order
        # The predictions in all_predictions correspond to pairs in all_pair_ids
        
        # Update Step2 metrics with all collected data
        metrics.update(
            loss=avg_loss,
            pair_id_all=dataset_true_pairs,
            pair_id=all_pair_ids,  # This should match the order we processed batches
            pred_y=np.array(all_predictions)
        )
        
        step2_results = metrics.compute()
    else:
        # For training, just return loss
        step2_results = {'avg_loss': avg_loss}
    
    return step2_results

def train_step2(config):
    """Train Step2: Emotion-Cause Pair Classification"""
    print("🚀 Starting Step2 Training (Emotion-Cause Pair Classification)...")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    # Create datasets
    print("\n📊 Loading Step2 datasets...")
    try:
        train_dataset, dev_dataset, test_dataset = create_step2_datasets(config, tokenizer)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Please run Step1 training first to generate the required prediction files.")
        return None, 0.0
    
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
        print(f"\n=== Step2 Epoch {epoch + 1}/{config.training.step2_num_epochs} ===")
        
        # Training
        train_metrics = run_step2_epoch(model, train_loader, optimizer, device, is_train=True, config=config)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        
        # Evaluation on test set (using test instead of dev)
        test_metrics = run_step2_epoch(model, test_loader, device=device, is_train=False, config=config)
        
        # Print Step2 metrics
        if config.model.use_emotion_categories:
            main_f1 = test_metrics.get('filtered_weighted_f1', 0.0)
            print(f"Test F1emotion: {test_metrics.get('filtered_f1_emotion', 0.0):.4f}")
            print(f"Test F1cause: {test_metrics.get('filtered_f1_cause', 0.0):.4f}")
            print(f"Test F1pair: {main_f1:.4f}")
        else:
            main_f1 = test_metrics.get('filtered_f1', 0.0)
            print(f"Test F1emotion: {test_metrics.get('filtered_f1_emotion', 0.0):.4f}")
            print(f"Test F1cause: {test_metrics.get('filtered_f1_cause', 0.0):.4f}")
            print(f"Test F1pair: {main_f1:.4f}")
        
        print(f"Test Keep Rate: {test_metrics.get('keep_rate', 0.0):.4f}")
        
        # Save best model
        if main_f1 > best_f1:
            best_f1 = main_f1
            best_epoch = epoch + 1
            
            # Save model
            model_save_path = os.path.join(config.training.save_dir, 'best_step2_model.pt')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved new best Step2 model (F1pair: {best_f1:.4f})")
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.1f}s")
    
    print(f"\n🎉 Step2 Training completed!")
    print(f"Best Test F1pair: {best_f1:.4f} (Epoch {best_epoch})")
    
    return model, best_f1

def main():
    """Main training function"""
    # 选择运行模式：1=Step1, 2=Step2, 0=全部
    RUN_MODE = 2  # 🔥 修改这里：1(仅Step1), 2(仅Step2), 0(全部)
    
    # Load config
    config = Config("configs/base_config.yaml")
    
    print("="*60)
    print("🎯 MECPE Training Pipeline")
    print("="*60)
    
    if RUN_MODE == 1:
        print("Running Step1 only...")
        step1_model, step1_f1 = train_step1(config)
        print(f"\n✅ Step1 completed with F1: {step1_f1:.4f}")
        
    elif RUN_MODE == 2:
        print("Running Step2 only...")
        step2_model, step2_f1 = train_step2(config)
        if step2_model is not None:
            print(f"\n✅ Step2 completed with F1: {step2_f1:.4f}")
        
    else:  # RUN_MODE == 0
        print("Running complete pipeline: Step1 → Step2...")
        
        # Run Step1
        print("\n" + "="*40)
        print("🥇 PHASE 1: Step1 Training")
        print("="*40)
        step1_model, step1_f1 = train_step1(config)
        print(f"\n✅ Step1 completed with F1: {step1_f1:.4f}")
        
        # Run Step2
        print("\n" + "="*40)
        print("🥈 PHASE 2: Step2 Training")
        print("="*40)
        step2_model, step2_f1 = train_step2(config)
        
        if step2_model is not None:
            print(f"\n✅ Step2 completed with F1: {step2_f1:.4f}")
            
            print("\n" + "="*60)
            print("🎉 COMPLETE PIPELINE FINISHED!")
            print("="*60)
            print(f"📊 Final Results:")
            print(f"   Step1 F1: {step1_f1:.4f}")
            print(f"   Step2 F1: {step2_f1:.4f}")
            print("="*60)
        else:
            print("\n❌ Step2 training failed")

if __name__ == "__main__":
    main()