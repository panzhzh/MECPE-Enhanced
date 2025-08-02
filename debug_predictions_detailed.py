"""
Deep debug of model predictions after training
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import numpy as np
from collections import Counter

from src.utils.config import Config
from src.data.pair_dataset import create_pair_datasets, collate_pair_samples
from src.models.pair_model import create_pair_model

def analyze_predictions():
    """Analyze model predictions in detail"""
    print("🔬 Deep Analysis of Model Predictions")
    
    config = Config("configs/base_config.yaml")
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    # Load datasets
    train_dataset, dev_dataset, test_dataset = create_pair_datasets(config, tokenizer)
    
    # Small dev loader
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_pair_samples
    )
    
    # Create model (untrained for comparison)
    model = create_pair_model(config)
    device = torch.device(config.training.device)
    model.to(device)
    model.eval()
    
    print("📊 Analyzing Dev Set Predictions...")
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if i >= 10:  # Only first 10 batches
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            distance = batch['distance'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                distance=distance
            )
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels_np.tolist())
            all_logits.extend(logits_np.tolist())
    
    # Analyze results
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    logits = np.array(all_logits)
    
    print(f"\n📈 Prediction Analysis (first 640 samples):")
    print(f"  Label distribution: {Counter(labels)}")
    print(f"  Prediction distribution: {Counter(predictions)}")
    
    # Check logits distribution
    logits_class0 = logits[:, 0]  # Negative class logits
    logits_class1 = logits[:, 1]  # Positive class logits
    
    print(f"\n🎯 Logits Analysis:")
    print(f"  Class 0 (neg) logits - mean: {logits_class0.mean():.4f}, std: {logits_class0.std():.4f}")
    print(f"  Class 1 (pos) logits - mean: {logits_class1.mean():.4f}, std: {logits_class1.std():.4f}")
    print(f"  Class 0 > Class 1 rate: {np.mean(logits_class0 > logits_class1)*100:.1f}%")
    
    # Show some examples
    print(f"\n📝 Sample Logits (first 10):")
    for i in range(min(10, len(logits))):
        print(f"  Sample {i}: logits=[{logits[i,0]:.3f}, {logits[i,1]:.3f}], pred={predictions[i]}, true={labels[i]}")
    
    # Calculate metrics manually
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions)
    
    print(f"\n🎲 Manual Metrics Calculation:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Check if model is learning anything
    print(f"\n💡 Model Behavior Analysis:")
    if len(set(predictions)) == 1:
        print(f"  ❌ Model predicts ONLY class {predictions[0]} - complete bias!")
    else:
        print(f"  ✅ Model predicts both classes")
    
    # Check weight status
    print(f"\n⚖️ Loss Weight Analysis:")
    print(f"  Configured class weights: [1.0, 4.6]")
    print(f"  Expected effect: Positive class should be favored")

if __name__ == "__main__":
    analyze_predictions()