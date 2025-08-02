"""
Quick test of new pair-level metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from src.utils.config import Config
from src.data.pair_dataset import create_pair_datasets, collate_pair_samples
from src.models.pair_model import create_pair_model
from src.evaluation.pair_level_metrics import PairLevelMetrics

def test_metrics():
    """Test new pair-level metrics with a small batch"""
    print("🧪 Testing New Pair-Level Metrics...")
    
    config = Config("configs/base_config.yaml")
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    # Load datasets
    train_dataset, dev_dataset, test_dataset = create_pair_datasets(config, tokenizer)
    
    # Small dev loader
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_pair_samples
    )
    
    # Create model
    model = create_pair_model(config)
    device = torch.device(config.training.device)
    model.to(device)
    model.eval()
    
    # Test metrics
    metrics = PairLevelMetrics()
    
    with torch.no_grad():
        batch = next(iter(dev_loader))
        
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
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        print(f"Batch size: {len(predictions)}")
        print(f"Predictions: {predictions[:10]}")
        print(f"Labels: {labels.cpu().numpy()[:10]}")
        
        # Group by document and test metrics
        doc_data = {}
        
        for i, pair_id in enumerate(batch['pair_ids']):
            doc_id, emo_utt, cause_utt, emotion_cat = pair_id
            
            if doc_id not in doc_data:
                doc_data[doc_id] = {'predicted_pairs': [], 'true_pairs': []}
            
            if predictions[i] == 1:
                doc_data[doc_id]['predicted_pairs'].append((emo_utt, cause_utt, emotion_cat))
            
            if labels[i].item() == 1:
                doc_data[doc_id]['true_pairs'].append((emo_utt, cause_utt, emotion_cat))
        
        print(f"\nFound {len(doc_data)} documents in batch")
        
        # Update metrics
        for doc_id, data in doc_data.items():
            print(f"Doc {doc_id}: {len(data['predicted_pairs'])} pred, {len(data['true_pairs'])} true")
            metrics.update(
                loss=0.5,
                doc_id=doc_id,
                pair_predictions=data['predicted_pairs'],
                true_pairs=data['true_pairs']
            )
        
        # Compute results
        results = metrics.compute()
        
        print(f"\n📊 Metrics Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_metrics()