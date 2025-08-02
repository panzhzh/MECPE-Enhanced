"""
Debug Step1 prediction quality to identify potential data issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_step1_predictions():
    """Analyze Step1 prediction files for quality issues"""
    print("🔍 Analyzing Step1 Prediction Quality...")
    
    step1_dir = "experiments/checkpoints/step1_results"
    files = ["train_predictions.txt", "dev_predictions.txt", "test_predictions.txt"]
    
    for filename in files:
        filepath = os.path.join(step1_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
            
        print(f"\n📄 Analyzing {filename}:")
        
        # Analyze file content
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        docs = 0
        total_utts = 0
        emotion_preds_count = 0
        cause_preds_count = 0
        true_pairs_count = 0
        
        i = 0
        while i < len(lines):
            if i >= len(lines):
                break
                
            # Read document header
            header = lines[i].strip().split()
            if len(header) != 2:
                i += 1
                continue
                
            try:
                doc_id, doc_len = int(header[0]), int(header[1])
            except:
                i += 1
                continue
            
            docs += 1
            i += 1
            
            # Read true pairs
            if i >= len(lines):
                break
            true_pairs_line = lines[i].strip()
            try:
                true_pairs = eval(true_pairs_line) if true_pairs_line else []
                true_pairs_count += len(true_pairs)
            except:
                pass
            i += 1
            
            # Read utterances
            for j in range(doc_len):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split(' | ')
                if len(parts) >= 6:
                    emotion_pred = int(parts[1])
                    cause_pred = int(parts[2])
                    
                    total_utts += 1
                    if emotion_pred == 1:
                        emotion_preds_count += 1
                    if cause_pred == 1:
                        cause_preds_count += 1
                
                i += 1
        
        print(f"  Documents: {docs}")
        print(f"  Total utterances: {total_utts}")
        print(f"  Emotion predictions (1): {emotion_preds_count} ({emotion_preds_count/total_utts*100:.1f}%)")
        print(f"  Cause predictions (1): {cause_preds_count} ({cause_preds_count/total_utts*100:.1f}%)")
        print(f"  True emotion-cause pairs: {true_pairs_count}")
        
        # Check if predictions are reasonable
        if emotion_preds_count == 0:
            print("  ⚠️  WARNING: No emotion predictions!")
        if cause_preds_count == 0:
            print("  ⚠️  WARNING: No cause predictions!")
        if true_pairs_count == 0:
            print("  ⚠️  WARNING: No true pairs!")
        
        # Check prediction rates
        if emotion_preds_count / total_utts < 0.05:
            print("  ⚠️  WARNING: Very few emotion predictions (<5%)")
        if cause_preds_count / total_utts < 0.05:
            print("  ⚠️  WARNING: Very few cause predictions (<5%)")

if __name__ == "__main__":
    analyze_step1_predictions()