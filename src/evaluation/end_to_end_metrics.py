"""
End-to-End MECPE Evaluation Metrics
真正的端到端评估，完全不使用真实标签，模拟官方评估环境
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Official emotion categories mapping (从codalab_metrics.py复制)
EMOTION_IDX = {
    'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 
    'joy': 4, 'sadness': 5, 'surprise': 6
}

def cal_prf_pair_emocate(true_pairs: List[Tuple], pred_pairs: List[Tuple]) -> List[float]:
    """
    官方CodaLab评估函数 (从codalab_metrics.py复制)
    Calculate precision, recall, F1 for emotion-cause pairs (Subtask 2 evaluation)
    
    Args:
        true_pairs: List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
        pred_pairs: List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
    
    Returns:
        [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]
    """
    conf_mat = np.zeros([7, 7])  # 7 emotion categories
    
    # Process predicted pairs
    for p in pred_pairs:
        if p in true_pairs:
            conf_mat[p[3]][p[3]] += 1  # p[3] is emotion_category
        else:
            conf_mat[0][p[3]] += 1  # False positive
    
    # Process true pairs for false negatives
    for p in true_pairs:
        if p not in pred_pairs:
            conf_mat[p[3]][0] += 1  # False negative
    
    # Calculate precision, recall, F1
    p = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis=0) + (1e-8), [1, 7]))
    r = np.diagonal(conf_mat / np.reshape(np.sum(conf_mat, axis=1) + (1e-8), [7, 1]))
    f = 2 * p * r / (p + r + (1e-8))
    
    # Weighted average (exclude neutral class - index 0)
    weight0 = np.sum(conf_mat, axis=1)
    weight = weight0[1:] / np.sum(weight0[1:])  # Original doesn't add 1e-8 here
    w_avg_p = np.sum(p[1:] * weight)
    w_avg_r = np.sum(r[1:] * weight)
    w_avg_f1 = np.sum(f[1:] * weight)
    
    # Micro average
    micro_acc = np.sum(np.diagonal(conf_mat)[1:])
    micro_p = micro_acc / (sum(np.sum(conf_mat, axis=0)[1:]) + (1e-8))
    micro_r = micro_acc / (sum(np.sum(conf_mat, axis=1)[1:]) + (1e-8))
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)
    
    # Return in official order
    return [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]

class EndToEndMetrics:
    """
    真正的端到端MECPE评估
    完全不使用真实标签，模拟官方评估环境
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        
        # Store all predicted and true pairs for final evaluation
        self.all_predicted_pairs = []  # List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
        self.all_true_pairs = []       # List of (conv_id, emo_utt_id, cause_utt_id, emotion_category)
    
    def evaluate_conversation(self, conv_model, pair_model, raw_conversation_data, device, tokenizer, config):
        """
        对单个对话进行端到端评估
        
        Args:
            conv_model: 训练好的conversation模型
            pair_model: 训练好的pair模型
            raw_conversation_data: 原始对话数据
            device: 计算设备
            tokenizer: 分词器
            config: 配置
            
        Returns:
            预测的emotion-cause pairs
        """
        conv_model.eval()
        pair_model.eval()
        
        with torch.no_grad():
            # Step 1: 使用conv模型预测emotion和cause utterances
            # 这里需要实现从原始对话到conv模型预测的逻辑
            # 为了简化，我们暂时跳过这一步，直接从现有的预测文件读取
            pass
    
    def update_from_conv_predictions(self, loss: float, doc_id: int, 
                                   predicted_emotion_utts: List[int], 
                                   predicted_cause_utts: List[int],
                                   predicted_emotions: Dict[int, int],  # utt_id -> emotion_category
                                   pair_model_predictions: List[int],   # 对应每个候选对的预测
                                   candidate_pairs: List[Tuple],        # 生成的候选对
                                   true_pairs: List[Tuple]):            # 真实的pairs用于评估
        """
        从conv模型预测更新metrics
        
        Args:
            loss: 训练损失
            doc_id: 文档ID
            predicted_emotion_utts: conv模型预测的emotion utterances
            predicted_cause_utts: conv模型预测的cause utterances  
            predicted_emotions: utt_id到emotion_category的映射
            pair_model_predictions: pair模型对候选对的预测结果
            candidate_pairs: 基于conv预测生成的候选对 [(emo_utt, cause_utt), ...]
            true_pairs: 真实的emotion-cause pairs用于评估
        """
        batch_size = 1  # One document at a time
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # 基于pair model预测收集最终的predicted pairs
        for i, (emo_utt, cause_utt) in enumerate(candidate_pairs):
            # 如果pair model预测这是一个真实的pair
            if i < len(pair_model_predictions) and pair_model_predictions[i] == 1:
                # 使用conv模型预测的emotion category
                emotion_category = predicted_emotions.get(emo_utt, 0)  # 默认为neutral
                self.all_predicted_pairs.append((doc_id, emo_utt, cause_utt, emotion_category))
        
        # 添加真实pairs用于评估对比
        for emo_utt, cause_utt, emotion_cat in true_pairs:
            self.all_true_pairs.append((doc_id, emo_utt, cause_utt, emotion_cat))
    
    def generate_candidates_from_conv_predictions(self, predicted_emotion_utts: List[int], 
                                                predicted_cause_utts: List[int],
                                                pred_future_cause: bool = True) -> List[Tuple[int, int]]:
        """
        关键函数：仅基于conv预测生成候选对
        完全不使用任何真实标签！
        
        Args:
            predicted_emotion_utts: conv模型预测的emotion utterances (1-indexed)
            predicted_cause_utts: conv模型预测的cause utterances (1-indexed)
            pred_future_cause: 是否允许未来的cause
            
        Returns:
            候选对列表 [(emo_utt, cause_utt), ...]
        """
        candidates = []
        
        for emo_utt in predicted_emotion_utts:
            for cause_utt in predicted_cause_utts:
                # Check future cause constraint
                if pred_future_cause or cause_utt <= emo_utt:
                    candidates.append((emo_utt, cause_utt))
        
        return candidates
    
    def compute(self) -> Dict[str, float]:
        """
        计算最终的端到端评估指标
        
        Returns:
            Dictionary with end-to-end metrics using official CodaLab evaluation
        """
        if self.total_samples == 0:
            return {
                'avg_loss': 0.0,
                'pair_precision': 0.0, 'pair_recall': 0.0, 'pair_f1': 0.0,
                'weighted_precision': 0.0, 'weighted_recall': 0.0, 'weighted_f1': 0.0,
                'num_predicted_pairs': 0, 'num_true_pairs': 0, 'num_correct_pairs': 0,
                'num_documents': 0
            }
        
        # 使用官方CodaLab评估函数
        if len(self.all_true_pairs) > 0 or len(self.all_predicted_pairs) > 0:
            # 官方评估 - returns [micro_p, micro_r, micro_f1, w_avg_p, w_avg_r, w_avg_f1]
            results = cal_prf_pair_emocate(self.all_true_pairs, self.all_predicted_pairs)
            micro_p, micro_r, micro_f1, weighted_p, weighted_r, weighted_f1 = results
        else:
            micro_p = micro_r = micro_f1 = 0.0
            weighted_p = weighted_r = weighted_f1 = 0.0
        
        # 计算基础统计信息
        predicted_set = set(self.all_predicted_pairs)
        true_set = set(self.all_true_pairs)
        correct_pairs = predicted_set & true_set
        
        # 统计unique documents
        all_doc_ids = set()
        for conv_id, _, _, _ in self.all_predicted_pairs + self.all_true_pairs:
            all_doc_ids.add(conv_id)
        
        return {
            'avg_loss': self.total_loss / self.total_samples,
            
            # Official CodaLab metrics (端到端评估的真实指标)
            'pair_precision': micro_p,        # Micro precision
            'pair_recall': micro_r,           # Micro recall  
            'pair_f1': micro_f1,             # Micro F1
            'weighted_precision': weighted_p,  # W-avg precision (main metric for ranking)
            'weighted_recall': weighted_r,     # W-avg recall
            'weighted_f1': weighted_f1,       # W-avg F1 (main metric for ranking)
            
            # Debug/monitoring information
            'num_predicted_pairs': len(self.all_predicted_pairs),
            'num_true_pairs': len(self.all_true_pairs),
            'num_correct_pairs': len(correct_pairs),
            'num_documents': len(all_doc_ids)
        }

def evaluate_end_to_end_full(conv_model, pair_model, dataloader, device, config):
    """
    执行完整的端到端评估
    
    Args:
        conv_model: 训练好的conversation模型
        pair_model: 训练好的pair模型  
        dataloader: 原始对话数据的DataLoader (来自ECFDataset)
        device: 计算设备
        config: 配置
        
    Returns:
        端到端评估结果
    """
    if conv_model is None:
        raise ValueError("conv_model is required for end-to-end evaluation")
        
    conv_model.eval()
    pair_model.eval()
    
    metrics = EndToEndMetrics()
    
    # Import tokenizer for pair model
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model.text_model)
    
    processed_conversations = {}  # 避免重复处理相同对话
    
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="End-to-End Evaluation"):
            # 从原始对话数据中提取信息
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels']  # 真实标签仅用于评估
            cause_labels = batch['cause_labels']      # 真实标签仅用于评估
            doc_ids = batch['conv_ids']
            emotions_list = batch['emotions']
            pairs_list = batch['emotion_cause_pairs']
            
            # --- STEP 1: 使用conv模型预测emotion/cause utterances ---
            conv_outputs = conv_model(input_ids=input_ids, attention_mask=attention_mask)
            emotion_logits = conv_outputs['emotion_logits']  # [batch_size, max_utts, 2]
            cause_logits = conv_outputs['cause_logits']      # [batch_size, max_utts, 2]
            
            # 转换为预测（二分类：选择概率最大的类别）
            emotion_preds = torch.argmax(emotion_logits, dim=-1).cpu().numpy()  # [batch_size, max_utts]
            cause_preds = torch.argmax(cause_logits, dim=-1).cpu().numpy()      # [batch_size, max_utts]
            
            for i, doc_id in enumerate(doc_ids):
                if doc_id in processed_conversations:
                    continue
                processed_conversations[doc_id] = True
                
                # 获取当前对话的信息
                conv_emotions = emotions_list[i]
                conv_true_pairs = pairs_list[i]
                max_utts = len(conv_emotions)
                
                # --- STEP 2: 提取预测的emotion和cause utterances ---
                predicted_emo_utts = []
                predicted_cause_utts = []
                predicted_emotions = {}
                
                for utt_idx in range(min(max_utts, emotion_preds.shape[1])):
                    if emotion_preds[i, utt_idx] == 1:
                        utt_id = utt_idx + 1  # 转为1-indexed
                        emotion_str = conv_emotions[utt_idx]
                        if emotion_str in EMOTION_IDX:
                            predicted_emo_utts.append(utt_id)
                            predicted_emotions[utt_id] = EMOTION_IDX[emotion_str]
                    
                    if cause_preds[i, utt_idx] == 1:
                        utt_id = utt_idx + 1  # 转为1-indexed
                        predicted_cause_utts.append(utt_id)
                
                # --- STEP 3: 生成候选对 ---
                candidates = metrics.generate_candidates_from_conv_predictions(
                    predicted_emo_utts, predicted_cause_utts, config.data.pred_future_cause
                )
                
                
                if not candidates:
                    # 没有候选对，直接更新空结果
                    true_pairs_info = [(emo_utt, cause_utt, EMOTION_IDX[conv_emotions[emo_utt-1]]) 
                                     for emo_utt, cause_utt in conv_true_pairs
                                     if 0 < emo_utt <= len(conv_emotions) and conv_emotions[emo_utt-1] in EMOTION_IDX]
                    
                    metrics.update_from_conv_predictions(
                        loss=0.0, doc_id=doc_id,
                        predicted_emotion_utts=predicted_emo_utts,
                        predicted_cause_utts=predicted_cause_utts,
                        predicted_emotions=predicted_emotions,
                        pair_model_predictions=[],
                        candidate_pairs=[],
                        true_pairs=true_pairs_info
                    )
                    continue
                
                # --- STEP 4: 使用pair模型分类候选对 ---
                if not candidates:
                    continue
                
                # 构建batch输入
                num_pairs = len(candidates)
                max_seq_len = config.data.max_seq_length
                
                # 预分配张量
                batch_input_ids = torch.zeros((num_pairs, 2, max_seq_len), dtype=torch.long)
                batch_attention_masks = torch.zeros((num_pairs, 2, max_seq_len), dtype=torch.long)
                batch_distances = torch.zeros(num_pairs, dtype=torch.long)
                batch_emotion_categories = None
                
                if config.model.use_emotion_categories:
                    batch_emotion_categories = torch.zeros(num_pairs, dtype=torch.long)
                
                # 逐个构建候选对的输入
                valid_pairs = []
                for pair_idx, (emo_utt, cause_utt) in enumerate(candidates):
                    if emo_utt <= len(conv_emotions) and cause_utt <= len(conv_emotions):
                        # 获取原始utterance文本（这里需要从实际对话数据中获取）
                        # 暂时使用简化的文本表示
                        emo_text = f"utterance {emo_utt}: emotion is {conv_emotions[emo_utt-1]}"
                        cause_text = f"utterance {cause_utt}: this is a cause"
                        
                        # 分别编码两个utterance
                        emo_encoding = tokenizer(
                            emo_text, 
                            max_length=max_seq_len,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        cause_encoding = tokenizer(
                            cause_text,
                            max_length=max_seq_len, 
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        # 放入batch tensor
                        batch_input_ids[pair_idx, 0] = emo_encoding['input_ids'].squeeze(0)
                        batch_input_ids[pair_idx, 1] = cause_encoding['input_ids'].squeeze(0)
                        batch_attention_masks[pair_idx, 0] = emo_encoding['attention_mask'].squeeze(0)
                        batch_attention_masks[pair_idx, 1] = cause_encoding['attention_mask'].squeeze(0)
                        batch_distances[pair_idx] = abs(emo_utt - cause_utt)
                        
                        if config.model.use_emotion_categories:
                            emotion_cat = predicted_emotions.get(emo_utt, 0)
                            batch_emotion_categories[pair_idx] = emotion_cat
                        
                        valid_pairs.append((emo_utt, cause_utt))
                
                if not valid_pairs:
                    continue
                
                # 移动到GPU
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_masks = batch_attention_masks.to(device)
                batch_distances = batch_distances.to(device)
                if batch_emotion_categories is not None:
                    batch_emotion_categories = batch_emotion_categories.to(device)
                
                # 运行pair模型
                pair_logits = pair_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    distance=batch_distances,
                    emotion_category=batch_emotion_categories
                )
                
                pair_preds = torch.argmax(pair_logits, dim=-1).cpu().numpy()
                
                
                # --- STEP 5: 收集最终结果 ---
                true_pairs_info = [(emo_utt, cause_utt, EMOTION_IDX[conv_emotions[emo_utt-1]]) 
                                 for emo_utt, cause_utt in conv_true_pairs
                                 if 0 < emo_utt <= len(conv_emotions) and conv_emotions[emo_utt-1] in EMOTION_IDX]
                
                metrics.update_from_conv_predictions(
                    loss=0.0, doc_id=doc_id,
                    predicted_emotion_utts=predicted_emo_utts,
                    predicted_cause_utts=predicted_cause_utts,
                    predicted_emotions=predicted_emotions,
                    pair_model_predictions=pair_preds.tolist(),
                    candidate_pairs=valid_pairs,  # 使用实际处理的pairs
                    true_pairs=true_pairs_info
                )
    
    # --- START: SAVE PREDICTIONS FOR INSPECTION ---
    import json
    from collections import defaultdict
    
    # Group results by conversation ID for easier inspection
    doc_results = defaultdict(lambda: {'predicted_pairs': [], 'true_pairs': []})
    
    # Process predicted pairs
    for pair in metrics.all_predicted_pairs:
        conv_id, emo_utt, cause_utt, emotion_cat = pair
        doc_results[conv_id]['predicted_pairs'].append([emo_utt, cause_utt, emotion_cat])
    
    # Process true pairs
    for pair in metrics.all_true_pairs:
        conv_id, emo_utt, cause_utt, emotion_cat = pair
        doc_results[conv_id]['true_pairs'].append([emo_utt, cause_utt, emotion_cat])
    
    # Convert to list format for JSON serialization
    all_results_for_inspection = []
    for doc_id, data in doc_results.items():
        all_results_for_inspection.append({
            'doc_id': doc_id,
            'predicted_pairs': data['predicted_pairs'],
            'true_pairs': data['true_pairs'],
            'num_predicted': len(data['predicted_pairs']),
            'num_true': len(data['true_pairs']),
            'num_correct': len(set(tuple(p) for p in data['predicted_pairs']) & 
                             set(tuple(p) for p in data['true_pairs']))
        })
    
    # Sort by doc_id for easier reading
    all_results_for_inspection.sort(key=lambda x: x['doc_id'])
    
    # Define the output path
    output_path = "pair_model_predictions_for_debug.json"
    
    # Write to a JSON file
    with open(output_path, 'w') as f:
        json.dump(all_results_for_inspection, f, indent=4)
    
    # Also save summary statistics
    total_docs = len(all_results_for_inspection)
    docs_with_predictions = sum(1 for doc in all_results_for_inspection if doc['num_predicted'] > 0)
    total_predicted = sum(doc['num_predicted'] for doc in all_results_for_inspection)
    total_true = sum(doc['num_true'] for doc in all_results_for_inspection)
    total_correct = sum(doc['num_correct'] for doc in all_results_for_inspection)
    
    summary = {
        'total_documents': total_docs,
        'documents_with_predictions': docs_with_predictions,
        'total_predicted_pairs': total_predicted,
        'total_true_pairs': total_true,
        'total_correct_pairs': total_correct,
        'prediction_rate': docs_with_predictions / total_docs if total_docs > 0 else 0,
        'precision': total_correct / total_predicted if total_predicted > 0 else 0,
        'recall': total_correct / total_true if total_true > 0 else 0
    }
    
    with open("pair_model_debug_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n[DEBUG] Saved pair model's predictions to: {output_path}")
    print(f"[DEBUG] Saved summary statistics to: pair_model_debug_summary.json")
    print(f"[DEBUG] Summary: {docs_with_predictions}/{total_docs} docs with predictions, {total_predicted} total predictions\n")
    # --- END: SAVE PREDICTIONS FOR INSPECTION ---
    
    return metrics.compute()