"""
PyTorch版本的评估指标
迁移自原TensorFlow版本的评估函数
"""
import numpy as np
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def calculate_prf(pred_y, true_y, doc_len, average='binary'):
    """
    计算Precision, Recall, F1
    对应原版本的cal_prf函数
    """
    # 将输入转换为numpy数组，确保形状正确
    pred_y = np.array(pred_y)
    true_y = np.array(true_y)
    doc_len = np.array(doc_len)
    
    # 如果是一维数组，直接计算
    if pred_y.ndim == 1:
        pred_num = np.sum(pred_y)
        true_num = np.sum(true_y)
        acc_num = np.sum(pred_y & true_y)
    else:
        # 原来的二维逻辑
        pred_num, acc_num, true_num = 0, 0, 0
        for i in range(pred_y.shape[0]):
            for j in range(doc_len[i]):
                if pred_y[i][j]:
                    pred_num += 1
                if true_y[i][j]:
                    true_num += 1
                if pred_y[i][j] and true_y[i][j]:
                    acc_num += 1
    
    p = acc_num / (pred_num + 1e-8)
    r = acc_num / (true_num + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    
    return p, r, f1


def calculate_prf_emocate(pred_y, true_y, doc_len):
    """
    计算情感类别的PRF
    对应原版本的cal_prf_emocate函数
    """
    conf_mat = np.zeros([7, 7])
    
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            conf_mat[true_y[i][j]][pred_y[i][j]] += 1
    
    # 计算每个类别的PRF
    p = np.diagonal(conf_mat / (np.sum(conf_mat, axis=0) + 1e-8).reshape(1, 7))
    r = np.diagonal(conf_mat / (np.sum(conf_mat, axis=1) + 1e-8).reshape(7, 1))
    f1 = 2 * p * r / (p + r + 1e-8)
    
    # 计算加权平均
    weight = np.sum(conf_mat, axis=1) / np.sum(conf_mat)
    w_avg_f1 = np.sum(f1 * weight)
    
    return np.append(f1, w_avg_f1)


def prf_2nd_step(pair_id_all, pair_id, pred_y):
    """
    Step2的评估函数
    对应原版本的prf_2nd_step函数
    """
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    
    def cal_prf(pair_id_all, pair_id):
        acc_num = len(pair_id_all) if len(pair_id_all) <= len(pair_id) else len(pair_id)
        true_num = len(pair_id_all)
        pred_num = len(pair_id)
        
        acc_num = 0
        for p in pair_id:
            if p in pair_id_all:
                acc_num += 1
        
        p = acc_num / (pred_num + 1e-8)
        r = acc_num / (true_num + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        return [p, r, f1]
    
    keep_rate = len(pair_id_filtered) / (len(pair_id) + 1e-8)
    filtered_results = cal_prf(pair_id_all, pair_id_filtered)
    all_results = cal_prf(pair_id_all, pair_id)
    
    return filtered_results + all_results + [keep_rate]


def prf_2nd_step_emocate(pair_id_all, pair_id, pred_y):
    """
    Step2情感类别评估函数
    对应原版本的prf_2nd_step_emocate函数
    """
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    
    keep_rate = len(pair_id_filtered) / (len(pair_id) + 1e-8)
    
    def cal_prf_emocate(pair_id_all, pair_id):
        conf_mat = np.zeros([7, 7])
        
        for p in pair_id:
            if p in pair_id_all:
                conf_mat[p[3]][p[3]] += 1
            else:
                conf_mat[0][p[3]] += 1
        
        for p in pair_id_all:
            if p not in pair_id:
                conf_mat[p[3]][0] += 1
        
        p = np.diagonal(conf_mat / (np.sum(conf_mat, axis=0) + 1e-8).reshape(1, 7))
        r = np.diagonal(conf_mat / (np.sum(conf_mat, axis=1) + 1e-8).reshape(7, 1))
        f1 = 2 * p * r / (p + r + 1e-8)
        
        weight0 = np.sum(conf_mat, axis=1)
        weight = weight0[1:] / np.sum(weight0[1:])
        w_avg_p = np.sum(p[1:] * weight)
        w_avg_r = np.sum(r[1:] * weight)
        w_avg_f = np.sum(f1[1:] * weight)
        
        # 不考虑占比较小的disgust/fear
        idx = [1, 4, 5, 6]  # anger, joy, sadness, surprise
        weight1 = weight0[idx]
        weight = weight1 / np.sum(weight1)
        
        w_avg_p_part = np.sum(p[idx] * weight)
        w_avg_r_part = np.sum(r[idx] * weight)
        w_avg_f_part = np.sum(f1[idx] * weight)
        
        results = list(f1[1:]) + [w_avg_p, w_avg_r, w_avg_f, w_avg_p_part, w_avg_r_part, w_avg_f_part]
        return results
    
    filtered_results = cal_prf_emocate(pair_id_all, pair_id_filtered)
    all_results = cal_prf_emocate(pair_id_all, pair_id)
    
    return filtered_results + all_results + [keep_rate]


def list_round(input_list, decimals=4):
    """四舍五入列表中的数值"""
    return [round(x, decimals) for x in input_list]


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.emotion_preds = []
        self.emotion_trues = []
        self.cause_preds = []
        self.cause_trues = []
        self.doc_lens = []
    
    def update(self, emotion_pred, emotion_true, cause_pred, cause_true, doc_len):
        """更新指标"""
        self.emotion_preds.append(emotion_pred)
        self.emotion_trues.append(emotion_true)
        self.cause_preds.append(cause_pred)
        self.cause_trues.append(cause_true)
        self.doc_lens.append(doc_len)
    
    def compute(self, choose_emocate=False):
        """计算最终指标"""
        emotion_preds = np.concatenate(self.emotion_preds)
        emotion_trues = np.concatenate(self.emotion_trues)
        cause_preds = np.concatenate(self.cause_preds)
        cause_trues = np.concatenate(self.cause_trues)
        doc_lens = np.concatenate(self.doc_lens)
        
        if choose_emocate:
            emotion_metrics = calculate_prf_emocate(emotion_preds, emotion_trues, doc_lens)
        else:
            emotion_metrics = calculate_prf(emotion_preds, emotion_trues, doc_lens)
        
        cause_metrics = calculate_prf(cause_preds, cause_trues, doc_lens)
        
        return {
            'emotion': emotion_metrics,
            'cause': cause_metrics
        }


def print_metrics(metrics, prefix="", choose_emocate=False):
    """打印指标"""
    if choose_emocate:
        emotion_f1s = metrics['emotion'][:-1]  # 除了最后的加权平均
        w_avg_f1 = metrics['emotion'][-1]
        
        emotion_names = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        print(f"{prefix}Emotion F1 by category:")
        for i, (name, f1) in enumerate(zip(emotion_names, emotion_f1s)):
            print(f"  {name}: {f1:.4f}")
        print(f"  Weighted Avg: {w_avg_f1:.4f}")
    else:
        emotion_p, emotion_r, emotion_f1 = metrics['emotion']
        print(f"{prefix}Emotion - P: {emotion_p:.4f}, R: {emotion_r:.4f}, F1: {emotion_f1:.4f}")
    
    cause_p, cause_r, cause_f1 = metrics['cause']
    print(f"{prefix}Cause - P: {cause_p:.4f}, R: {cause_r:.4f}, F1: {cause_f1:.4f}")


def format_results_for_step2(predictions, doc_ids, true_pairs):
    """
    格式化Conv的结果供Step2使用
    """
    # 这里需要根据具体需求实现
    # 将预测结果转换为Step2可以使用的格式
    formatted_results = []
    
    for i, doc_id in enumerate(doc_ids):
        # 处理每个文档的预测结果
        # 格式应该与原版本兼容
        pass
    
    return formatted_results