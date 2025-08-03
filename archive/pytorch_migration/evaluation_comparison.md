# MECPE-Enhanced 评估逻辑对比分析

## 概述

本文档对比分析了MECPE-Enhanced项目中内置评估函数与官方CodaLab评估脚本的逻辑差异。

## 评估方式对比

### 内置评估 (Conv + Step2)
- **位置**: `utils/pre_data_bert.py`
- **核心函数**: 
  - `cal_prf()` - Conv情感/原因识别评估
  - `prf_2nd_step()` - Step2配对评估
  - `prf_2nd_step_emocate()` - Step2情感分类配对评估

### 官方CodaLab评估
- **位置**: `CodaLab/evaluation/evaluate.py`
- **核心函数**:
  - `evaluate_1_2()` - Subtask 1评估 (span-level)
  - `evaluate_2_2()` - Subtask 2评估 (pair-level)

## 核心逻辑对比

### ✅ 一致的部分

#### 1. 混淆矩阵计算逻辑
两者都使用相同的7×7混淆矩阵计算情感类别的P/R/F1：

**内置评估**:
```python
for p in pair_id:
    if p in pair_id_all:
        conf_mat[p[3]][p[3]] += 1  # 正确预测
    else:
        conf_mat[0][p[3]] += 1     # 错误预测
```

**官方评估**:
```python
for p in pred_pairs:
    if p in true_pairs:
        conf_mat[p[3]][p[3]] += 1  # 正确预测
    else:
        conf_mat[0][p[3]] += 1     # 错误预测
```

#### 2. 加权F1计算
两者都计算weighted average F1，权重基于真实样本分布：
```python
weight = weight0[1:] / np.sum(weight0[1:])
w_avg_f1 = np.sum(f[1:] * weight)
```

#### 3. 情感类别处理
都支持6种情感类别（除neutral外）：
- anger, disgust, fear, joy, sadness, surprise

### ⚠️ 关键差异

| 方面 | 内置评估 | 官方CodaLab评估 |
|------|---------|---------------|
| **评估粒度** | **话语级配对** | **Span级配对** |
| **数据格式** | `[doc_id, emo_id, cau_id, emotion]` | `[conv_id, emo_id, cau_id, span_start, span_end, emotion]` |
| **Span匹配** | ❌ 不支持 | ✅ 支持strict/fuzzy/proportional |
| **评估方式** | 简单配对匹配 | 多种匹配模式 |
| **输入格式** | 内部数据结构 | 标准JSON格式 |

## 具体差异分析

### 1. Span-level评估缺失

**内置评估**: 
- 只判断`(U5_Disgust, U3)` 配对是否正确
- 不考虑cause span在话语中的具体位置

**官方评估**: 
- 需要判断span位置 `U3_2_8` 是否准确匹配
- 支持span位置的精确评估

### 2. 匹配模式差异

**官方评估支持3种匹配模式:**

#### Strict Match
```python
if [start_cur, end_cur, emo] in true_spans_pos_dict[cur_key]:
    true_spans_pos_dict[cur_key].remove([start_cur, end_cur, emo])
    return True
```

#### Fuzzy Match  
```python
if emo == emo_y and not(end_cur<=t_start or start_cur>=t_end):
    true_spans_pos_dict[cur_key].remove([t_start, t_end, emo_y])
    return True
```

#### Proportional Match
```python
cur_match_score = cur_match_length / float(cur_gold_length)
# 按重叠比例计算分数
```

### 3. 数据结构不兼容

**内置格式:**
```python
pair_id = [1, 5, 3, 2]  # [doc_id, emo_utt, cau_utt, emotion_idx]
```

**官方格式:**
```python
pred_pair = [1, 5, 3, 2, 8, 2]  # [conv_id, emo_utt, cau_utt, span_start, span_end, emotion_idx]
```

### 4. 输入数据格式

**内置评估**: 使用内部数据结构
```python
pair_id_all = [[doc_id, emo_id, cau_id, emotion_idx], ...]
```

**官方评估**: 使用标准JSON格式
```json
{
  "conversation_ID": 1,
  "emotion-cause_pairs": [
    ["U5_Disgust", "U5_2_8"],
    ["U3_Joy", "U1_0_5"]
  ]
}
```

## 评估指标对比

### 内置评估指标
- **Conv**: Precision, Recall, F1 (二元分类)
- **Step2**: 
  - 基础P/R/F1
  - 加权平均F1 (7类情感)
  - 部分加权F1 (4类主要情感，排除disgust/fear)

### 官方评估指标
- **Subtask 1**: 
  - Weighted Strict P/R/F1
  - Weighted Proportional P/R/F1
  - Micro P/R/F1
- **Subtask 2**:
  - Precision/Recall/F1 (配对级别)
  - Weighted Precision/Recall/F1

## 结论

### 评估逻辑一致性
✅ **配对层面**: 两者核心逻辑完全一致
- 混淆矩阵计算方式相同
- 加权F1计算公式相同
- 情感类别处理方式相同

❌ **Span层面**: 内置评估缺少span位置评估
- 官方评估更严格，要求span级别的精确匹配
- 内置评估只进行话语级别的配对判断

### 最终分数差异
⚠️ **可能存在显著差异**，因为:
1. 官方评估在span级别更严格
2. 支持多种匹配模式，评估更全面
3. 数据格式和处理流程不同

### 建议
如果要获得与官方评估一致的结果，需要：

1. **实现span-level预测**: 在Step2中预测cause span的具体位置
2. **格式转换**: 将内部数据结构转换为官方JSON格式
3. **添加span匹配**: 实现strict/fuzzy/proportional匹配模式
4. **完善评估流程**: 建立从Step2输出到官方评估的完整pipeline

## 技术实现路径

### 短期方案
- 编写格式转换脚本，将现有输出转换为官方JSON格式
- 使用官方evaluate.py进行最终评估

### 长期方案  
- 在Step2中集成span位置预测
- 实现与官方评估完全一致的内置评估函数
- 建立端到端的评估pipeline

---

*分析日期: 2025-08-01*  
*项目: MECPE-Enhanced*