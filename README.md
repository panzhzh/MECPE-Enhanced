# MECPE-Enhanced: PyTorch Implementation

## 🎯 Project Status

**Phase 1 完成**: Successfully migrated original TensorFlow Step1 evaluation to PyTorch

### ✅ Current Features

- **PyTorch实现**: 完整的MECPE Step1话语级别评估系统
- **双重评估**: Step1 metrics (情感/原因话语识别) + CodaLab metrics (情感-原因对识别)
- **多模态支持**: 文本、音频、视频数据处理 (当前仅文本已测试)
- **完整数据流**: 从ECF数据集到模型训练的完整pipeline

### 🏗️ Architecture

```
src/
├── data/dataset.py          # ECF数据集加载和预处理
├── models/baseline_model.py # BERT/BiLSTM基线模型
├── evaluation/
│   ├── step1_metrics.py     # Step1话语级别评估 (移植自TensorFlow)
│   ├── codalab_metrics.py   # 官方CodaLab对级别评估
│   └── metrics.py           # 统一评估接口
└── utils/config.py          # 配置管理
```

### 🚀 Quick Start

```bash
# 训练基线模型
python scripts/train_baseline.py

# 输出示例:
# Test Step1 Emotion F1: 0.7736  (情感话语识别F1)
# Test Step1 Cause F1: 0.7251    (原因话语识别F1)  
# Test CodaLab Weighted F1: 0.2493 (情感-原因对F1)
```

### 📊 Key Metrics

- **F1emotion**: 情感话语识别F1分数
- **F1cause**: 原因话语识别F1分数  
- **F1pair**: 情感-原因对识别F1分数

### 🔧 Configuration

编辑 `configs/base_config.yaml` 调整模型和训练参数。

### 📝 Data Format

项目使用ECF (Emotion-Cause in Friends)数据集，包含:
- 对话文本数据
- 情感标签 (7类: neutral, anger, disgust, fear, joy, sadness, surprise)
- 原因标签 (二分类: cause/non-cause)
- 情感-原因对标注

## 🎯 Next Steps

- [ ] Step2 implementation (情感-原因对提取)
- [ ] 音频和视频模态集成
- [ ] 完整的多模态评估框架
- [ ] 超参数优化

---

**里程碑**: 成功完成TensorFlow到PyTorch的Step1评估系统迁移 ✨
