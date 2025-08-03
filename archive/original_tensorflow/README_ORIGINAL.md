# 原始TensorFlow版本归档

## 📋 概述
这是MECPE-Enhanced项目的原始TensorFlow实现版本，基于2018-2019年的技术栈。

## 📁 文件说明

### 核心训练脚本
- `conv.py` - TensorFlow版本的Conv（情感和原因识别）训练脚本
- `step2.py` - TensorFlow版本的Step2（情感-原因配对）训练脚本

### BERT支持模块
- `bert/modeling.py` - BERT模型定义
- `bert/optimization.py` - 优化器和学习率调度
- `bert/tokenization.py` - BERT分词器

### 工具函数
- `utils/pre_data_bert.py` - 数据预处理工具
- `utils/tf_funcs.py` - TensorFlow工具函数

### 评估框架
- `CodaLab/` - 原始评估系统，与SemEval-2024兼容

## 🔧 技术栈
- **深度学习框架**: TensorFlow 1.x
- **文本编码**: BiLSTM + Attention
- **多模态**: 预处理特征拼接
- **数据处理**: TensorFlow placeholders和sessions

## 📊 性能基准
- Emotion F1: ~0.72
- Cause F1: ~0.65

## ⚠️ 使用说明
此版本已归档，不再维护。如需运行：
1. 安装TensorFlow 1.x环境
2. 确保CUDA兼容性
3. 使用原始数据格式

## 🔗 相关资源
- 原始论文: [SemEval-2024 Task 3]
- 数据集: ECF 1.0
- 评估: CodaLab平台

---
**归档时间**: 2024年8月
**归档原因**: 技术栈现代化升级