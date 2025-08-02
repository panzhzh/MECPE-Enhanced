# 项目归档计划

## 📁 归档结构

### 保留文件（不移动）
- `data/` - 数据文件夹（原始数据和特征）

### 归档分类

#### 🔵 **archive/original_tensorflow/** - 原始TensorFlow版本
```
archive/original_tensorflow/
├── step1.py                    # 原始TensorFlow Step1训练脚本
├── step2.py                    # 原始TensorFlow Step2训练脚本
├── bert/                       # BERT相关工具（TensorFlow版）
│   ├── modeling.py
│   ├── optimization.py
│   └── tokenization.py
├── utils/                      # TensorFlow工具函数
│   ├── pre_data_bert.py
│   └── tf_funcs.py
├── CodaLab/                    # 原始评估框架
│   ├── README.md
│   └── evaluation/
└── README_ORIGINAL.md          # 原始项目说明
```

#### 🟢 **archive/pytorch_migration/** - PyTorch迁移版本
```
archive/pytorch_migration/
├── config.py                   # PyTorch配置系统
├── step1_pytorch.py           # PyTorch Step1训练脚本
├── step2_pytorch.py           # PyTorch Step2训练脚本
├── pytorch_utils/             # PyTorch工具模块
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── models.py
├── test_*.py                  # 所有测试脚本
├── debug_*.py                 # 所有调试脚本
├── log/                       # 训练日志和模型
├── PYTORCH_MIGRATION_SUMMARY.md
├── MODERNIZATION_ROADMAP.md
├── evaluation_comparison.md
└── README_PYTORCH.md          # PyTorch版本说明
```

### 📋 归档后的根目录结构
```
MECPE-Enhanced/
├── data/                      # 保留：原始数据
├── archive/
│   ├── original_tensorflow/   # 原始TensorFlow版本
│   └── pytorch_migration/     # PyTorch迁移版本
├── DATA_ACQUISITION_STRATEGY.md
├── README.md                  # 更新后的主README
└── [现代化项目文件...]       # 未来的新架构
```

## 🎯 归档目标

1. **清理工作空间** - 为现代化重构腾出空间
2. **保存历史** - 完整保留所有开发历程
3. **分类明确** - 原始版本vs迁移版本清晰分离
4. **便于参考** - 归档但可随时查阅
5. **为未来准备** - 干净的根目录用于新架构