# MECPE-Enhanced: Multimodal Emotion-Cause Pair Extraction

[![Dataset](https://img.shields.io/badge/Dataset-ECF_2.0-F0A336)](https://huggingface.co/datasets/NUSTM/ECF) [![Task](https://img.shields.io/badge/Task-SemEval_2024-488DF8)](https://nustm.github.io/SemEval-2024_ECAC/) [![Paper](https://img.shields.io/badge/Paper-TAFFC_2023-2E6396)](https://ieeexplore.ieee.org/document/9969873)

An enhanced implementation for **Multimodal Emotion-Cause Pair Extraction in Conversations** based on the SemEval-2024 Task 3 dataset.

This project focuses on **Subtask 2: Multimodal Emotion-Cause Pair Extraction**, which analyzes emotions and their causes across text, audio, and visual modalities in conversational contexts.

## 🎯 Task Overview

Given a multimodal conversation from the TV show *Friends*, the goal is to:
1. **Identify emotion utterances** and their emotion categories
2. **Extract cause utterances** that trigger these emotions  
3. **Pair emotions with their corresponding causes** across all modalities

**Example**: If character Phoebe shows *disgust* in utterance U5 because Monica and Chandler are kissing (visible in the video), the model should extract the pair `(U5_Disgust, U5)`. 

## 📊 Dataset

**ECF 2.0 (Emotion-Cause-in-Friends)**: A multimodal conversational dataset from the TV show *Friends*

| Split | Conversations | Utterances | Emotion-Cause Pairs |
|-------|---------------|------------|---------------------|
| Train | 1,374 | 13,619 | ~9,800 |  
| Test | 341 | 3,101 | ~2,500 |

**Modalities**:
- **Text**: Conversational utterances with speaker information
- **Audio**: 6,373-dimensional acoustic features (openSMILE)
- **Visual**: 4,096-dimensional visual features (3D-CNN)

**Emotions**: 6 categories - *anger, disgust, fear, joy, sadness, surprise*

❗️ **Data is for research purposes only**

## 🚀 Quick Start

### Prerequisites
```bash
# Python environment
Python 3.6+ 
TensorFlow 1.15.4
```

### Installation  
```bash
git clone https://github.com/yourusername/MECPE-Enhanced
cd MECPE-Enhanced
pip install -r requirements.txt
```

### Download Pre-extracted Features
The multimodal features are available at:
- [Audio Features (662MB)](https://drive.google.com/file/d/1EhU2jFSr_Vi67Wdu1ARJozrTJtgiQrQI/view) → `data/features/audio_embedding_6373.npy`
- [Video Features (426MB)](https://drive.google.com/file/d/1NGSsiQYDTqgen_g9qndSuha29JA60x14/view) → `data/features/video_embedding_4096.npy`

### Usage

**Step 1: Emotion & Cause Recognition**
```bash
# BiLSTM + Multimodal
python step1.py --use_x_a yes --use_x_v yes --scope BiLSTM_A_V

# BERT + Multimodal  
python step1.py --model_type BERTcased --use_x_a yes --use_x_v yes --scope BERT_A_V
```

**Step 2: Emotion-Cause Pairing**
```bash
python step2.py --use_x_a yes --use_x_v yes --scope BiLSTM_A_V
```

## 📁 Project Structure

```
MECPE-Enhanced/
├── step1.py              # Stage 1: Emotion & Cause Recognition
├── step2.py              # Stage 2: Emotion-Cause Pairing  
├── bert/                 # BERT model components
├── utils/                # Utility functions
├── data/
│   ├── Subtask_2_train.json    # Training data (SemEval-2024)
│   ├── Subtask_2_test.json     # Test data (SemEval-2024)
│   ├── features/
│   │   ├── audio_embedding_6373.npy  # Audio features
│   │   └── video_embedding_4096.npy  # Video features
│   └── [ECF 1.0 data files...]
└── CodaLab/evaluation/   # Official evaluation scripts
```

## 🏆 Performance

Evaluation metrics: **Weighted Average F1** across 6 emotion categories

| Model | Text | +Audio | +Video | +Both |
|-------|------|--------|--------|-------|
| BiLSTM | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| BERT | 0.xxx | 0.xxx | 0.xxx | 0.xxx |

## 📚 Citation

```bibtex
@article{wang2023multimodal,
  title={Multimodal Emotion-Cause Pair Extraction in Conversations},
  author={Wang, Fanfan and Ding, Zixiang and Xia, Rui and Li, Zhaoyu and Yu, Jianfei},
  journal={IEEE Transactions on Affective Computing},
  volume={14}, number={3}, pages={1832--1844}, year={2023}
}

@inproceedings{wang2024semeval,
  title={SemEval-2024 Task 3: Multimodal Emotion Cause Analysis in Conversations},
  author={Wang, Fanfan and Ma, Heqing and Xia, Rui and Yu, Jianfei and Cambria, Erik},
  booktitle={Proceedings of SemEval-2024}, 
  pages={2022--2033}, year={2024}
}
```

## 📄 License

This project is licensed under GPL-3.0 - see the [LICENSE](LICENSE.txt) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
# MECPE-Enhanced
