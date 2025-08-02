# ECF数据集原始多模态数据获取策略

## 🎯 目标
获取ECF数据集的原始视频/音频数据，实现端到端的现代化多模态架构

## 📊 数据源分析

### 当前可用数据
1. **文本数据** ✅ 
   - `train.txt`, `dev.txt`, `test.txt` - 完整的对话和标注
   - 包含时间戳: `Friends_S1E1: 00:16:45.504 - 00:16:46.672`

2. **预处理特征** ✅ (但不适用于现代化)
   - `audio_embedding_6373.npy` - openSMILE提取的音频特征
   - `video_embedding_4096.npy` - 3D-CNN提取的视频特征

3. **映射关系** ✅
   - `all_data_pair_ECFvsMELD.txt` - ECF与MELD的对应关系

## 🚀 数据获取方案

### 📋 方案对比

| 方案 | 数据源 | 优势 | 劣势 | 可行性 |
|------|--------|------|------|--------|
| 方案1 | MELD原始视频 | 官方提供，质量保证 | ECF修改了时间戳，不完全匹配 | ⭐⭐⭐ |
| 方案2 | Friends原始剧集 | 完全匹配ECF时间戳 | 需要自行获取，版权问题 | ⭐⭐⭐⭐ |
| 方案3 | 混合方案 | 结合两者优势 | 实现复杂 | ⭐⭐⭐⭐⭐ |

### 🎯 推荐方案：混合策略

#### Step 1: 获取MELD原始数据
```bash
# 下载MELD原始视频数据
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 解压
tar -xzf MELD.Raw.tar.gz

# 数据结构
MELD.Raw/
├── train/
│   ├── dia0_utt0.mp4
│   ├── dia0_utt1.mp4
│   └── ...
├── dev/
└── test/
```

#### Step 2: 构建ECF-MELD映射
```python
def build_ecf_meld_mapping():
    """构建ECF到MELD视频文件的映射"""
    # 读取 all_data_pair_ECFvsMELD.txt
    # 格式: ECF_utterance | MELD_reference (如: train_dia559_utt3)
    
    mapping = {}
    with open('all_data_pair_ECFvsMELD.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(' | ')
            if len(parts) >= 5:  # 包含MELD引用
                meld_ref = parts[-1]  # 最后一列
                if 'dia' in meld_ref and 'utt' in meld_ref:
                    # 提取dia和utt编号
                    dia_id = extract_dia_id(meld_ref)
                    utt_id = extract_utt_id(meld_ref)
                    video_file = f"dia{dia_id}_utt{utt_id}.mp4"
                    mapping[ecf_id] = video_file
    
    return mapping
```

#### Step 3: Friends原始剧集数据处理
```python
def extract_from_friends_episodes():
    """基于ECF时间戳从Friends剧集提取片段"""
    
    # ECF时间戳格式: Friends_S1E1: 00:16:45.504 - 00:16:46.672
    def parse_timestamp(timestamp_str):
        # 解析季、集、时间范围
        pattern = r'Friends_S(\d+)E(\d+): ([\d:\.]+) - ([\d:\.]+)'
        match = re.match(pattern, timestamp_str)
        return season, episode, start_time, end_time
    
    def extract_clip(episode_file, start_time, end_time, output_file):
        """使用FFmpeg提取视频片段"""
        cmd = [
            'ffmpeg', '-i', episode_file,
            '-ss', start_time, '-to', end_time,
            '-c', 'copy', output_file
        ]
        subprocess.run(cmd)
```

## 🛠️ 实施计划

### Phase 1: MELD数据获取 (立即可行)
```bash
# 1. 下载MELD原始数据
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 2. 构建映射关系
python build_meld_mapping.py

# 3. 验证匹配率
python validate_mapping.py
```

### Phase 2: 数据清洗和对齐
```python
# 处理ECF修改过的数据
def align_ecf_meld_data():
    """处理ECF对MELD的修改"""
    
    # 1. 识别完全匹配的utterances
    # 2. 处理时间戳调整的utterances  
    # 3. 标记ECF新增的utterances
    # 4. 生成缺失数据报告
```

### Phase 3: Friends剧集数据补充
```python
def supplement_with_friends_episodes():
    """用Friends原始剧集补充缺失数据"""
    
    # 1. 基于ECF时间戳提取片段
    # 2. 音视频分离和预处理
    # 3. 质量验证和人工检查
```

## 📁 目标数据结构

```
ECF_Multimodal/
├── videos/
│   ├── train/
│   │   ├── dia0_utt0.mp4
│   │   └── ...
│   ├── dev/
│   └── test/
├── audios/
│   ├── train/
│   │   ├── dia0_utt0.wav
│   │   └── ...
│   ├── dev/
│   └── test/
├── metadata/
│   ├── train_metadata.json
│   ├── dev_metadata.json
│   └── test_metadata.json
└── mapping/
    ├── ecf_to_video.json
    └── coverage_report.json
```

## 🎯 期望输出

### 数据完整性目标
- **MELD匹配率**: >85% 的ECF utterances有对应MELD视频
- **时间戳准确性**: 手动验证关键样本的对齐质量
- **覆盖率**: 确保train/dev/test的平衡覆盖

### 质量指标
- **视频质量**: 1080p或以上分辨率
- **音频质量**: 44.1kHz采样率，立体声
- **时长准确性**: 与ECF标注时间戳误差<1秒

## 🔧 技术栈

### 数据处理工具
```python
# 核心依赖
pip install ffmpeg-python opencv-python librosa torchaudio transformers

# 数据处理流水线
class ECFDataProcessor:
    def __init__(self):
        self.meld_mapping = self.load_meld_mapping()
        self.ecf_data = self.load_ecf_annotations()
    
    def extract_multimodal_data(self):
        """提取多模态数据"""
        for sample in self.ecf_data:
            video_path = self.get_video_path(sample)
            audio_data = self.extract_audio(video_path)
            video_frames = self.extract_frames(video_path)
            yield {
                'text': sample['utterance'],
                'audio': audio_data,
                'video': video_frames,
                'metadata': sample['metadata']
            }
```

## ⚠️ 风险和缓解

### 潜在问题
1. **版权限制**: Friends剧集的版权问题
2. **数据缺失**: 部分ECF utterances无对应视频
3. **质量不一**: 不同来源数据质量差异

### 缓解策略
1. **学术使用声明**: 仅用于研究目的
2. **多源补充**: MELD + Friends双重覆盖
3. **质量检验**: 自动化 + 人工验证

---

## 🚀 下一步行动

### 立即执行 (今天)
1. **下载MELD数据**: 获取原始视频文件
2. **分析映射关系**: 统计ECF-MELD匹配率
3. **评估可行性**: 确定数据获取的真实覆盖率

### 本周目标
1. **建立数据管道**: 自动化视频处理流程
2. **质量验证**: 样本检查和对齐验证
3. **现代化架构设计**: 基于端到端数据的新架构

一旦数据获取完成，我们就能构建真正现代化的多模态Transformer架构！🎬🤖