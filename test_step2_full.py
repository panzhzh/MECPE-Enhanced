# encoding: utf-8

import sys, os
import torch
import numpy as np
sys.path.append('./src')

def create_mock_data():
    """创建模拟数据文件和嵌入用于测试"""
    # 创建测试目录
    os.makedirs('./test_data', exist_ok=True)
    
    # 创建模拟的step1输出数据文件 (step2的输入)
    mock_step2_data = """1 3
[[1, 2]]
1 | 1 | 0 | dia1utt1 | joy | hello world
2 | 0 | 1 | dia1utt2 | neutral | this is sad
3 | 0 | 0 | dia1utt3 | neutral | okay then
2 2
[[1, 2]]  
1 | 1 | 0 | dia2utt1 | anger | i am angry
2 | 0 | 1 | dia2utt2 | neutral | why so
"""
    
    with open('./test_data/test_step2.txt', 'w', encoding='utf-8') as f:
        f.write(mock_step2_data)
    
    # 创建模拟词汇数据
    vocab_data = """hello 0.1 0.2 0.3
world 0.4 0.5 0.6  
this 0.7 0.8 0.9
is 0.1 0.4 0.7
sad 0.2 0.5 0.8
okay 0.3 0.6 0.9
then 0.4 0.7 0.1
i 0.5 0.8 0.2
am 0.6 0.9 0.3
angry 0.7 0.1 0.4
why 0.8 0.2 0.5
so 0.9 0.3 0.6
joy 0.1 0.3 0.5
neutral 0.2 0.4 0.6
anger 0.3 0.5 0.7"""
    
    with open('./test_data/test_w2v.txt', 'w', encoding='utf-8') as f:
        f.write("15 3\n")  # vocab_size, embedding_dim
        f.write(vocab_data)
    
    # 创建all_data_pair.txt用于加载词汇
    all_data = """1 3
[[1, 2]]
1 | dia1utt1 | joy | hello world
2 | dia1utt2 | neutral | this is sad
3 | dia1utt3 | neutral | okay then
2 2
[[1, 2]]
1 | dia2utt1 | anger | i am angry
2 | dia2utt2 | neutral | why so
"""
    
    with open('./test_data/test_all_data.txt', 'w', encoding='utf-8') as f:
        f.write(all_data)
    
    # 创建模拟的多模态特征
    video_id_mapping = {'dia1utt1': 1, 'dia1utt2': 2, 'dia1utt3': 3, 'dia2utt1': 4, 'dia2utt2': 5}
    np.save('./test_data/video_id_mapping.npy', video_id_mapping)
    
    # 创建模拟的视频和音频特征
    video_embedding = np.random.randn(10, 100)  # 10个视频特征，每个100维
    audio_embedding = np.random.randn(10, 80)   # 10个音频特征，每个80维
    np.save('./test_data/video_embedding.npy', video_embedding)
    np.save('./test_data/audio_embedding.npy', audio_embedding)
    
    print("✓ Mock data created successfully")

def test_step2_data_loading():
    """测试step2数据加载"""
    print("Testing step2 data loading...")
    try:
        from src.utils.pre_data_bert import load_w2v, load_embedding_from_npy, load_data_utt_step2
        
        # 加载词向量和多模态特征
        word_idx_rev, word_idx, _, _, word_embedding, pos_embedding = load_w2v(
            3, 5, './test_data/test_all_data.txt', './test_data/test_w2v.txt')
        
        video_idx, video_embedding, audio_embedding = load_embedding_from_npy(
            './test_data/video_id_mapping.npy', './test_data/video_embedding.npy', './test_data/audio_embedding.npy')
        
        # 加载step2数据
        x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs = load_data_utt_step2(
            './test_data/test_step2.txt', word_idx, video_idx, max_sen_len=10, choose_emocate='', pred_future_cause=1)
        
        print(f"✓ Data loaded: x.shape={x.shape}, y.shape={y.shape}")
        print(f"  Pairs: {len(pair_id_all)} true pairs, {len(pair_id)} candidate pairs")
        return True, (word_embedding, pos_embedding, video_embedding, audio_embedding, 
                     x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id)
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_step2_model():
    """测试step2模型"""
    print("Testing step2 model...")
    try:
        from src.step2 import Config, MECPE_Step2_Model, build_model
        
        # 创建模拟嵌入
        word_embedding = np.random.randn(20, 3)
        pos_embedding = np.random.randn(200, 5)
        video_embedding = np.random.randn(10, 100)
        audio_embedding = np.random.randn(10, 80)
        
        embeddings = [word_embedding, pos_embedding, video_embedding, audio_embedding]
        
        # 构建模型
        model = build_model(embeddings, device='cpu')
        
        # 创建测试输入
        batch_size = 2
        x = torch.randint(0, 20, (batch_size, 2, 10))  # [batch_size, 2, max_sen_len]
        sen_len = torch.randint(1, 10, (batch_size, 2))  # [batch_size, 2]
        distance = torch.randint(50, 150, (batch_size,))  # [batch_size]
        x_emocate = torch.randint(0, 7, (batch_size,))  # [batch_size]
        x_v = torch.randint(0, 10, (batch_size, 2))  # [batch_size, 2]
        
        # 前向传播
        with torch.no_grad():
            pred_pair = model(x, sen_len, distance, x_emocate, x_v, is_training=False)
        
        print(f"✓ Model forward pass successful: output shape={pred_pair.shape}")
        assert pred_pair.shape == (batch_size, 2)
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step2_training_loop():
    """测试step2训练循环的一个iteration"""
    print("Testing step2 training loop...")
    try:
        from src.step2 import Config, MECPE_Step2_Model, MECPEStep2Dataset, get_batch_data
        import torch.optim as optim
        import torch.nn as nn
        
        # 使用之前创建的数据
        success, data = test_step2_data_loading()
        if not success:
            return False
            
        word_embedding, pos_embedding, video_embedding, audio_embedding, x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id = data
        
        # 创建数据集 (模拟)
        class MockDataset:
            def __init__(self):
                self.x = torch.LongTensor(x)
                self.sen_len = torch.LongTensor(sen_len)
                self.distance = torch.LongTensor(distance)
                self.x_emocate = torch.LongTensor(x_emocate)
                self.x_v = torch.LongTensor(x_v)
                self.y = torch.FloatTensor(y)
                self.all = [self.x, self.sen_len, self.distance, self.x_emocate, self.x_v, self.y]
        
        dataset = MockDataset()
        
        # 构建模型
        embeddings = [word_embedding, pos_embedding, video_embedding, audio_embedding]
        model = MECPE_Step2_Model(word_embedding, pos_embedding, video_embedding, audio_embedding, Config())
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 运行一个训练步
        model.train()
        batch_data = [t[:4] for t in dataset.all]  # 取前4个样本
        x, sen_len, distance, x_emocate, x_v, y = batch_data
        
        optimizer.zero_grad()
        pred_pair = model(x, sen_len, distance, x_emocate, x_v, is_training=True)
        
        loss = criterion(pred_pair, y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful: loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有step2测试"""
    print("="*60)
    print("Step2 Full Functionality Tests")
    print("="*60)
    
    # 创建模拟数据
    create_mock_data()
    print()
    
    tests = [
        test_step2_data_loading,
        test_step2_model,
        test_step2_training_loop,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("="*60)
    print(f"Test Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 All step2 tests passed! Step2 is ready to use!")
        return True
    else:
        print("❌ Some step2 tests failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)