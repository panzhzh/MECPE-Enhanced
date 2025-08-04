# encoding: utf-8

import sys, os
sys.path.append('./src')

# 测试step2的基本功能
def test_step2_imports():
    """测试step2模块导入"""
    print("Testing step2 imports...")
    try:
        from src.step2 import Config, MECPE_Step2_Model, build_model, MECPEStep2Dataset
        from src.utils.pre_data_bert import load_data_utt_step2, prf_2nd_step, prf_2nd_step_emocate
        from src.utils.tf_funcs import layer_normalize
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """测试配置类"""
    print("Testing Config class...")
    try:
        from src.step2 import Config
        config = Config()
        assert hasattr(config, 'embedding_dim')
        assert hasattr(config, 'n_hidden')
        assert hasattr(config, 'max_sen_len')
        print(f"✓ Config class works, embedding_dim={config.embedding_dim}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_layer_normalize():
    """测试layer_normalize函数"""
    print("Testing layer_normalize function...")
    try:
        import torch
        from src.utils.tf_funcs import layer_normalize
        
        # 创建测试张量
        x = torch.randn(2, 3, 4)
        result = layer_normalize(x)
        
        assert result.shape == x.shape
        print(f"✓ layer_normalize works, input shape: {x.shape}, output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ layer_normalize test failed: {e}")
        return False

def test_evaluation_functions():
    """测试评估函数"""
    print("Testing evaluation functions...")
    try:
        from src.utils.pre_data_bert import prf_2nd_step, prf_2nd_step_emocate
        import numpy as np
        
        # 创建测试数据
        pair_id_all = [[1, 1, 2, 1], [1, 2, 3, 2]]
        pair_id = [[1, 1, 2, 1], [1, 2, 3, 2], [1, 3, 4, 1]]
        pred_y = [1, 0, 1]  # 预测结果
        
        # 测试prf_2nd_step
        result = prf_2nd_step(pair_id_all, pair_id, pred_y)
        assert len(result) == 7  # [p, r, f1] + [o_p, o_r, o_f1] + [keep_rate]
        print(f"✓ prf_2nd_step works, result length: {len(result)}")
        
        # 测试prf_2nd_step_emocate
        result_emocate = prf_2nd_step_emocate(pair_id_all, pair_id, pred_y)
        assert len(result_emocate) > 10  # 复杂的情绪类别评估结果
        print(f"✓ prf_2nd_step_emocate works, result length: {len(result_emocate)}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation functions test failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*50)
    print("Step2 Basic Functionality Tests")
    print("="*50)
    
    tests = [
        test_step2_imports,
        test_config,
        test_layer_normalize,
        test_evaluation_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)