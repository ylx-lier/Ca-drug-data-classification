#!/usr/bin/env python3
"""
GPU内存冲突诊断和修复工具
"""

import torch
import gc
import psutil
import subprocess
import os
import sys

def check_gpu_status():
    """检查GPU状态"""
    print("=== GPU状态检查 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_cached = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = props.total_memory / 1024**3
        
        print(f"GPU {i}: {props.name}")
        print(f"  总内存: {memory_total:.2f} GB")
        print(f"  已分配: {memory_allocated:.2f} GB")
        print(f"  已缓存: {memory_cached:.2f} GB")
        print(f"  空闲: {memory_total - memory_cached:.2f} GB")
    
    return True

def check_xgboost_config():
    """检查XGBoost配置"""
    print("\n=== XGBoost配置检查 ===")
    
    try:
        import xgboost as xgb
        print(f"✅ XGBoost版本: {xgb.__version__}")
        
        # 检查GPU支持
        gpu_support = hasattr(xgb, 'gpu_hist')
        print(f"GPU支持: {'✅' if gpu_support else '❌'}")
        
        # 创建测试模型检查默认配置
        import numpy as np
        X = np.random.random((10, 5))
        y = np.random.randint(0, 2, 10)
        
        model = xgb.XGBClassifier(tree_method='auto')
        try:
            model.fit(X, y, verbose=False)
            print("✅ XGBoost auto模式工作正常")
        except Exception as e:
            print(f"❌ XGBoost auto模式失败: {e}")
            
    except ImportError:
        print("❌ XGBoost未安装")

def check_processes():
    """检查占用GPU的进程"""
    print("\n=== GPU进程检查 ===")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi输出:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi命令失败")
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到")

def clear_gpu_memory():
    """清理GPU内存"""
    print("\n=== 清理GPU内存 ===")
    
    if torch.cuda.is_available():
        before_memory = torch.cuda.memory_allocated() / 1024**3
        
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        after_memory = torch.cuda.memory_allocated() / 1024**3
        freed = before_memory - after_memory
        
        print(f"清理前: {before_memory:.2f} GB")
        print(f"清理后: {after_memory:.2f} GB")
        print(f"释放: {freed:.2f} GB")
    else:
        print("❌ GPU不可用，无法清理")

def suggest_fixes():
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    
    print("1. 强制XGBoost使用CPU:")
    print("   clf = Classifier(use_gpu=False)")
    print("")
    
    print("2. 在分类前清理GPU内存:")
    print("   torch.cuda.empty_cache()")
    print("   gc.collect()")
    print("")
    
    print("3. 设置环境变量:")
    print("   export CUDA_VISIBLE_DEVICES=0")
    print("")
    
    print("4. 分离训练和分类:")
    print("   先完成GAE训练，再启动分类")
    print("")
    
    print("5. 使用CPU版本XGBoost:")
    print("   pip install xgboost==1.7.4  # CPU版本")

def test_fix():
    """测试修复效果"""
    print("\n=== 测试修复效果 ===")
    
    try:
        import numpy as np
        import xgboost as xgb
        
        # 模拟数据
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        # 测试CPU模式
        model_cpu = xgb.XGBClassifier(
            tree_method='hist',
            device='cpu',
            n_estimators=10,
            verbosity=0
        )
        
        model_cpu.fit(X, y)
        pred = model_cpu.predict(X)
        print("✅ CPU模式XGBoost工作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("🔧 GPU内存冲突诊断工具")
    print("=" * 50)
    
    # 检查各项状态
    gpu_ok = check_gpu_status()
    check_xgboost_config()
    check_processes()
    
    # 清理内存
    clear_gpu_memory()
    
    # 建议修复
    suggest_fixes()
    
    # 测试修复
    if test_fix():
        print("\n🎉 修复测试通过！")
    else:
        print("\n⚠️ 需要进一步调试")

if __name__ == "__main__":
    main()
