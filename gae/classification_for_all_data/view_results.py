#!/usr/bin/env python3
# view_results.py - 查看实验结果的便捷脚本

import os
import sys
import subprocess
from pathlib import Path
import json
import webbrowser
import time

def find_latest_experiment():
    """找到最新的实验目录"""
    results_dir = Path("../../results")
    if not results_dir.exists():
        print("❌ 结果目录不存在，请先运行实验")
        return None
    
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]
    if not exp_dirs:
        print("❌ 没有找到实验结果")
        return None
    
    # 按照数字排序找到最新的
    latest_exp = max(exp_dirs, key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 0)
    return latest_exp

def show_results_summary(exp_dir):
    """显示实验结果摘要"""
    results_file = exp_dir / "all_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 实验结果摘要 ({exp_dir.name})")
        print("=" * 50)
        
        if 'model_type' in data:
            print(f"模型类型: {data['model_type'].upper()}")
        
        if 'experiment_config' in data and 'timestamp' in data['experiment_config']:
            print(f"实验时间: {data['experiment_config']['timestamp']}")
        
        print("\n各组准确率:")
        results = data.get('results', data)  # 兼容旧格式
        accuracies = []
        for dataset, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                acc = metrics['accuracy']
                accuracies.append(acc)
                print(f"  {dataset:20s}: {acc:.4f}")
        
        if accuracies:
            print("-" * 50)
            print(f"  {'平均准确率':20s}: {sum(accuracies)/len(accuracies):.4f}")
            print(f"  {'最高准确率':20s}: {max(accuracies):.4f}")
            print(f"  {'最低准确率':20s}: {min(accuracies):.4f}")
        
        print("=" * 50)

def start_tensorboard(exp_dir):
    """启动TensorBoard"""
    tensorboard_dir = exp_dir / "tensorboard"
    
    if not tensorboard_dir.exists():
        print("❌ TensorBoard目录不存在")
        return False
    
    print(f"\n🚀 启动TensorBoard...")
    print(f"📁 实验目录: {exp_dir.name}")
    print(f"📊 TensorBoard目录: {tensorboard_dir}")
    print(f"\n请在浏览器中打开: http://localhost:6006")
    print("按 Ctrl+C 停止TensorBoard\n")
    
    try:
        # 启动TensorBoard
        subprocess.run([
            "tensorboard", 
            f"--logdir={tensorboard_dir}", 
            "--port=6006", 
            "--host=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 TensorBoard已停止")
    except FileNotFoundError:
        print("❌ 未找到tensorboard命令，请确保已安装TensorBoard")
        print("安装命令: pip install tensorboard")
        return False
    
    return True

def main():
    """主函数"""
    print("🔍 查找最新实验结果...")
    
    exp_dir = find_latest_experiment()
    if exp_dir is None:
        sys.exit(1)
    
    print(f"✅ 找到实验: {exp_dir.name}")
    
    # 显示结果摘要
    show_results_summary(exp_dir)
    
    # 询问是否启动TensorBoard
    try:
        choice = input("\n是否启动TensorBoard查看详细结果? (y/n): ").lower()
        if choice in ['y', 'yes', '']:
            start_tensorboard(exp_dir)
        else:
            print("📂 实验结果位置:")
            print(f"  - 主目录: {exp_dir}")
            print(f"  - 图片: {exp_dir / 'figures'}")
            print(f"  - 日志: {exp_dir / 'experiment.log'}")
            print(f"  - TensorBoard: {exp_dir / 'tensorboard'}")
    except KeyboardInterrupt:
        print("\n\n👋 退出")

if __name__ == "__main__":
    main()
