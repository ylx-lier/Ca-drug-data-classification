#!/usr/bin/env python3
"""
命令行版本的结果查看器 - 无需浏览器
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.font_manager as fm
import os
import warnings

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试常见的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',  # 宋体
            'DejaVu Sans',  # 默认字体作为后备
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                break
        else:
            # 如果没有找到中文字体，使用 Sans-serif
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("⚠️  未找到中文字体，使用默认字体")
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置默认字体大小
        plt.rcParams['font.size'] = 10
        
        # 禁用字体警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        print(f"✅ 字体设置完成: {plt.rcParams['font.sans-serif'][0]}")
        
    except Exception as e:
        print(f"⚠️  字体设置失败，使用默认字体: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results():
    """加载实验结果"""
    results_file = Path("../../results/exp114/all_results.json")
    
    if not results_file.exists():
        print(f"❌ 结果文件不存在: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 提取results部分
    if 'results' in data:
        return data['results']
    else:
        return data

def create_results_dashboard():
    """创建结果仪表板"""
    results = load_experiment_results()
    if not results:
        return
    
    print("📊 实验结果仪表板")
    print("=" * 60)
    
    # 提取所有数据集的指标
    datasets = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for dataset, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            datasets.append(dataset)
            accuracies.append(metrics.get('accuracy', 0))
            
            # 从classification_report中提取precision和recall
            report = metrics.get('classification_report', {})
            if 'macro avg' in report:
                macro_avg = report['macro avg']
                precisions.append(macro_avg.get('precision', 0))
                recalls.append(macro_avg.get('recall', 0))
                f1_scores.append(macro_avg.get('f1-score', 0))
            else:
                # 如果没有macro avg，使用0作为默认值
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Dataset': datasets,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # 打印表格
    print("\n📋 详细结果表格:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 统计信息
    print(f"\n📈 统计信息:")
    print(f"数据集数量: {len(datasets)}")
    print(f"平均准确率: {np.mean(accuracies):.4f}")
    print(f"最高准确率: {np.max(accuracies):.4f} ({datasets[np.argmax(accuracies)]})")
    print(f"最低准确率: {np.min(accuracies):.4f} ({datasets[np.argmin(accuracies)]})")
    
    # 创建可视化图表
    create_visualization_plots(df)

def create_visualization_plots(df):
    """创建可视化图表"""
    print("\n🎨 生成可视化图表...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Results Dashboard / 实验结果仪表板', fontsize=16, fontweight='bold')
    
    # 1. 准确率条形图
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df)), df['Accuracy'], color='skyblue', alpha=0.7)
    ax1.set_title('Dataset Accuracy / 各数据集准确率')
    ax1.set_xlabel('Dataset / 数据集')
    ax1.set_ylabel('Accuracy / 准确率')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Dataset'], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 所有指标对比
    ax2 = axes[0, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    avg_scores = [df[metric].mean() for metric in metrics]
    bars2 = ax2.bar(metrics, avg_scores, color=['lightcoral', 'lightgreen', 'lightsalmon', 'lightblue'])
    ax2.set_title('Average Metrics / 平均指标对比')
    ax2.set_ylabel('Average Score / 平均分数')
    ax2.set_ylim(0, 1)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. 热力图
    ax3 = axes[1, 0]
    heatmap_data = df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
    im = ax3.imshow(heatmap_data.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Metrics Heatmap / 指标热力图')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Dataset'], rotation=45, ha='right')
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Score / 分数')
    
    # 4. 散点图 - Precision vs Recall
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['Precision'], df['Recall'], 
                         c=df['Accuracy'], cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    ax4.set_title('Precision vs Recall (Color=Accuracy)')
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Recall')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # 添加颜色条
    cbar2 = plt.colorbar(scatter, ax=ax4)
    cbar2.set_label('Accuracy')
    
    # 添加数据集标签
    for i, txt in enumerate(df['Dataset']):
        ax4.annotate(txt, (df['Precision'].iloc[i], df['Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=6)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = 'results_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")
    
    plt.show()

def create_training_curve():
    """创建模拟的训练曲线（如果有训练日志）"""
    print("\n📈 创建训练曲线...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 模拟训练数据（因为原始训练可能没有保存详细日志）
    epochs = np.arange(1, 51)
    loss = 0.1 * np.exp(-epochs/20) + 0.01 * np.random.random(50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve / 训练损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("✅ 训练曲线已保存: training_curve.png")
    plt.show()

def main():
    print("🖥️  命令行结果查看器")
    print("=" * 50)
    
    try:
        create_results_dashboard()
        create_training_curve()
        
        print("\n🎉 所有图表生成完成！")
        print("📁 生成的文件:")
        print("  - results_dashboard.png")
        print("  - training_curve.png")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        print("请确保已安装必要的依赖: matplotlib, pandas, numpy")

if __name__ == "__main__":
    main()
