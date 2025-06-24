# train.py (支持可选模型版本)
import os
import torch
import gc

# GPU内存管理 - 防止XGBoost冲突
if torch.cuda.is_available():
    # 清理GPU缓存
    torch.cuda.empty_cache()
    # 设置环境变量防止XGBoost自动使用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 限制可见GPU
    
from sklearn.model_selection import train_test_split
from data_process import load_and_normalize_datasets
from simple_gae import create_model, train_model, generate_embeddings  # 修改导入
from classification import Classifier
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import datetime
import pytz
from pathlib import Path
import logging
from collections import Counter
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard导入
# 原有可视化函数保持不变
def visualize_embeddings(embeddings, labels, label_encoder, save_path=None):
    tsne = TSNE(n_components=2)
    vis_data = tsne.fit_transform(embeddings)
    
    original_labels = label_encoder.inverse_transform(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vis_data[:,0], vis_data[:,1], c=labels, alpha=0.6)
    
    handles, _ = scatter.legend_elements()
    plt.legend(handles, label_encoder.classes_, title="Classes")
    
    plt.title("t-SNE Visualization of Graph Embeddings")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def print_label_distribution(name, y, label_encoder):
    """打印标签分布"""
    labels_str = label_encoder.inverse_transform(y)
    counter = Counter(labels_str)
    logging.info(f"\n{name} 标签分布:")
    for label, count in counter.items():
        logging.info(f"  类别 '{label}': {count} 个样本")

def create_accuracy_comparison_plot(dataset_accuracies, save_path):
    """创建各组accuracy对比图"""
    plt.figure(figsize=(12, 8))
    
    # 按accuracy排序
    sorted_data = sorted(dataset_accuracies.items(), key=lambda x: x[1], reverse=True)
    datasets, accuracies = zip(*sorted_data)
    
    # 创建条形图
    bars = plt.bar(range(len(datasets)), accuracies, alpha=0.7, color='steelblue')
    
    # 添加数值标签
    for i, (dataset, acc) in enumerate(sorted_data):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 添加平均线
    avg_acc = np.mean(accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'平均准确率: {avg_acc:.3f}')
    
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('各数据集分类准确率对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha='right')
    plt.ylim(0, max(accuracies) * 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Accuracy对比图已保存到: {save_path}")

# 原有训练函数，现在支持模型选择
def train_and_evaluate(paths, model_type="simple"):
    """
    训练和评估函数
    
    参数:
        paths: 路径字典
        model_type: "simple" 或 "graphmae"，默认使用simple
    """
    # 1. 加载并归一化所有数据集
    logging.info("Loading and normalizing datasets...")
    all_csv_paths = list(Path("../../data/calcium_data_all/").rglob("*.csv"))
    grouped_data = load_and_normalize_datasets(all_csv_paths)
    
    # 2. 准备联合训练数据
    logging.info("Preparing data for joint training...")
    all_graphs = []
    for dataset in grouped_data.values():
        all_graphs.extend(dataset['graphs'])
    logging.info(f"Total graphs for joint training: {len(all_graphs)}")

    # 3. 训练模型 - 支持模型选择
    logging.info(f"训练{model_type.upper()}图自编码器...")
    input_dim = all_graphs[0].num_node_features
    model = create_model(
        model_type=model_type, 
        num_node_features=input_dim, 
        hidden_channels=64,
        embedding_dim=32  # 仅对simple模型有效
    )
    model = train_model(model, all_graphs, paths, model_type=model_type, epochs=50, lr=1e-3)
    
    # 创建TensorBoard writer用于记录各组accuracy
    writer = SummaryWriter(paths["tensorboard_path"] / "classification_results")
    
    # 4. 对每个数据集单独处理并记录结果
    results = {}
    dataset_accuracies = {}  # 存储各组accuracy用于TensorBoard
    dataset_metrics = {}     # 存储详细指标
    
    for dataset_name, dataset_info in grouped_data.items():
        logging.info(f"\nProcessing dataset embeddings: {dataset_name}")
        
        logging.info("生成嵌入...")
        embeddings = generate_embeddings(model, dataset_info['graphs'], model_type=model_type)
        labels = dataset_info['labels']
        label_encoder = dataset_info['label_encoder']
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(np.unique(y_train))
        print_label_distribution(f"{dataset_name} (train)", y_train, label_encoder)
        print_label_distribution(f"{dataset_name} (test)", y_test, label_encoder)
        
        print("X_train type:", type(X_train))
        print("y_train type:", type(y_train))
        print("embeddings type:", type(embeddings))
        
        logging.info("Training classifier...")
        
        # GPU内存清理，防止与XGBoost冲突
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("GPU内存已清理")
        
        # 强制使用CPU避免与PyTorch GPU冲突
        clf = Classifier(use_gpu=False)
        clf.train(X_train, y_train)
        metrics, y_pred = clf.evaluate(X_test, y_test,save_path=paths['result_dir']/f"{dataset_name}_confusion_matrix.png")
        
        # 记录到TensorBoard
        accuracy = metrics["accuracy"]
        precision = metrics["precision"] 
        recall = metrics["recall"]
        f1_score = metrics["f1"]
        
        # 按数据集记录指标
        writer.add_scalar(f'Accuracy/{dataset_name}', accuracy, 0)
        writer.add_scalar(f'Precision/{dataset_name}', precision, 0)
        writer.add_scalar(f'Recall/{dataset_name}', recall, 0)
        writer.add_scalar(f'F1-Score/{dataset_name}', f1_score, 0)
        
        # 存储结果
        dataset_accuracies[dataset_name] = accuracy
        dataset_metrics[dataset_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        logging.info(f"{dataset_name} 分类结果: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
        
        results[dataset_name] = {
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "label_mapping": dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        }
        
        visualize_embeddings(
            embeddings, labels, label_encoder,
            save_path=paths['result_dir']/f"{dataset_name}_embeddings.png"
        )
    
    # 创建汇总的accuracy对比图
    create_accuracy_comparison_plot(dataset_accuracies, paths['result_dir'] / "accuracy_comparison.png")
    
    # 记录整体统计到TensorBoard
    avg_accuracy = np.mean(list(dataset_accuracies.values()))
    writer.add_scalar('Overall/Average_Accuracy', avg_accuracy, 0)
    writer.add_scalar('Overall/Max_Accuracy', max(dataset_accuracies.values()), 0)
    writer.add_scalar('Overall/Min_Accuracy', min(dataset_accuracies.values()), 0)
    
    # 创建accuracy分布的直方图
    writer.add_histogram('Distribution/Accuracy', np.array(list(dataset_accuracies.values())), 0)
    
    # 记录每个数据集的样本数量
    for dataset_name, dataset_info in grouped_data.items():
        sample_count = len(dataset_info['graphs'])
        writer.add_scalar(f'Sample_Count/{dataset_name}', sample_count, 0)
    
    writer.close()
    
    # 打印汇总信息
    logging.info("\n" + "="*50)
    logging.info("实验结果汇总:")
    logging.info("="*50)
    for dataset_name, acc in sorted(dataset_accuracies.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{dataset_name:20s}: {acc:.4f}")
    logging.info("-"*50)
    logging.info(f"{'平均准确率':20s}: {avg_accuracy:.4f}")
    logging.info(f"{'最高准确率':20s}: {max(dataset_accuracies.values()):.4f}")
    logging.info(f"{'最低准确率':20s}: {min(dataset_accuracies.values()):.4f}")
    logging.info("="*50)
    
    return results