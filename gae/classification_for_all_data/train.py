# train.py (兼容GraphMAE版本)
from sklearn.model_selection import train_test_split
from data_process import load_and_normalize_datasets
from graph_mae import GraphMAE, train_graph_mae, generate_graph_embeddings  # 修改导入
from classification import Classifier
import torch
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

# 原有训练函数几乎不变，只修改了导入和模型初始化
def train_and_evaluate(paths):
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

    # 3. 训练模型（唯一需要修改的是hidden_channels可能需要调整）
    logging.info("Training Graph Autoencoder with all data...")
    input_dim = all_graphs[0].num_node_features
    model = GraphMAE(num_node_features=input_dim, hidden_channels=64)  
    model = train_graph_mae(model, all_graphs, paths, epochs=50, lr=1e-3)
    
    # 4. 对每个数据集单独处理（以下完全不变）
    results = {}
    for dataset_name, dataset_info in grouped_data.items():
        logging.info(f"\nProcessing dataset embeddings: {dataset_name}")
        
        logging.info("Generating embeddings...")
        embeddings = generate_graph_embeddings(model, dataset_info['graphs'])
        labels = dataset_info['labels']
        label_encoder = dataset_info['label_encoder']
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.8, random_state=42, stratify=labels
        )
        print(np.unique(y_train))
        print_label_distribution(f"{dataset_name} (train)", y_train, label_encoder)
        print_label_distribution(f"{dataset_name} (test)", y_test, label_encoder)
        
        print("X_train type:", type(X_train))
        print("y_train type:", type(y_train))
        print("embeddings type:", type(embeddings))
        
        logging.info("Training classifier...")
        clf = Classifier()
        clf.train(X_train, y_train)
        metrics, y_pred = clf.evaluate(X_test, y_test,save_path=paths['result_dir']/f"{dataset_name}_confusion_matrix.png")
        
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
    
    return results