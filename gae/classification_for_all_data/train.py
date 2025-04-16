
# train.py
from sklearn.model_selection import train_test_split
from data_process import load_and_normalize_datasets
from graph_autoencoder import GraphAutoEncoder, train_graph_autoencoder, generate_graph_embeddings
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

def visualize_embeddings(embeddings, labels, label_encoder, save_path=None):
    tsne = TSNE(n_components=2)
    vis_data = tsne.fit_transform(embeddings)
    
    # 将整数编码的标签转换回原始标签
    original_labels = label_encoder.inverse_transform(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vis_data[:,0], vis_data[:,1], c=labels, alpha=0.6)
    
    # 使用原始标签创建图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, label_encoder.classes_, title="Classes")
    
    plt.title("t-SNE Visualization of Graph Embeddings")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

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


    # 3. 训练GAE模型（在所有数据上）
    logging.info("Training Graph Autoencoder with all data...")
    input_dim = all_graphs[0].num_node_features
    model = GraphAutoEncoder(num_node_features=input_dim, hidden_channels=64)
    model = train_graph_autoencoder(model, all_graphs, paths, epochs=50, lr=1e-3)
    
    # 4. 对每个数据集单独处理
    results = {}
    for dataset_name, dataset_info in grouped_data.items():
        logging.info(f"\nProcessing dataset embeddings: {dataset_name}")
        
        # 生成该数据集的embeddings
        logging.info("Generating embeddings...")
        embeddings = generate_graph_embeddings(model, dataset_info['graphs'])
        labels = dataset_info['labels']
        label_encoder = dataset_info['label_encoder']
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 训练和评估分类器
        logging.info("Training classifier...")
        clf = Classifier()
        clf.train(X_train, y_train, use_balanced_weights=False)
        metrics, y_pred = clf.evaluate(X_test, y_test)
        
        # 保存结果
        results[dataset_name] = {
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "label_mapping": dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        }
        
        # 可视化
        visualize_embeddings(
            embeddings, labels, label_encoder,
            save_path=paths['result_dir']/f"{dataset_name}_embeddings.png"
        )
    
    return results