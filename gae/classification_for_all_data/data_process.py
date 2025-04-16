import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from resize_interpolate import linear_interpolate
from torch_geometric.data import Data
import torch
from itertools import combinations
import logging
from tqdm import tqdm

def random_downsample(features):
    """
    对特征矩阵进行随机降采样，保留一半的样本。

    参数:
        features: 原始特征矩阵 (num_nodes, num_features)
    
    返回:
        downsampled_features: 降采样后的特征矩阵
    """
    # 获取原始特征矩阵的行数
    num_nodes = features.shape[0]

    # 计算保留一半节点的数量
    num_samples = num_nodes // 2

    # 随机选择num_samples个节点的索引
    selected_indices = np.random.choice(num_nodes, num_samples, replace=False)
    downsampled_features = features[selected_indices]
    
    return downsampled_features

def load_and_normalize_datasets(data_paths, normalize=True):
    """
    加载并归一化多个数据集，保留数据集来源信息
    返回:
        grouped_data: {
            'dataset1': {
                'graphs': [Data(graph1), Data(graph2),...],
                'labels': [label1, label2,...],
                'label_encoder': LabelEncoder对象
            },
            'dataset2': {...},
            ...
        }
    """
    # 第一阶段：收集所有特征用于计算归一化参数
    all_features = []
    temp_data = {}

    for path in tqdm(data_paths, desc="收集特征归一化"):
        
        dataset_name = str(path.parent.name)
        filename = path.stem  # 获取不带扩展名的文件名
        
        # 从文件名提取标签
        label = extract_label(filename)  # 使用你提供的extract_label函数
        
        # 加载数据
        matrix = pd.read_csv(path, header=None).values
        matrix = linear_interpolate(matrix, target_dim=4570)  # 使用你提供的插值函数
        
        if dataset_name not in temp_data:
            temp_data[dataset_name] = {'features': [], 'raw_labels': []}
        
        temp_data[dataset_name]['features'].append(matrix)
        temp_data[dataset_name]['raw_labels'].append(label)
        all_features.append(matrix)
    
    # 计算归一化参数
    if normalize and len(all_features) > 0:
        scaler = MinMaxScaler()
        scaler.fit(np.vstack([f.reshape(-1, all_features[0].shape[1]) for f in tqdm(all_features)]))

    
    # 第二阶段：应用归一化并创建图数据
    grouped_data = {}
    
    for name, data in temp_data.items():
        # 对标签进行分组编码（baseline/non_baseline）
        encoded_labels, label_encoder = encode_labels(data['raw_labels'])
        
        normalized_graphs = []
        for feat in data['features']:
            if normalize:
                orig_shape = feat.shape
                # 进行随机降采样，保留一半节点
                downsampled_feat = random_downsample(feat)
                
                # 然后对降采样后的特征进行归一化
                normalized = scaler.transform(downsampled_feat.reshape(-1, orig_shape[1])).reshape(downsampled_feat.shape)
            else:
                normalized = feat
            print("Feature range after normalization: ", normalized.min(), normalized.max())

            edge_index = generate_full_edges(normalized.shape[0])
            graph_data = Data(
                x=torch.tensor(normalized, dtype=torch.float),
                edge_index=edge_index
            )
            normalized_graphs.append(graph_data)
        
        grouped_data[name] = {
            'graphs': normalized_graphs,
            'labels': encoded_labels,
            'label_encoder': label_encoder,
            'raw_labels': data['raw_labels']  # 保留原始标签信息
        }
    
    return grouped_data

# 辅助函数（与你提供的相同）
def extract_label(filename):
    """Extracts the label from the filename."""
    return filename.split("_")[-2]

def encode_labels(labels):
    """Encodes string labels into numeric labels (baseline vs non_baseline)."""
    grouped_labels = np.array(["baseline" if label == "baseline" else "non_baseline" for label in labels])
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(grouped_labels)
    logging.info(f"Encoded labels mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    return encoded_labels, label_encoder

def generate_full_edges(num_nodes):
    edges = list(combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
