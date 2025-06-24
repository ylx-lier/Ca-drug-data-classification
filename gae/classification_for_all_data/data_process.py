#data_process.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from resize_interpolate import linear_interpolate
from torch_geometric.data import Data
import torch
from itertools import combinations
import logging
from tqdm import tqdm
# 导入傅里叶变换相关包
from scipy import fft
np.random.seed(42)
def time_slice(features, start_idx, end_idx):
    """
    对特征矩阵进行时间切片，只保留指定区间的特征（列）。

    参数:
        features: 原始特征矩阵 (num_nodes, num_features)
        start_idx: 起始时间步的索引
        end_idx: 结束时间步的索引（不包含）
    
    返回:
        sliced_features: 切片后的特征矩阵
    """
    return features[:, start_idx:end_idx]

def random_downsample(features, seed=None):
    """
    对特征矩阵进行随机降采样，保留一半的节点（行）。

    参数:
        features: 原始特征矩阵 (num_nodes, num_features)
        seed: 随机种子，用于保证实验可重现性
    
    返回:
        downsampled_features: 降采样后的特征矩阵
    """
    # 获取原始特征矩阵的行数
    num_nodes = features.shape[0]

    # 计算保留一半节点的数量
    num_samples = num_nodes // 2
    
    # 设置局部随机种子以确保可重现性
    if seed is not None:
        rng = np.random.RandomState(seed)
        selected_indices = rng.choice(num_nodes, num_samples, replace=False)
    else:
        # 使用hash(features.data.tobytes())作为种子，确保相同数据产生相同结果
        data_seed = hash(features.data.tobytes()) % (2**31)
        rng = np.random.RandomState(data_seed)
        selected_indices = rng.choice(num_nodes, num_samples, replace=False)
    
    downsampled_features = features[selected_indices]
    
    return downsampled_features

def load_and_normalize_datasets(data_paths, normalize=True, time_start=0, time_end=1000, full_data=False):
    """
    加载并归一化多个数据集，保留数据集来源信息
    full_data: 若为True，则不做时间切片和节点降采样，直接用全量特征
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
    # 直接加载数据并处理
    temp_data = {}

    for path in tqdm(data_paths, desc="加载数据"):
        
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
    
    # 应用归一化并创建图数据
    grouped_data = {}
    
    for name, data in temp_data.items():
        # 对标签进行分组编码（baseline/non_baseline）
        encoded_labels, label_encoder = encode_labels(data['raw_labels'])
        
        normalized_graphs = []
        for feat in data['features']:
            if normalize:
                # 对每一行做归一化
                min_vals = feat.min(axis=1, keepdims=True)
                max_vals = feat.max(axis=1, keepdims=True)
                # 防止除零
                denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
                normalized = (feat - min_vals) / denom
            else:
                normalized = feat

            if full_data:
                used_feat = normalized
            else:
                # 先做时间切片
                sliced_feat = time_slice(normalized, start_idx=time_start, end_idx=time_end)
                # 再做节点降采样，使用固定种子确保可重现性
                used_feat = random_downsample(sliced_feat, seed=42)

            logging.info(f"Feature range after normalization: {used_feat.min()} to {used_feat.max()}")

            edge_index = generate_full_edges(used_feat.shape[0])
            graph_data = Data(
                x=torch.tensor(used_feat, dtype=torch.float),
                edge_index=edge_index
            )
            normalized_graphs.append(graph_data)
        grouped_data[name] = {
            'graphs': normalized_graphs,
            'labels': encoded_labels,
            'label_encoder': label_encoder,
            'raw_labels': data['raw_labels']
        }
    return grouped_data

# 辅助函数（与你提供的相同）
def extract_label(filename):
    

    parts = filename.split("_")
    for token in parts:
        if token in ['Fore', 'ACM']:
            return token
    raise ValueError(f"No valid label in'{filename}'")


def encode_labels(labels):
    label_encoder = LabelEncoder()
    
    # 手动指定顺序 ['Fore', 'ACM'] -> Fore 编码为 0，ACM 编码为 1
    label_encoder.classes_ = np.array(['Fore', 'ACM'])
    
    encoded_labels = label_encoder.transform(labels)
    logging.info(f"Manual label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    return encoded_labels, label_encoder

def generate_full_edges(num_nodes):
    edges = list(combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def extract_fft_features(features, num_components=None):
    """
    对特征矩阵进行傅里叶变换提取频域特征。
    
    参数:
        features: 原始特征矩阵 (num_nodes, num_features)
        num_components: 保留的频域分量数量，默认为None(保留所有分量)
        
    返回:
        fft_features: 傅里叶变换后的特征矩阵
    """
    num_nodes, num_features = features.shape
    
    # 对每个节点的时间序列进行FFT
    fft_features = []
    for i in range(num_nodes):
        # 进行FFT变换
        fft_result = fft.rfft(features[i])
        
        # 取幅值作为特征（复数的模）
        magnitudes = np.abs(fft_result)
        
        # 如果指定了要保留的分量数量，则只保留前num_components个分量
        if num_components is not None and num_components < len(magnitudes):
            magnitudes = magnitudes[:num_components]
            
        fft_features.append(magnitudes)
    
    return np.array(fft_features)

if __name__ == "__main__":
    # 简单归一化例子
    # 输入两个矩阵
    features1 = np.array([[1, 2, 3], [4, 5, 6]])
    features2 = np.array([[7, 8, 9], [10, 11, 12]])
    all_features = [features1, features2]
    for feat in all_features:
        scaler = MinMaxScaler()
        # 对每个样本单独fit和transform
        normalized = scaler.fit_transform(feat)
        print("Normalized features:\n", normalized)