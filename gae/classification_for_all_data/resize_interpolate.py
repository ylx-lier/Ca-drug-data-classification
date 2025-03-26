import torch
import torch.nn.functional as F
import numpy as np

def linear_interpolate(x:np.ndarray, target_dim):
    """
    将图的节点特征矩阵插值到目标维度 (target_dim)，
    使得所有图的特征维度一致。
    
    参数:
        x: 原始的图特征矩阵 (num_nodes, num_features)
        target_dim: 目标维度（统一后的特征维度，例如 4570）
    
    返回:
        插值后的特征矩阵
    """
    num_nodes, num_features = x.shape
    
    # 使用元组表示新形状
    if num_features != target_dim:
        
        x = x.reshape(1, num_nodes, num_features)
        x = torch.tensor(x, dtype=torch.float)
        x = F.interpolate(x, size=target_dim, mode='linear', align_corners=False)
        x = x.squeeze(0)
        x = x.numpy()
        

    return x


