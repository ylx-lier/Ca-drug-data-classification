# simple_gae.py - 简单可靠的图自编码器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def create_model(model_type="simple", num_node_features=None, hidden_channels=64, **kwargs):
    """
    创建图自编码器模型的统一接口
    
    参数:
        model_type: "simple", "graphmae", 或 "original"
        num_node_features: 节点特征维度
        hidden_channels: 隐藏层维度
        **kwargs: 其他模型参数
    """
    if model_type.lower() == "simple":
        embedding_dim = kwargs.get('embedding_dim', 32)
        return SimpleGraphAutoEncoder(num_node_features, hidden_channels, embedding_dim)
    elif model_type.lower() == "graphmae":
        try:
            from graph_mae import GraphMAE
            mask_ratio = kwargs.get('mask_ratio', 0.3)
            return GraphMAE(num_node_features, hidden_channels, mask_ratio)
        except ImportError:
            logging.warning("GraphMAE模块未找到，使用SimpleGraphAutoEncoder代替")
            embedding_dim = kwargs.get('embedding_dim', 32)
            return SimpleGraphAutoEncoder(num_node_features, hidden_channels, embedding_dim)
    elif model_type.lower() == "original":
        try:
            from graph_autoencoder import GraphAutoEncoder
            return GraphAutoEncoder(num_node_features, hidden_channels)
        except ImportError:
            logging.warning("原始GraphAutoEncoder模块未找到，使用SimpleGraphAutoEncoder代替")
            embedding_dim = kwargs.get('embedding_dim', 32)
            return SimpleGraphAutoEncoder(num_node_features, hidden_channels, embedding_dim)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: 'simple', 'graphmae', 'original'")

def train_model(model, graph_data_list, paths, model_type="simple", epochs=50, lr=1e-3, **kwargs):
    """
    训练图自编码器的统一接口
    """
    model_class_name = model.__class__.__name__
    
    if 'GraphMAE' in model_class_name:
        # 使用GraphMAE的训练函数
        try:
            from graph_mae import train_graph_mae
            return train_graph_mae(model, graph_data_list, paths, epochs, lr)
        except ImportError:
            logging.warning("GraphMAE训练函数未找到，使用Simple训练方法")
            return train_simple_gae(model, graph_data_list, paths, epochs, lr, **kwargs)
    elif 'GraphAutoEncoder' in model_class_name and 'Simple' not in model_class_name:
        # 使用原始GraphAutoEncoder的训练函数
        try:
            from graph_autoencoder import train_graph_autoencoder
            batch_size = kwargs.get('batch_size', 32)
            return train_graph_autoencoder(model, graph_data_list, paths, epochs, lr, batch_size)
        except ImportError:
            logging.warning("原始GraphAutoEncoder训练函数未找到，使用Simple训练方法")
            return train_simple_gae(model, graph_data_list, paths, epochs, lr, **kwargs)
    else:
        # 使用Simple的训练函数
        batch_size = kwargs.get('batch_size', 32)
        return train_simple_gae(model, graph_data_list, paths, epochs, lr, batch_size)

def generate_embeddings(model, graphs, model_type="simple"):
    """
    生成图嵌入的统一接口
    """
    model_class_name = model.__class__.__name__
    
    if 'GraphMAE' in model_class_name:
        # 使用GraphMAE的嵌入生成函数
        try:
            from graph_mae import generate_graph_embeddings
            return generate_graph_embeddings(model, graphs)
        except ImportError:
            logging.warning("GraphMAE嵌入函数未找到，使用Simple方法")
            return generate_simple_embeddings(model, graphs)
    elif 'GraphAutoEncoder' in model_class_name and 'Simple' not in model_class_name:
        # 使用原始GraphAutoEncoder的嵌入生成函数
        try:
            from graph_autoencoder import generate_graph_embeddings
            return generate_graph_embeddings(model, graphs)
        except ImportError:
            logging.warning("原始GraphAutoEncoder嵌入函数未找到，使用Simple方法")
            return generate_simple_embeddings(model, graphs)
    else:
        # 使用Simple的嵌入生成函数
        return generate_simple_embeddings(model, graphs)

class SimpleGraphAutoEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, embedding_dim=32):
        super().__init__()
        
        # 编码器：节点特征 -> 图嵌入
        self.encoder1 = GCNConv(num_node_features, hidden_channels)
        self.encoder2 = GCNConv(hidden_channels, hidden_channels)
        self.encoder3 = GCNConv(hidden_channels, embedding_dim)
        
        # 解码器：图嵌入 -> 节点特征重构
        self.decoder1 = GCNConv(embedding_dim, hidden_channels)
        self.decoder2 = GCNConv(hidden_channels, hidden_channels)
        self.decoder3 = GCNConv(hidden_channels, num_node_features)
        
        self.dropout = nn.Dropout(0.1)
        
    def encode(self, x, edge_index, batch):
        """编码：从节点特征生成图级别嵌入"""
        # 节点级别编码
        h1 = F.relu(self.encoder1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.encoder2(h1, edge_index))
        h2 = self.dropout(h2)
        
        node_embeddings = self.encoder3(h2, edge_index)
        
        # 图级别池化
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return graph_embedding, node_embeddings
    
    def decode(self, node_embeddings, edge_index):
        """解码：从节点嵌入重构原始特征"""
        h1 = F.relu(self.decoder1(node_embeddings, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.decoder2(h1, edge_index))
        h2 = self.dropout(h2)
        
        reconstructed = self.decoder3(h2, edge_index)
        
        return reconstructed
    
    def forward(self, x, edge_index, batch):
        graph_embedding, node_embeddings = self.encode(x, edge_index, batch)
        reconstructed = self.decode(node_embeddings, edge_index)
        
        return graph_embedding, reconstructed

def train_simple_gae(model, graph_data_list, paths, epochs=50, lr=1e-3, batch_size=32):
    """训练简单图自编码器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    loss_values = []
    
    logging.info("开始训练简单图自编码器...")
    writer = SummaryWriter(paths["tensorboard_path"])
    
    for epoch in tqdm(range(epochs), desc="训练进度"):
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            graph_embedding, reconstructed = model(batch.x, batch.edge_index, batch.batch)
            
            # 重构损失（MSE）
            recon_loss = F.mse_loss(reconstructed, batch.x)
            
            # 可选：添加嵌入正则化
            reg_loss = 0.01 * torch.norm(graph_embedding, p=2, dim=1).mean()
            
            total_loss_batch = recon_loss + reg_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item() * batch.num_graphs
            num_batches += batch.num_graphs
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        loss_values.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
    writer.close()
    
    # 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(paths["loss_path"])
    plt.close()
    
    return model

def generate_simple_embeddings(model, graphs):
    """生成图嵌入"""
    model.eval()
    embeddings = []
    device = next(model.parameters()).device
    
    for graph in tqdm(graphs, desc="生成嵌入"):
        num_nodes = graph.x.shape[0]
        
        # 确保edge_index正确
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            edge_index = graph.edge_index.to(device)
        else:
            # 如果没有edge_index，生成全连接图
            from itertools import combinations
            edges = list(combinations(range(num_nodes), 2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        
        x = graph.x.to(device) if hasattr(graph.x, 'to') else torch.tensor(graph.x, dtype=torch.float).to(device)
        batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        
        with torch.no_grad():
            graph_embedding, _ = model.encode(x, edge_index, batch)
        
        embeddings.append(graph_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

# 向后兼容的函数名
def GraphMAE(num_node_features, hidden_channels):
    """为了向后兼容，返回SimpleGraphAutoEncoder"""
    return SimpleGraphAutoEncoder(num_node_features, hidden_channels)

def train_graph_mae(model, graph_data_list, paths, epochs=50, lr=1e-3):
    """为了向后兼容"""
    return train_simple_gae(model, graph_data_list, paths, epochs, lr)

def generate_graph_embeddings(model, graphs):
    """为了向后兼容"""
    return generate_simple_embeddings(model, graphs)
