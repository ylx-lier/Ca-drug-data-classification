#!/usr/bin/env python3
"""
改进的简单图自编码器 - 针对loss下降慢的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

class ImprovedSimpleGraphAutoEncoder(nn.Module):
    """改进的简单图自编码器"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # 增加模型容量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        
        # GCN layers for graph-level embedding
        self.gcn1 = GCNConv(input_dim, 128)
        self.gcn2 = GCNConv(128, hidden_dim)
        
    def forward(self, x, edge_index, batch):
        # Node-level encoding
        node_encoded = self.encoder(x)
        
        # Graph-level encoding with GCN
        gcn_out = F.relu(self.gcn1(x, edge_index))
        gcn_out = self.gcn2(gcn_out, edge_index)
        graph_embedding = global_mean_pool(gcn_out, batch)
        
        # Combine encodings
        combined_encoding = node_encoded + gcn_out
        
        # Decode
        reconstructed = self.decoder(combined_encoding)
        
        return graph_embedding, reconstructed

def improved_loss_function(reconstructed, original, embeddings, alpha=0.7, beta=0.2, gamma=0.1):
    """改进的损失函数"""
    # 重构损失组合
    mse_loss = F.mse_loss(reconstructed, original)
    mae_loss = F.l1_loss(reconstructed, original)
    recon_loss = alpha * mse_loss + (1-alpha) * mae_loss
    
    # 嵌入正则化
    reg_loss = beta * torch.norm(embeddings, p=2, dim=1).mean()
    
    # 平滑性约束（相邻节点应该相似）
    smooth_loss = gamma * F.mse_loss(reconstructed[:-1], reconstructed[1:])
    
    return recon_loss + reg_loss + smooth_loss, {
        'recon': recon_loss.item(),
        'reg': reg_loss.item(),
        'smooth': smooth_loss.item()
    }

def train_improved_gae(model, graph_data_list, paths, epochs=50, lr=1e-3, batch_size=32):
    """改进的训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 改进的优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 改进的学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.8, verbose=True, min_lr=1e-6
    )
    
    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    logging.info("开始训练改进的图自编码器...")
    writer = SummaryWriter(paths["tensorboard_path"])
    
    for epoch in tqdm(range(epochs), desc="训练进度"):
        total_loss = 0
        total_losses = {'recon': 0, 'reg': 0, 'smooth': 0}
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            graph_embedding, reconstructed = model(batch.x, batch.edge_index, batch.batch)
            
            # 改进的损失计算
            loss, loss_components = improved_loss_function(
                reconstructed, batch.x, graph_embedding
            )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            for key in loss_components:
                total_losses[key] += loss_components[key] * batch.num_graphs
            num_batches += batch.num_graphs
        
        avg_loss = total_loss / num_batches
        avg_losses = {k: v/num_batches for k, v in total_losses.items()}
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        loss_history.append(avg_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Reconstruction', avg_losses['recon'], epoch)
        writer.add_scalar('Loss/Regularization', avg_losses['reg'], epoch)
        writer.add_scalar('Loss/Smoothness', avg_losses['smooth'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if (epoch + 1) % 5 == 0:  # 更频繁的日志
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} "
                        f"(Recon: {avg_losses['recon']:.6f}, Reg: {avg_losses['reg']:.6f}), "
                        f"LR: {lr:.8f}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{paths['model_path']}/best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= 15:  # 早停
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    logging.info(f"训练完成，最佳损失: {best_loss:.6f}")
    
    return model, loss_history

if __name__ == "__main__":
    print("这是改进的图自编码器模块")
    print("主要改进:")
    print("1. 增加了模型容量和BatchNorm")
    print("2. 组合损失函数 (MSE + MAE + 正则化 + 平滑性)")
    print("3. 自适应学习率调度")
    print("4. 梯度裁剪和早停")
    print("5. 更详细的损失分解记录")
