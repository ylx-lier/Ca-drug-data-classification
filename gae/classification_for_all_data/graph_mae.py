# graph_mae.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from itertools import combinations
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

def generate_full_edges(num_nodes):
    edges = list(combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

class GraphMAEEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        # GraphMAE-style encoder with deeper architecture
        self.encoder1 = GCNConv(num_node_features, hidden_channels)
        self.encoder2 = GCNConv(hidden_channels, hidden_channels*2)
        self.encoder3 = GCNConv(hidden_channels*2, hidden_channels)
        
        # Projection head like in GraphMAE paper
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Mask token embedding
        self.register_parameter('mask_token', 
                              nn.Parameter(torch.zeros(1, num_node_features)))
        nn.init.xavier_uniform_(self.mask_token.data)
        
        # Scale factor for normalization
        self.scale_factor = math.sqrt(hidden_channels)

    def forward(self, x, edge_index, mask=None):
        if mask is not None:
            mask_token = self.mask_token.to(x.device)
            assert mask.shape[0] == x.shape[0], f"Mask shape {mask.shape} doesn't match x shape {x.shape}"
            # Replace masked nodes with mask token
            x = x * (~mask).unsqueeze(-1).float() + mask_token * mask.unsqueeze(-1).float()
        
        # Encoder with residual connections
        h1 = F.relu(self.encoder1(x, edge_index))
        h2 = F.relu(self.encoder2(h1, edge_index))
        h3 = self.encoder3(h2, edge_index) + h1  # Skip connection
        
        # Projection
        h = self.proj_head(h3)
        return h / self.scale_factor  # Normalize as in GraphMAE

class GraphMAEDecoder(nn.Module):
    def __init__(self, hidden_channels, num_node_features):
        super().__init__()
        # Decoder with 2 GCN layers as in GraphMAE
        self.decoder1 = GCNConv(hidden_channels, hidden_channels)
        self.decoder2 = GCNConv(hidden_channels, num_node_features)
        
        # Layer norm as in original paper
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = F.relu(self.decoder1(x, edge_index))
        x = self.decoder2(x, edge_index)
        return x

class GraphMAE(nn.Module):
    def __init__(self, num_node_features, hidden_channels, mask_ratio=0.3):
        super().__init__()
        self.encoder = GraphMAEEncoder(num_node_features, hidden_channels)
        self.decoder = GraphMAEDecoder(hidden_channels, num_node_features)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, num_node_features))
        nn.init.xavier_uniform_(self.mask_token.data)
        
        # Positional embedding for graph-level features
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))
        
        # Transformer encoder for graph-level processing
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=4,
            dim_feedforward=hidden_channels*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)
    def to(self, device):
        # 覆盖to()方法，确保mask_token也转移设备
        super().to(device)
        self.mask_token = self.mask_token.to(device)
        return self
    def random_masking(self, x, batch):
        """批处理友好的随机掩码生成"""
        num_nodes_per_graph = torch.bincount(batch)
        masks = []
        for num_nodes in num_nodes_per_graph:
            num_mask = max(1, int(num_nodes * self.mask_ratio))
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
            mask_indices = torch.randperm(num_nodes, device=x.device)[:num_mask]
            mask[mask_indices] = True
            masks.append(mask)
        return torch.cat(masks)  # 合并所有图的mask

    def forward(self, x, edge_index, batch):
        # Generate random mask
        mask = self.random_masking(x, batch)
        
        # Encode with masking
        node_embeddings = self.encoder(x, edge_index, mask)
        
        # Graph-level embedding with pooling
        graph_embedding = global_mean_pool(node_embeddings, batch)
        graph_embedding = graph_embedding.unsqueeze(1) + self.pos_embedding
        graph_embedding = self.transformer(graph_embedding).squeeze(1)
        
        # Reconstruct all nodes (including masked ones)
        reconstructed = self.decoder(node_embeddings, edge_index)
        
        return graph_embedding, reconstructed, mask

def train_graph_mae(model, graph_data_list, paths, epochs=100, lr=5e-4, batch_size=32):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torch_geometric.loader import DataLoader
    # Cosine learning rate scheduler as in GraphMAE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True,follow_batch=[], num_workers=0)
    
    model.train()
    loss_values = []
    logging.info("Start training GraphMAE...")
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            # logging.info(batch.x.shape) #看是否要注释掉
            
            
            
            optimizer.zero_grad()
            
            _, reconstructed, mask = model(batch.x, batch.edge_index, batch.batch)
            
            # Only compute loss on masked nodes
            loss = F.mse_loss(reconstructed[mask], batch.x[mask])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        scheduler.step()
        
        avg_loss = total_loss / len(graph_data_list)
        loss_values.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.savefig(paths["loss_path"])
    plt.close()
    
    return model

def generate_graph_embeddings(model, graphs):
    model.eval()
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for graph in graphs:
        num_nodes = graph.x.shape[0]
        edge_index = generate_full_edges(num_nodes).to(device)
        x = torch.tensor(graph.x, dtype=torch.float).to(device)
        batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        
        with torch.no_grad():
            embedding, _, _ = model(x, edge_index, batch)
        
        embeddings.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings)