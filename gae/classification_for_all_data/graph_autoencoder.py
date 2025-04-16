# graph_autoencoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn.functional as F
from itertools import combinations
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def generate_full_edges(num_nodes):
    edges = list(combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

class Encoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels*2)
        # self.conv4 = GCNConv(hidden_channels*2, hidden_channels)
        self.linear = nn.Linear(num_node_features, hidden_channels)

    def forward(self, x, edge_index, batch):
        # print(f"Input shape: {x.shape}")
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        # print(f"After conv1: {x.shape}")
        # x = F.relu(self.conv2(x, edge_index))
        # # print(f"After conv2: {x.shape}")
        # x = F.relu(self.conv3(x, edge_index))
        # print(f"After conv3: {x.shape}")
        x = self.conv2(x, edge_index)
        # print(f"After conv4: {x.shape}")
        identity = self.linear(identity)
        # print(f"Adjusted identity shape: {identity.shape}")
        x += identity
        x = global_mean_pool(x, batch)
        # print(f"Final embedding shape: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_node_features)
      
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.leaky_relu(x, 0.5)
        return x

class GraphAutoEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.encoder = Encoder(num_node_features, hidden_channels)
        self.decoder = Decoder(hidden_channels, num_node_features)
        
        self.pos_embedding = nn.Parameter(torch.randn(1,1,hidden_channels))
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=4,
            dim_feedforward=hidden_channels * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=1)

    def forward(self, x, edge_index, batch):
        embedding = self.encoder(x, edge_index, batch)
        # 注意力增强
        embedding = embedding.unsqueeze(1)  # (1, batch_size, hidden)
        embedding = embedding + self.pos_embedding
        embedding = self.transformer(embedding)
        embedding = embedding.squeeze(1)  # (batch_size, hidden)
        
        decoded = self.decoder(embedding[batch], edge_index)
        return embedding, decoded

def train_graph_autoencoder(model, graph_data_list, paths, epochs=100, lr=5e-4, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)
    
    model.train()
    loss_values = []
    logging.info("Start training...")
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, decoded = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(decoded, batch.x)
            loss.backward()
            optimizer.step()
            logging.info(f"loss: {loss.item()}, len(batch): {batch.num_graphs}")
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(graph_data_list)
        logging.info(f"total loss: {total_loss:.4f}, len(graph_data_list): {len(graph_data_list)}")
        loss_values.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存训练曲线
    plt.figure(figsize=(10,5))
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
        # 创建batch参数（所有节点属于同一图）
        batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        with torch.no_grad():
            embedding, _ = model(x, edge_index, batch)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)