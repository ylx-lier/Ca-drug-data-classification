'''
# train.py
from sklearn.model_selection import train_test_split
from data_process import load_graph_data, encode_labels
from graph_autoencoder import GraphAutoEncoder, train_graph_autoencoder, generate_graph_embeddings
from classification import Classifier
import torch

def train_and_evaluate(folder_path, model_save_path):
    """Loads data, trains GAE and classifier, and evaluates them."""
    # Load graph data and labels
    graphs, labels = load_graph_data(folder_path)
    for i in range(len(graphs)):
        print(graphs[i].shape)

    # Encode labels
    y_encoded, label_encoder = encode_labels(labels)

    # Initialize GAE
    input_dim = len(graphs[0][1])  # Node feature dimension
    print("input_dim: ", input_dim)
    model = GraphAutoEncoder(num_node_features=input_dim, hidden_channels=64)

    # Train GAE
    model = train_graph_autoencoder(model, graphs, epochs=100, lr=0.01)

    # Generate graph embeddings
    embeddings = generate_graph_embeddings(model, graphs)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    clf = Classifier()
    clf.train(X_train, y_train)

    # Evaluate the classifier
    accuracy, y_pred = clf.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the trained model
    clf.save_model(model_save_path)

    return label_encoder, y_pred
'''
# train.py
from sklearn.model_selection import train_test_split
from data_process import load_graph_data, encode_labels
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
    """Loads data, trains GAE and classifier, and evaluates them."""
    # Load graph data and labels
    logging.info("loading data...")
    graphs, labels = load_graph_data(paths["data_path"])
    
    # Encode labels
    y_encoded, label_encoder = encode_labels(labels)
    
    # Split data into train and test sets first to avoid data leakage
    train_graphs, test_graphs, y_train, y_test = train_test_split(
        graphs, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Initialize GAE with input dimension from training data
    input_dim = train_graphs[0].shape[1]  # num_node_features
    model = GraphAutoEncoder(num_node_features=input_dim, hidden_channels=64)
    
    # Train GAE on training graphs only
    logging.info("Starting training...")
    model = train_graph_autoencoder(model, train_graphs, epochs=10, lr=5e-4)
    
    # Generate embeddings for train and test sets
    train_embeddings = generate_graph_embeddings(model, train_graphs)
    test_embeddings = generate_graph_embeddings(model, test_graphs)
    visualize_embeddings(train_embeddings, y_train, label_encoder, save_path=paths["embedding_plot_path"])
    # Train classifier
    clf = Classifier()
    clf.train(train_embeddings, y_train)
    
    # Evaluate
    metrics, y_pred = clf.evaluate(test_embeddings, y_test)
    logging.info(f"Accuracy: {metrics['accuracy']:.2f}")
    logging.info("Classification Report:\n", metrics['classification_report'])
    logging.info("Confusion Matrix:\n", metrics['confusion_matrix'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(paths["confusion_matrix_path"])
    # clf.save_model(model_save_path)
    return label_encoder, y_pred