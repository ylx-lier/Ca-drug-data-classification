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

    # Encode labels
    y_encoded, label_encoder = encode_labels(labels)

    # Initialize GAE
    input_dim = graphs[0].shape[1]  # Node feature dimension
    print("input_dim: ", input_dim)
    model = GraphAutoEncoder(num_node_features=input_dim, hidden_channels=64)

    # Train GAE
    model = train_graph_autoencoder(model, graphs, batch_size=8, epochs=50, lr=0.01)

    # Generate graph embeddings
    embeddings = generate_graph_embeddings(model, graphs, batch_size=8)

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