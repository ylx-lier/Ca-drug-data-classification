# Step 1: Split Code
# data_process.py
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from resize_interpolate import linear_interpolate

def extract_label(filename):
    """Extracts the label from the filename."""
    return filename.split("_")[-2]

def load_graph_data(folder_path):
    """Loads graph data and their labels from a folder."""
    graphs = []
    labels = []

    for file in os.listdir(folder_path):
        if file.endswith("_spike.csv"):
            filepath = os.path.join(folder_path, file)
            matrix = np.loadtxt(filepath, delimiter=',')
            label = extract_label(file)
            matrix = linear_interpolate(matrix, 4570)
            graphs.append(matrix)
            labels.append(label)
    print(f"Total samples loaded: {len(graphs)}")
    print(f"Sample matrix shape: {graphs[0].shape}")
    print(f"Unique labels: {np.unique(labels)}")
    return graphs, np.array(labels)

def encode_labels(labels):
    """Encodes string labels into numeric labels."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels), label_encoder