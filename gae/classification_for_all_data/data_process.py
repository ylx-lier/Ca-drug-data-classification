# Step 1: Split Code
# data_process.py
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from resize_interpolate import linear_interpolate
import logging
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
            matrix = linear_interpolate(matrix, target_dim=1000)
            graphs.append(matrix)
            labels.append(label)
    logging.info(f"Total samples loaded: {len(graphs)}")
    logging.info(f"Sample matrix shape: {graphs[0].shape}")
    logging.info(f"Unique labels: {np.unique(labels)}")
    return graphs, np.array(labels)

def encode_labels(labels):
    """Encodes string labels into numeric labels."""
    grouped_labels = np.array(["baseline" if label == "baseline" else "non_baseline" for label in labels])
    label_encoder = LabelEncoder()
    encode_labels = label_encoder.fit_transform(grouped_labels)
    logging.info(f"Encoded labels mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    return encode_labels, label_encoder