import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

def hierarchical_clustering_reorder(M, method='ward', metric='euclidean'):
    """
    Perform hierarchical clustering on a matrix M and reorder its rows.

    Parameters:
        M (numpy.ndarray): Input matrix (rows represent nodes, columns are feature dimensions).
                          The columns represent time series signals.
        method (str): Linkage method for hierarchical clustering (default: 'ward').
        metric (str): Distance metric for clustering (default: 'euclidean').
                      'euclidean' works well for direct comparison, but for time series,
                      'correlation' or custom metrics might be more suitable.

    Returns:
        numpy.ndarray: Reordered matrix where similar rows are closer together.
    """
    # Ensure input is a numpy array
    M = np.asarray(M)

    # Compute pairwise distances using correlation metric to account for phase shift similarity
    distance_matrix = pdist(M, metric='correlation')

    # Perform hierarchical clustering using the computed distance matrix
    linkage_matrix = linkage(distance_matrix, method=method)

    # Get the order of rows based on hierarchical clustering
    order = leaves_list(linkage_matrix)

    # Reorder the matrix
    reordered_matrix = M[order]

    return reordered_matrix
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

def hierarchical_clustering_reorder(M, method='ward', metric='euclidean', plot=True):
    """
    Perform hierarchical clustering on a matrix M and reorder its rows.

    Parameters:
        M (numpy.ndarray): Input matrix (rows represent nodes, columns are feature dimensions).
                          The columns represent time series signals.
        method (str): Linkage method for hierarchical clustering (default: 'ward').
        metric (str): Distance metric for clustering (default: 'euclidean').
                      'euclidean' works well for direct comparison, but for time series,
                      'correlation' or custom metrics might be more suitable.
        plot (bool): Whether to plot the reordered matrix as line plots (default: True).

    Returns:
        numpy.ndarray: Reordered matrix where similar rows are closer together.
    """
    # Ensure input is a numpy array
    M = np.asarray(M)

    # Compute pairwise distances using correlation metric to account for phase shift similarity
    distance_matrix = pdist(M, metric='correlation')

    # Perform hierarchical clustering using the computed distance matrix
    linkage_matrix = linkage(distance_matrix, method=method)

    # Get the order of rows based on hierarchical clustering
    order = leaves_list(linkage_matrix)

    # Reorder the matrix
    reordered_matrix = M[order]

    # Plot the reordered matrix as line plots in local coordinates
    if plot:
        plt.figure(figsize=(10, 15))
        for i, row in enumerate(reordered_matrix):
            plt.plot(row*2 + i, label=f'Node {i+1}', linewidth=1,color='black')  # Offset each row for local coordinate visualization
        plt.title('Reordered Time Series Signals in Local Coordinates')
        plt.xlabel('Time')
        plt.ylabel('Signal Intensity')
        plt.yticks([])  # Remove global y-axis ticks for local coordinate emphasis
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.tight_layout()
        # plt.show()
        plt.savefig("/home/featurize/work/ylx/MEA/gae/subsampling/reordered_signals.png", dpi=1200)

    return reordered_matrix

def matrix_to_image(matrix, colormap='turbo', output_path=None, figsize=(12, 4), pixel_scale=1, interpolation='nearest', extent=None):
     matrix_np = np.array(matrix)
     print(matrix_np.shape)
     matrix_np = hierarchical_clustering_reorder(matrix_np/np.max(matrix_np), plot=True)
    #  matrix_np_dist = np.linalg.norm(matrix_np - matrix_np[:1,:], axis=1)
    #  index=np.argsort(matrix_np_dist)
    #  matrix_np = matrix_np[index] 
     rgba=plt.get_cmap(colormap)(matrix_np)
     f2uint = lambda x: (x*255).astype(np.uint8)
     import imageio.v3 as iio
    #  scaled_matrix = np.repeat(np.repeat(matrix_np, pixel_scale, axis=1), pixel_scale, axis=0)
    #  fig= plt.figure(figsize=figsize)
     iio.imwrite(output_path, f2uint(rgba))
    # #  plt.figure(figsize=figsize)
    #  plt.imshow(matrix_np, cmap=colormap, interpolation=interpolation, extent=extent)
    #  plt.axis('off')
    #  plt.tight_layout()
    #  if output_path:
    #      plt.savefig(output_path, dpi=1200)
    #  else:
    #      plt.show()
    #  plt.close()

def subsample(node_features, target_row, target_col):
    original_rows, original_cols = node_features.shape
    row_step = original_rows / target_row
    col_step = original_cols / target_col
    sampled_row = [int(i*row_step) for i in range(target_row)]
    sampled_col = [int(i*col_step) for i in range(target_col)]
    sampled_matrix = node_features[sampled_row, :][:, sampled_col]
    return sampled_matrix

file = "/home/featurize/work/ylx/MEA/overfitting/220918_0911_05_F53H9_ACM_GABA_3_spike.csv"
df = pd.read_csv(file)
df = subsample(df.values, 111, 4570)
matrix_to_image(df, output_path="/home/featurize/work/ylx/MEA/gae/subsampling/sample_sequence.png")