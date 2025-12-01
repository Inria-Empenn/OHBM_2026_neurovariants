import os

import matplotlib.pyplot as plt
import nilearn.image as nimg
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import seaborn as sb


def plot_brain(img_path, coords, ref_path=None):
    img = nimg.load_img(img_path)
    config = os.path.basename(os.path.dirname(img_path))
    img_label = f'{config[:6]}'

    axes_idx = [0, 1, 2]
    if ref_path:
        axes_idx = [1, 3, 5, 0, 2, 4]

    fig, axes = plt.subplots(1, len(axes_idx), figsize=(15, 5))

    plotting.plot_stat_map(img, cut_coords=[coords[0]], display_mode='x', axes=axes[axes_idx[0]], vmin=-3, vmax=3,
                           title=img_label, colorbar=False)
    plotting.plot_stat_map(img, cut_coords=[coords[1]], display_mode='y', axes=axes[axes_idx[1]], vmin=-3, vmax=3,
                           title=img_label, colorbar=False)
    plotting.plot_stat_map(img, cut_coords=[coords[2]], display_mode='z', axes=axes[axes_idx[2]], vmin=-3, vmax=3,
                           title=img_label, colorbar=False)
    if ref_path:
        ref_img = nimg.load_img(ref_path)
        ref_label = f'{os.path.splitext(os.path.basename(ref_path))[0]}'
        plotting.plot_stat_map(ref_img, cut_coords=[coords[0]], display_mode='x', axes=axes[axes_idx[3]],
                               title=ref_label, colorbar=False)
        plotting.plot_stat_map(ref_img, cut_coords=[coords[1]], display_mode='y', axes=axes[axes_idx[4]],
                               title=ref_label, colorbar=False)
        plotting.plot_stat_map(ref_img, cut_coords=[coords[2]], display_mode='z', axes=axes[axes_idx[5]],
                               title=ref_label, colorbar=False)
    plt.show()


def get_cluster_distance_densities(dist_values, cluster_labels):
    distance_densities = {}
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        cluster_dist = dist_values[mask][:, mask]
        avg_dist = (np.sum(cluster_dist) - np.sum(np.diag(cluster_dist))) / (cluster_dist.size - len(cluster_dist))
        distance_densities[label] = avg_dist
    return distance_densities


def get_cluster_inertia(dist_values, cluster_labels):
    inertia = 0
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        cluster_dist = dist_values[mask][:, mask]
        centroid_dist = np.mean(cluster_dist)
        inertia += np.sum(cluster_dist ** 2) - len(cluster_dist) * centroid_dist ** 2
    return inertia


def get_cluster_cophenetic(Z, distance_values):
    coph_dist, _ = cophenet(Z, squareform(distance_values))
    return coph_dist


def get_cluster_silhouette(distance_values, cluster_labels):
    if len(np.unique(cluster_labels)) < 2:
        return None
    return silhouette_score(distance_values, cluster_labels, metric='precomputed')


def get_davies_bouldin(distance_values, cluster_labels):
    if len(np.unique(cluster_labels)) < 2:
        return None
    return davies_bouldin_score(distance_values, cluster_labels)


def get_medoids(distance_matrix: pd.DataFrame, clusters: pd.Series):
    # medoid = point with the smallest sum of distances

    medoids = {}
    unique_clusters = clusters.unique()

    for cluster in unique_clusters:
        cluster_indices = clusters[clusters == cluster].index
        cluster_distance_matrix = distance_matrix.loc[cluster_indices, cluster_indices]
        sum_distances = cluster_distance_matrix.sum(axis=1)
        medoid_id = sum_distances.idxmin()
        medoids[cluster] = medoid_id

    return medoids


def plot_dendogram(z_linkage):
    plt.figure(figsize=(10, 5))
    dendrogram(z_linkage, truncate_mode='lastp', p=50)  # Show last 50 merges
    plt.title('Dendrogram')
    plt.show()


def plot_heatmap(matrix, z_linkage):
    plt.figure(figsize=(12, 12))
    cmap = LinearSegmentedColormap.from_list("red_cmap", ["#FFCCCC", "#FF0000"])
    sb.clustermap(matrix, cmap=cmap, vmin=0, vmax=1, row_cluster=False, col_cluster=False, row_linkage=z_linkage,
                  col_linkage=z_linkage)
    plt.title('Correlation Heatmap')
    plt.show()