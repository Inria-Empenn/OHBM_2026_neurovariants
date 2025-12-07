import os

import matplotlib.pyplot as plt
import nilearn.image as nimg
import seaborn as sb
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
from scipy.cluster.hierarchy import dendrogram


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