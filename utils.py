import numpy as np
import pandas as pd
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


def clusterize_hierch(dataset: pd.DataFrame, correlations: pd.DataFrame, ids: list, nb_clusters: object, metric: str,
                      threshold: object = 0.2) -> object:
    """

    :param threshold:
    :param correlations:
    :param dataset: DataFrame
    :param ids: list of configs ids
    :param nb_clusters:
    :param metric: 'spearman', 'dice', jaccard'
    :return:
    """

    filtered_ds = dataset[dataset['id'].isin(ids)].copy()
    filtered_corr = correlations[correlations['source'].isin(ids) & correlations['target'].isin(ids)].copy()
    filtered_matrix = filtered_corr.pivot(index='source', columns='target', values=metric).fillna(1.0)
    filtered_distance_matrix = 1 - filtered_matrix

    Z = linkage(squareform(filtered_distance_matrix), method='complete')

    if nb_clusters is not None:
        clusters = fcluster(Z, nb_clusters, criterion='maxclust')
    else:
        clusters = fcluster(Z, t=threshold, criterion='distance')

    matrix_df = pd.DataFrame(filtered_distance_matrix)
    matrix_df['cluster'] = clusters
    cluster_centroids = matrix_df.groupby('cluster').mean().mean(axis=1)
    sorted_clusters = cluster_centroids.sort_values().index
    consistent_label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(sorted_clusters)}
    consistent_clusters = matrix_df['cluster'].map(consistent_label_mapping)

    mapping = pd.DataFrame({'config': filtered_distance_matrix.index, 'cluster': consistent_clusters})
    for id, row in mapping.iterrows():
        config = row['config']
        cluster = row['cluster']
        filtered_ds.loc[filtered_ds['id'] == config, 'cluster'] = int(cluster)

    filtered_ds = filtered_ds.sort_values(by='id', ascending=False)
    return filtered_ds, filtered_matrix, Z, consistent_clusters


def predict_clusters(dataset, correlations, nb_clusters, corr_func, threshold=0.2, train_size=0.7):
    results = dict()
    results['train_size'] = train_size
    ignored = [col for col in dataset.columns if col.startswith('from_')]
    ignored.extend(['id', 'cluster'])

    # Cluster the entire dataset once
    if nb_clusters is not None:
        ds, matrix, Z, clusters = clusterize_hierch(dataset, correlations, dataset['id'].tolist(), nb_clusters, corr_func)
    else:
        ds, matrix, Z, clusters = clusterize_hierch(dataset, correlations, dataset['id'].tolist(), None, corr_func, threshold)

    # Split into train and test sets after clustering
    X = ds.drop(columns=ignored)
    y = ds['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Train the classifier
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Use all unique clusters for target_names
    all_classes = sorted(set(y_test.unique()).union(set(np.unique(y_pred))))
    target_names = [str(name) for name in all_classes]

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    results['accuracy'] = report['accuracy']
    results['recall'] = report['macro avg']['recall']

    # Decision tree visualization
    results['decision_tree'] = tree.export_graphviz(classifier, out_file=None,
                                                    feature_names=X_train.columns.values,
                                                    filled=True, rounded=True,
                                                    special_characters=True,
                                                    leaves_parallel=True, proportion=True)

    # Feature importances
    features = X_train.columns
    importances = classifier.feature_importances_
    feat_importances = {feat: imp for feat, imp in zip(features, importances) if imp > 0.0}
    results['feature_importances'] = feat_importances

    return results



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

def get_antimedoids(distance_matrix: pd.DataFrame, clusters: pd.Series):
    antimedoids = {}
    unique_clusters = clusters.unique()
    for cluster in unique_clusters:
        cluster_indices = clusters[clusters == cluster].index
        cluster_distance_matrix = distance_matrix.loc[cluster_indices, cluster_indices]
        sum_distances = cluster_distance_matrix.sum(axis=1)
        antimedoid_id = sum_distances.idxmax()
        antimedoids[cluster] = antimedoid_id
    return antimedoids

