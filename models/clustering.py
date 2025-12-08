"""
Clustering Models Module

This module contains implementations of various clustering algorithms
used in the ML Analysis Dashboard.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_kmeans_clustering(X):
    """
    Train a K-Means clustering model and return results.
    
    Args:
        X: Features for clustering
        
    Returns:
        dict: Clustering results (labels, silhouette score)
    """
    if len(X) < 3:
        raise ValueError("Dataset must have at least 3 samples for KMeans clustering")
        
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    
    return {
        'labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
        'silhouette_score': float(score)
    }