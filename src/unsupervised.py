from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(X):
    try:
        if len(X) < 3:
            raise ValueError("Dataset must have at least 3 samples for KMeans clustering")
            
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Silhouette Score (KMeans): {score:.2f}")
        
        return {
            'labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
            'silhouette_score': float(score)
        }
    except Exception as e:
        print(f"Error in kmeans_clustering: {str(e)}")
        raise
