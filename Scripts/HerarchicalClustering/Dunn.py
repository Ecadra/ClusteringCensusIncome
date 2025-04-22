import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist

def calcDunnIndex(points, labels):
    """
    Calcula el índice de Dunn con barra de progreso detallada.

    Args:
        points (list or np.ndarray): Lista o array de puntos (n_samples, n_features).
        labels (list or np.ndarray): Lista o array de etiquetas de cluster.

    Returns:
        float: Índice de Dunn.
    """
    points = np.array(points)
    labels = np.array(labels)
    unique_clusters = np.unique(labels)

    # Calcular total de comparaciones inter-cluster
    total_inter = sum(
        len(points[labels == ci]) * len(points[labels == cj])
        for i, ci in enumerate(unique_clusters)
        for cj in unique_clusters[i + 1:]
    )

    min_intercluster_dist = float('inf')
    with tqdm(total=total_inter, desc="Distancias inter-cluster") as pbar:
        for i, ci in enumerate(unique_clusters):
            for cj in unique_clusters[i + 1:]:
                points_ci = points[labels == ci]
                points_cj = points[labels == cj]
                dists = cdist(points_ci, points_cj, metric='euclidean')
                min_intercluster_dist = min(min_intercluster_dist, np.min(dists))
                pbar.update(len(points_ci) * len(points_cj))

    # Calcular total de comparaciones intra-cluster
    total_intra = sum(
        len(points[labels == c]) * (len(points[labels == c]) - 1) // 2
        for c in unique_clusters
    )

    max_intracluster_dist = 0
    with tqdm(total=total_intra, desc="Distancias intra-cluster") as pbar:
        for c in unique_clusters:
            cluster_points = points[labels == c]
            if len(cluster_points) > 1:
                dists = pdist(cluster_points, metric='euclidean')
                max_intracluster_dist = max(max_intracluster_dist, np.max(dists))
                pbar.update(len(dists))

    return min_intercluster_dist / max_intracluster_dist if max_intracluster_dist != 0 else float('inf')

# Ejemplo de uso
if __name__ == "__main__":
    points = [
        [1, 2], [2, 2], [1, 3],   # Cluster 0
        [8, 8], [9, 8], [8, 9],   # Cluster 1
        [5, 1], [6, 1]            # Cluster 2
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2]

    dunn = calcDunnIndex(points, labels)
    print(f"Dunn Index: {dunn:.3f}")
