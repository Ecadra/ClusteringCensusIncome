import numpy as np

def findDistance(a, b):
    # Usa distancia euclidiana
    return np.linalg.norm(np.array(a) - np.array(b))

def calcDunnIndex(points, labels):
    """
    Calcula el índice de Dunn.
    
    Args:
        points (list or np.array): Puntos de datos.
        labels (list or np.array): Etiquetas de cluster para cada punto.

    Returns:
        float: Valor del índice de Dunn.
    """
    points = np.array(points)
    labels = np.array(labels)
    unique_clusters = np.unique(labels)

    # 1. Calcular mínima distancia entre clusters (inter-cluster)
    min_intercluster_dist = float('inf')
    for i, ci in enumerate(unique_clusters):
        for j, cj in enumerate(unique_clusters):
            if i >= j:
                continue
            points_ci = points[labels == ci]
            points_cj = points[labels == cj]
            for p1 in points_ci:
                for p2 in points_cj:
                    d = findDistance(p1, p2)
                    min_intercluster_dist = min(min_intercluster_dist, d)

    # 2. Calcular máxima distancia dentro de clusters (intra-cluster)
    max_intracluster_dist = 0
    for c in unique_clusters:
        cluster_points = points[labels == c]
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                d = findDistance(cluster_points[i], cluster_points[j])
                max_intracluster_dist = max(max_intracluster_dist, d)

    if max_intracluster_dist == 0:
        return float('inf')  # Evita división por cero si los clusters son idénticos o de un solo punto

    return min_intercluster_dist / max_intracluster_dist

if __name__ == "__main__":
    points = [
        [1, 2], [2, 2], [1, 3],   # Cluster 0
        [8, 8], [9, 8], [8, 9],   # Cluster 1
        [5, 1], [6, 1]            # Cluster 2
    ]

    labels = [
        0, 0, 0,   # Primeros tres puntos pertenecen al cluster 0
        1, 1, 1,   # Siguientes tres puntos al cluster 1
        2, 2       # Últimos dos puntos al cluster 2
    ]
    dunn = calcDunnIndex(points, labels)
    print(f"Dunn Index: {dunn:.3f}")
