from sklearn.metrics import davies_bouldin_score
import numpy as np

def get_db_score(X, labels):
    valid_labels = labels != -1
    X_valid = X[valid_labels]  # Usar los datos originales, no el objeto DBSCAN
    labels_valid = labels[valid_labels]
    
    if len(np.unique(labels_valid)) > 1:  # Requiere al menos 2 clusters
        dbi = davies_bouldin_score(X_valid, labels_valid)
        # print(f"Índice de Davies-Bouldin: {dbi:.4f}")
    else:
        dbi='No aplica'
    return dbi