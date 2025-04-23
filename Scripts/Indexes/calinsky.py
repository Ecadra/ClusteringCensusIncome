import numpy as np
from sklearn.metrics import calinski_harabasz_score
def get_ch_score(X, labels):
    valid_labels = labels != -1
    X_valid = X[valid_labels]  # Usar los datos originales, no el objeto DBSCAN
    labels_valid = labels[valid_labels]
    
    if len(np.unique(labels_valid)) > 1:  # Requiere al menos 2 clusters
        dbi = calinski_harabasz_score(X_valid, labels_valid)
        # print(f"Índice de Calinski Harabasz: {dbi:.4f}")
    else:
        dbi='No aplica'
    return dbi