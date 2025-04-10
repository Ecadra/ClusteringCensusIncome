from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram 
import matplotlib.pyplot as plt

##Este script genera y muestra el dendograma

def plot_dendogram(model, **kwargs):#Función para crear una matriz de unión y plotear el dendograma
    
    counts = np.zeros(model.children_.shape[0])#hacer el conteo de muestras de cada nodo (esto dice en la documentación de scikit-learn. Dejo el link abajo)
    #https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    n_samples= len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count=0                 
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx- n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)


df = pd.read_csv('./Data/extracted_data/census-normalized.csv',header=0)#*Carga del dataset
#print(df)
                       

clustering = AgglomerativeClustering(metric='euclidean', linkage='complete', distance_threshold=0, n_clusters=None, compute_full_tree=True) #utilizando el objeto de sklearn AgglomerativeClustering

#usar distance_threshold para asegurarse de que se creara el árbol completo


clustering =clustering.fit(df)

plt.title("Hierarchical Clusterin Dendrogram")

plot_dendogram(clustering, truncate_mode="level", p=3)
plt.show()