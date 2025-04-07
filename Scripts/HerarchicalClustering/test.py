

#/////////////////////////////////Script de la documentación//////////////////////////////////////////

#Este script demo que aparece dentro de la página de la docuemtnación  usa una interfaz gráfica para mostrar el dendograma

'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data
#print(X)


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
'''


#////////////////////////////////////////////////////**********Demo1********************//////////////////////////////////////////////////////////////

#Implementación del script de la documentación con el csv de nuetro problema

'''
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram 
import matplotlib.pyplot as plt


def plot_dendogram(model, **kwargs):#Función para crear una matriz de unión y plotear el dendograma
    counts = np.zeros(model.children_.shape[0])#hacer el conteo de muestras de cada nodo (esto dice en la documentación de scikit-learn. Dejo el link abajo)
    #https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    n_samples= len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count=0                 #hay que estudiar este bloque de código
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx- n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)


df = pd.read_csv('./Data/dataset.csv')

#print(df)

clustering = AgglomerativeClustering(metric='euclidean', linkage='single', distance_threshold=0, n_clusters=None) #utilizando el objeto de sklearn AgglomerativeClustering

#usar distance_threshold para asegurarse de que se creara el árbol completo


clustering =clustering.fit(df)

plt.title("Hierarchical Clusterin Dendogram")

plot_dendogram(clustering, truncate_mode="level", p=3)
plt.show()
'''


#///////////////////////////////////////////////Test de un video tutorial////////////////////////////////
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


dataset= make_blobs(n_samples=200,
                    n_features=2,
                    centers=4,
                    cluster_std= 1.6,
                    random_state=50)

points = dataset[0]

dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
print(dendrogram)
'''


#Otro ejemplo de visualización de 
