import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#Este script es el tetativo para la demostración de 
#Cargar el dataset
df = pd.read_csv('./Data/dataset.csv')

#print(df)

clustering = AgglomerativeClustering(metric='euclidean', linkage='single', n_clusters=2) #utilizando el objeto de sklearn AgglomerativeClustering
clustering.fit(df)

#Para explicar las limitaciones del modelo de relación 'single
#Reducir la dimensionalidad a 2D usando PCA para la visualización (posiblimente es lo que oblique al 
# argoritmo a generar unicamente 2 clusters, ya lo proble con n_components = 4 y aún así no toma en cuenta la configuración del objeto. Posiblemente es el tipo de linkage)
pca=PCA(n_components=2)
df_reduced = pca.fit_transform(df)


#Plotear los clusters en un espacio reducido
plt.figure(figsize=(8,6))
plt.scatter(df_reduced[:, 0], df_reduced[:, 1], c=clustering.labels_, cmap='viridis')
plt.title("Clusters formados")
plt.show()

