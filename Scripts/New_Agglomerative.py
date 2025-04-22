
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from prettytable import PrettyTable

from Indexes.calinsky import get_ch_score
from Indexes.davies_bouldin import get_db_score
from Indexes.Dunn import calcDunnIndex
#Cargar el dataset
df = pd.read_csv('./Data/CSVs/census_income_30k_transformed.csv')

"""
#Solo se toma una muestra de 300 filas para el dendrograma
# Esto es para evitar problemas de memoria y tiempo de ejecución
df_sample = df.sample(n=300, random_state=42)

#Se genera el dendrograma (fines de visualización)
# Se utiliza el método de Ward para minimizar la varianza intra-cluster

linked = linkage(df_sample, method='ward')  

plt.figure(figsize=(15, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Dendrograma jerárquico (sample de 300)")
plt.xlabel("Observaciones")
plt.ylabel("Distancia (varianza)")
plt.grid(True)
#plt.show()
"""
# 4. Evaluación de diferentes números de clusters
table = PrettyTable()
table.field_names = ["Número de Clusters (k)", "Calinski-Harabasz Score","Dunn Score","Davies-Bouldin Score"]

for k in range(2, 5):
    model = AgglomerativeClustering(metric='euclidean', linkage='complete', n_clusters=k)
    labels = model.fit_predict(df)
    ch_score = get_ch_score(df, labels)
    dunn_score = calcDunnIndex(df,labels)
    db_score=get_db_score(df,labels)
    table.add_row([k, round(ch_score, 2),round(dunn_score,2),round(db_score,2)])

    # Asignar clusters temporalmente
    df_temp = df.copy()
    df_temp['cluster'] = labels
    # Tabla de distribución

    cluster_counts = df_temp['cluster'].value_counts().sort_index()
    cluster_table = PrettyTable()
    cluster_table.field_names = ["Cluster", "N° de objetos"]
    for cluster, count in cluster_counts.items():
        cluster_table.add_row([cluster, count])
    
    print(f"\nDistribución de objetos para k={k}:")
    print(cluster_table)

print("\nResultados del índice de Calinski-Harabasz para distintos k:")
print(table)

"""
# 5. Aplicar clustering final con k=2, ya que es el que da mejor score
k_final = 2
model = AgglomerativeClustering(metric='euclidean', linkage='complete', n_clusters=k_final)
labels = model.fit_predict(df)
df['cluster'] = labels
"""
"""
# Comprobar la calidad del clustering
Z = linkage(df.drop(columns=['cluster']), method='complete', metric='euclidean')
from scipy.cluster.hierarchy import fcluster

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['cluster'] = fcluster(Z, t=2, criterion='maxclust')  # K=2

plt.figure(figsize=(10,6))
for c in np.unique(df_pca['cluster']):
    plt.scatter(df_pca[df_pca['cluster'] == c]['PC1'],
                df_pca[df_pca['cluster'] == c]['PC2'],
                label=f"Cluster {c}",
                alpha=0.6)
plt.legend()
plt.title("Visualización de clusters para K=2")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()

labels_k2 = fcluster(Z, t=2, criterion='maxclust')
unique, counts = np.unique(labels_k2, return_counts=True)
print(dict(zip(unique, counts)))
"""