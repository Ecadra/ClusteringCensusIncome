import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
# from matplotlib import pyplot as plt
from kneed import KneeLocator
from preparacion_datos import get_df
from davies_bouldin import davies_bouldin
from tabulate import tabulate

# Cargar el CSV (ruta robusta para evitar problemas) ----
base_dir = os.path.dirname(__file__)
ruta_csv = os.path.join(base_dir, "..", "Data", "CSVs","census_income_30k_transformed.csv")
ruta_csv = os.path.abspath(ruta_csv)

data = pd.read_csv(ruta_csv)

# Usar los datos obtenidos para DBSCAN
X = data.values  # Convertir el DataFrame a una matriz NumPy

# 3. Visualización inicial de los datos (opcional)
# if X.shape[1] == 2:  # Verifica si las dimensiones son 2D para poder graficar
#     df = pd.DataFrame(X, columns=["x", "y"])
#     fig, ax = plt.subplots(figsize=(8, 8))
#     df.plot(ax=ax, kind="scatter", x="x", y="y")
#     plt.xlabel("X_1")
#     plt.ylabel("X_2")
#     plt.show()


# 4. Determinar el valor óptimo de eps usando el k-dist plot
def dbscan_clustering(eps, minPts):
    # minPts = 26  # Número mínimo de vecinos

    # Calcular las distancias a los vecinos más cercanos
    # neighbors = NearestNeighbors(n_neighbors=minPts)
    # neighbors_fit = neighbors.fit(X)
    # distances, indices = neighbors_fit.kneighbors(X)

    # Ordenar las distancias y graficar
    # distances = np.sort(distances[:, minPts - 1])
    # plt.figure(figsize=(8, 4))
    # plt.plot(distances)
    # plt.title(f'k-dist plot (k={minPts})')
    # plt.xlabel('Puntos ordenados')
    # plt.ylabel(f'Distancia al {minPts}-ésimo vecino')
    # plt.grid(True)
    # plt.show()

    # Encontrar el valor óptimo de eps usando la "rodilla" del gráfico
    # knee_locator = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    # eps_optimo = distances[knee_locator.knee]
    # eps_optimo =0.5
    # print(f"Valor sugerido para eps: {eps_optimo}")
    # print(f"Valor usado para min_samples (minPts): {minPts}")

    # Aplicar DBSCAN con el valor óptimo de eps y min_samples
    output = {}
    clustering = DBSCAN(eps=eps, min_samples=minPts).fit(X)
    labels = clustering.labels_

    # Imprimir resultados
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (
        1 if -1 in unique_labels else 0
    )  # Restar 1 si hay ruido (-1)
    n_noise = list(labels).count(-1)

    # davies_bouldin(X, labels)

    # print(f"Cantidad de clusters generados: {n_clusters}")
    # print(f"Número de objetos por cluster:")
    output["epsilon"] = eps
    output["minPts"] = minPts
    output["n_clusters"] = n_clusters
    output["ruido"] = n_noise
    for cluster_id in unique_labels:
        if cluster_id != -1:  # No mostrar ruido
            # print(f"Cluster {cluster_id}: {list(labels).count(cluster_id)} objetos")
            output[f"cluster{cluster_id}"] = {list(labels).count(cluster_id)}

    # print(f"Objetos considerados ruido: {n_noise}")

    return output


datos = []
minpts=14
for i in range (1,30):
    datos.append(dbscan_clustering(1.28,minpts))
    minpts+=1
print('------Clustering con DBSCAN------')
print('------B  I  T  Á  C  O  R  A-----')
print(tabulate(datos, headers="keys", tablefmt="grid"))
# 7. Visualizar los resultados del clustering
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired', s=50)
# plt.title('Clustering con DBSCAN')
# plt.xlabel('Componente 1')
# plt.ylabel('Componente 2')
# plt.grid(True)
# plt.show()
