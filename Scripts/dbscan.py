import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tabulate import tabulate
from Indexes.davies_bouldin import get_db_score
from Indexes.calinsky import get_ch_score
from Indexes.Dunn import calcDunnIndex
# Cargar el CSV (ruta robusta para evitar problemas) ----
base_dir = os.path.dirname(__file__)
ruta_csv = os.path.join(base_dir, "..", "Data", "CSVs","census_income_30k_transformed.csv")
ruta_csv = os.path.abspath(ruta_csv)

data = pd.read_csv(ruta_csv)

# Usar los datos obtenidos para DBSCAN
X = data.values  # Convertir el DataFrame a una matriz NumPy


def dbscan_clustering(eps, minPts):
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

    db=get_db_score(X, labels)
    ch=get_ch_score(X,labels)
    dunn=calcDunnIndex(X[labels != -1],labels[labels != -1])
    output["epsilon"] = eps
    output["minPts"] = minPts
    output["n_clusters"] = n_clusters
    output["noise"] = n_noise
    output["CH score"]=ch
    output["Dunn score"]=dunn
    output["DB score"]=db
    for cluster_id in unique_labels:
        if cluster_id != -1:  # No mostrar ruido
            output[f"c{cluster_id}"] = {list(labels).count(cluster_id)}

    return output


datos = []
configuraciones = [
    {"eps": 1.27, "min_samples": 17},
    {"eps": 1.27, "min_samples": 16},
    {"eps": 1.00, "min_samples": 20},
]

for config in configuraciones:
    datos.append(dbscan_clustering(config.get('eps'),config.get('min_samples')))
print('------Clustering con DBSCAN------')
print('------B  I  T  Á  C  O  R  A-----')
print(tabulate(datos, headers="keys", tablefmt="grid"))