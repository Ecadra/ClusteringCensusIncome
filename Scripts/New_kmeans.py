from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from kmeans_procedures import Transformation_Loading_Process as tlp
from Indexes.calinsky import get_ch_score
from Indexes.davies_bouldin import get_db_score
from Indexes.Dunn import calcDunnIndex

# Tabla resumen
table = PrettyTable()
table.field_names = ["K","Inercia","Iteraciones realizadas","Distancia hasta el centro", "CH Score","Dunn Score","DB Score"]

# Cargar y preparar datos
df_normalized = pd.read_csv("./Data/CSVs/census_income_30k_transformed.csv")
column_names = tlp.read_names("./Data/extracted_data/tarFile/census-income.names")
column_names.remove("income")
column_names.remove("instance weight")
df_normalized.columns = column_names

# Ciclo por valores de K
for k in range(2, 5):
    X = df_normalized.copy()

    # Inicializar y ajustar KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    print(f"Calculando el índice de Dunn con {k} clusters...")
    ch_score=get_ch_score(X,labels)
    dunn_score = calcDunnIndex(X,labels)
    db_score=get_db_score(X,labels)
    # Calcular distancia media hasta el centro más cercano
    distance = np.mean(np.min(kmeans.transform(X), axis=1))

    # Calcular Calinski-Harabasz score
    ch_score = calinski_harabasz_score(X, labels)

    # Crear DataFrame con clusters SOLO para graficar
    df_plot = df_normalized.copy()
    df_plot['cluster'] = labels

    # Tabla de distribución
    cluster_counts = df_plot['cluster'].value_counts().sort_index()
    cluster_table = PrettyTable()
    cluster_table.field_names = ["Cluster", "N° de objetos"]
    for cluster, count in cluster_counts.items():
        cluster_table.add_row([cluster, count])

    print(f"\nDistribución de objetos para k={k}:")
    print(cluster_table)
    # Agregar a tabla
    table.add_row([
        k,
        round(kmeans.inertia_, 2),
        kmeans.n_iter_,
        round(distance, 2),
        round(ch_score, 2),
        round(dunn_score,2),
        round(db_score,10)
    ])

# Mostrar tabla resumen
print("\nResultados del clustering:")
print(table)

#El agrupamiento más significativo se da en k=5, ya que el CH Score es el más alto, lo que indica que los clusters 
# son más compactos y separados entre sí. Además la inercia es baja, lo que indica que los puntos 
# dentro de cada cluster están cerca del centroide.
#El agregar un cluster más despues de k=5 puede significar sobrejuste

"""

Resultados del clustering:
+---+----------+------------------------+---------------------------+----------+
| K | Inercia  | Iteraciones realizadas | Distancia hasta el centro | CH Score |
+---+----------+------------------------+---------------------------+----------+
| 2 | 57267.76 |           3            |            1.35           | 7476.36  |
| 3 | 53068.4  |           7            |            1.29           |  5220.7  |
| 4 | 48954.52 |           8            |            1.23           | 4613.06  |
+---+----------+------------------------+---------------------------+----------+
"""