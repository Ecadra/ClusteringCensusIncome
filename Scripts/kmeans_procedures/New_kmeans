from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import seaborn as sns
import Transformation_Loading_Process as tlp
import os

# Tabla resumen
table = PrettyTable()
table.field_names = ["K","Inercia","Iteraciones realizadas","Distancia hasta el centro", "CH Score"]

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

    """
    # Graficar
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", n_colors=k)
    sns.scatterplot(
        data=df_plot,
        x='weeks worked in year',
        y='detailed industry recode',
        hue='cluster',
        palette=palette,
        s=100,
        alpha=0.8,
        edgecolor='w',
        legend='full'
    )
    plt.title(f"K-Means clustering (k={k})", fontsize=14)
    plt.xlabel("Semanas trabajadas")
    plt.ylabel("Industria normalizada")
    plt.grid(True, linestyle='--', alpha=0.5)

    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, df_normalized.columns.get_loc('weeks worked in year')],
        centers[:, df_normalized.columns.get_loc('detailed industry recode')],
        c='red', s=200, marker='X', label='Centros'
    )

    plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.25,1))
    plt.tight_layout()

    # Guardar gráfica
    plot_fig_path = f"./media/plt_figures/scatterplot_k={k}.png"
    if os.path.exists(plot_fig_path):
        os.remove(plot_fig_path)
    plt.savefig(plot_fig_path)
    plt.close()
    """
    # Agregar a tabla
    table.add_row([
        k,
        round(kmeans.inertia_, 2),
        kmeans.n_iter_,
        round(distance, 2),
        round(ch_score, 2)
    ])

# Mostrar tabla resumen
print("\nResultados del clustering:")
print(table)


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