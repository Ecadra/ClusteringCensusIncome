from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import seaborn as sns
import Transformation_Loading_Process as tlp
import os
#Creacion de la tabla para consola con PrettyTable
table = PrettyTable()
table.field_names = ["K","Inercia","Iteraciones realizadas","Distancia hasta el centro"]

#Rutas a los archivos
df_normalized = pd.read_csv("./Data/CSVs/census_income_30k_transformed.csv")
column_names = tlp.read_names("./Data/extracted_data/tarFile/census-income.names")

#Se quitan de las columnas income e instance weight
column_names.remove("income")
column_names.remove("instance weight")
df_normalized.columns = column_names
#Para 1 a 6 clusters
for k in range(1, 7):
    #Se realiza una copia del dataset para no modificar el original con las tags
    X = df_normalized.copy()
    #Creacion del nombre del archivo
    plot_fig_path=f"./media/plt_figures/scatterplot_k={k}"
    #Creacion del objeto kmeans con k clusters y 10 iteraciones para generar clusters
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(X)
    #Se crean las tag de pertenencia a clusters
    df_normalized['cluster'] = kmeans.labels_ 
    
    # Crear una tabla para mostrar el número de objetos por cluster
    cluster_counts = df_normalized['cluster'].value_counts().sort_index()
    cluster_table = PrettyTable()
    cluster_table.field_names = ["Cluster", "Número de objetos"]
    for cluster, count in cluster_counts.items():
        cluster_table.add_row([cluster, count])
    
    print(f"\nNúmero de objetos por cluster para k={k}:")
    print(cluster_table)

    plt.figure(figsize=(12,8))
    #Graficación del resultado
    palette = sns.color_palette("viridis",n_colors=df_normalized['cluster'].nunique())
    sns.scatterplot(
        data=df_normalized,
        x='weeks worked in year',
        y='detailed industry recode',
        hue='cluster',
        palette=palette,
        s=100,          # Tamaño de los puntos
        alpha=0.8,      # Transparencia
        edgecolor='w',  # Borde blanco para mejor visibilidad
        legend='full'
    )
    plt.title("Clustering con k-means graficado en edad e industria",fontsize=14,pad=20)
    plt.xlabel("Cantidad de semanas trabajadas en año",fontsize = 12)
    plt.ylabel("Industria normalizada",fontsize = 12)
    plt.grid(True,linestyle = '--', alpha = 0.5)
    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, df_normalized.columns.get_loc('weeks worked in year')],
        centers[:, df_normalized.columns.get_loc('detailed industry recode')],
        c='red',
        s=200,
        marker='X',
        label='Centros'
    )
    plt.legend(
        title = 'Cluster',
        title_fontsize = 12,
        fontsize = 11,
        loc = 'upper right',
        bbox_to_anchor = (1.25,1)
    )

    plt.tight_layout()
    distance = np.mean(np.min(kmeans.transform(X), axis=1))
    if os.path.exists(plot_fig_path):
        os.remove(plot_fig_path)
    plt.savefig(plot_fig_path)
    table.add_row([
        k,
        round(kmeans.inertia_, 2),
        kmeans.n_iter_,
        round(distance, 2)
    ])

print(table)
