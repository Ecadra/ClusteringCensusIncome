import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
from prettytable import PrettyTable

# Dataset cargado
df = pd.read_csv('./Data/CSVs/census_income_30k_transformed.csv')
X = df.values  # Convertir DataFrame a matriz NumPy

# Configuraciones de DBSCAN a evaluar
configuraciones = [
    {"eps": 1.27, "min_samples": 17},
    {"eps": 1.27, "min_samples": 16},
    {"eps": 1.00, "min_samples": 20},
]

# Tabla para resultados
tabla = PrettyTable()
tabla.field_names = ["Epsilon", "MinPts", "Clusters", "CH Score"]

# Evaluación
for config in configuraciones:
    labels = DBSCAN(eps=config["eps"], min_samples=config["min_samples"]).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters >= 2:
        ch_score = calinski_harabasz_score(X[labels != -1], labels[labels != -1])
        ch_score = round(ch_score, 2)
    else:
        ch_score = "No aplica"

    tabla.add_row([
        config["eps"],
        config["min_samples"],
        n_clusters,
        ch_score
    ])
 
# Resultados 
print("\nResultados Calinski-Harabasz para DBSCAN:")
print(tabla)

"""
Resultados Calinski-Harabasz para DBSCAN:
+---------+--------+----------+----------+
| Epsilon | MinPts | Clusters | CH Score |
+---------+--------+----------+----------+
|   1.27  |   17   |    2     | 7527.84  |
|   1.27  |   16   |    2     | 7526.48  |
|   1.0   |   20   |    9     | 1117.58  |
+---------+--------+----------+----------+
"""