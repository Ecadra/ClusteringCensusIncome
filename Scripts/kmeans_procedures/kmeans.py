from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Seleccionar el número de clusters (por ejemplo, k=3)
k = 3

df_normalized = pd.read_csv("./Data/extracted_data/census-normalized.csv")

# Para guardar la inercia (suma de distancias al centro de cada clúster)
inertia = []

# Probar entre 1 y 5 clústeres, con 10 iteraciones, modificando el centroide(s)
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    print(kmeans.fit(df_normalized))
    inertia.append(kmeans.inertia_)

# Gráfica del método del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), inertia, marker='o', linestyle='-', color='blue')
plt.title('Método del codo para elegir k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.xticks(range(1, 6))
plt.grid(True)
plt.show()
