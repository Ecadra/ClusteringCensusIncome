import pandas as pd
from sklearn.cluster import DBSCAN
import os

'''
# Cargar el CSV (ruta robusta para evitar problemas) ----
base_dir = os.path.dirname(__file__)
ruta_csv = os.path.join(
    base_dir, "..", "Data", "CSVs", "census_income_30k_transformed.csv"
)
ruta_csv = os.path.abspath(ruta_csv)
'''
ruta_csv='./Data/CSVs/census_income_30k_transformed.csv'

df = pd.read_csv(ruta_csv,header=0)

X= df.values  # Convertir el DataFrame a una matriz NumPy
eps = 1.27
min_samples = 17

clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
df['cluster'] = clustering.labels_

#print(df.head())

df.to_csv('./Data/CSVs/census_income_30k_transformed_DBSCAN.csv',index=False)