#Este archivo genera el csv: census_income_kdd.csv

from ucimlrepo import fetch_ucirepo
import pandas as pd

# Descargar el dataset desde la UCI ML Repo
census_income_kdd = fetch_ucirepo(id=117)

# Separar características y etiquetas
X = census_income_kdd.data.features
y = census_income_kdd.data.targets

# Unir en un solo DataFrame
df = pd.concat([X, y], axis=1)

# Guardar en archivo CSV
df.to_csv('./Data/CSVs/census_income_kdd.csv', index=False)

print("Archivo CSV guardado exitosamente en './Data/CSVs/census_income_kdd.csv'")