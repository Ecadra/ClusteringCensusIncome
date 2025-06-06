#Este archivo genera el csv: census_income_30k_transformed.csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Cargar dataset original
ruta = './Data/CSVs/census_income_30k.csv'

df = pd.read_csv(ruta)

total_registros = len(df)

sum_menor_50k = (df['income'] == 0).sum()
print(sum_menor_50k) #26,517
sum_mayor_50k = (df['income'] == 1).sum()
print(sum_mayor_50k) #3843

porcentaje_menor_50k = (df['income'] == 0).mean() * 100
porcentaje_mayor_50k = (df['income'] == 1).mean() * 100

print(f"El total de registros es de: {total_registros}")
print(f"Porcentaje de ingresos menor a 50,000: {porcentaje_menor_50k}")
print(f"Porcentaje de ingresos mayor a 50,000: {porcentaje_mayor_50k}")

# Detectar columnas categóricas (tipo 'object')
cat_cols = df.select_dtypes(include=['object']).columns

# Aplicar LabelEncoder a todas las columnas categóricas
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Detectar columnas numéricas (int y float)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Aplicar MinMaxScaler
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# Eliminar el instance weight (MARSUPWRT) y la columna de ingresos (income)
df = df.drop(columns=['income', 'MARSUPWRT'], errors='ignore')  # 'errors=ignore' evita error si alguna no existe

#Comprobar el resultado
print("\nDataFrame transformado:") 
print(df.head()) #imprime las primeras 5 filas
print("\nTipos de datos:") #imprime los tipos de datos 
print(df.dtypes)

# Guardar el DataFrame transformado
df.to_csv('./Data/CSVs/census_income_30k_transformed.csv', index=False)
print("Archivo transformado guardado como 'census_income_30k_transformed.csv'")


