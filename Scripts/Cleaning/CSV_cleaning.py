#Este archivo genera el csv: census_income_kdd_filtrado.csv
import pandas as pd

# Cargar el CSV
df = pd.read_csv('./Data/CSVs/census_income_kdd.csv')

# Total de filas
total_filas = len(df)

# Conteo del valor específico
conteo_excluido = (df['AMJIND'] == " Not in universe or children").sum()

# Resta: filas con valores distintos a "Not in universe or children"
resta = total_filas - conteo_excluido

print(f"Total de filas: {total_filas}")
print(f"Ocurrencias de 'Not in universe or children': {conteo_excluido}")
print(f"Filas restantes (otros valores): {resta}")

# Filtrar el DataFrame
df_filtrado = df[df['AMJIND'] != " Not in universe or children"]

# Binarizar income: ' - 50000' -> 0, ' 50000+.' -> 1
df_filtrado['income'] = df_filtrado['income'].apply(lambda x: 1 if str(x).strip() == '50000+.' else 0)

# Calcular proporciones
porcentaje_menor_50k = (df_filtrado['income'] == 0).mean() * 100
porcentaje_mayor_50k = (df_filtrado['income'] == 1).mean() * 100

print(f"Porcentaje de registros con ingreso <= 50,000: {porcentaje_menor_50k:.2f}%") #88.39%
print(f"Porcentaje de registros con ingreso > 50,000: {porcentaje_mayor_50k:.2f}%") #11.61%
# Proporciones
# 88.39% y 11.61%


#Ocurrencias de 'Not in universe or children': 100684
#Filas restantes (otros valores): 98839

# Filtrar y guardar nuevo CSV
df_filtrado.to_csv('./Data/CSVs/census_income_kdd_filtrado.csv', index=False)

print("Archivo filtrado guardado como 'census_income_kdd_filtrado.csv'")



