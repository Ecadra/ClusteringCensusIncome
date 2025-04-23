#Este archivo genera el csv: census_income_30k.csv
import pandas as pd

# Cargar dataset original
ruta = './Data/CSVs/census_income_kdd_filtrado.csv'
df = pd.read_csv(ruta)

# Eliminar registros duplicados
df = df.drop_duplicates()

# Confirmar proporciones
print("Distribución original tras eliminar duplicados:")
print(df['income'].value_counts(normalize=True) * 100)

# Calcular tamaños de muestra por clase
n_total = 30000
n_income_0 = round(n_total * 0.8839)  # ≈ 26,517
n_income_1 = n_total - n_income_0     # ≈ 3,483

# Tomar muestras estratificadas
df_0 = df[df['income'] == 0].sample(n=n_income_0, random_state=42)
df_1 = df[df['income'] == 1].sample(n=n_income_1, random_state=42)

# Combinar
df_sampled = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Verificar nueva proporciónprint("Distribución en muestra de 30,000:")
print("Distribución en muestra de 30,000:")
print(df_sampled['income'].value_counts(normalize=True) * 100)

# Guardar nuevo dataset reducido
df_sampled.to_csv('./Data/CSVs/census_income_30k.csv', index=False)
print("Archivo guardado como 'census_income_30k.csv'")