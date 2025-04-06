import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def get_df():
    
    # Cargar el CSV (ruta robusta para evitar problemas) ----
    base_dir = os.path.dirname(__file__)
    ruta_csv = os.path.join(base_dir, '..', 'Data', 'census_income_kdd_filtrado.csv')
    ruta_csv = os.path.abspath(ruta_csv)

    df_raw = pd.read_csv(ruta_csv)

    # Seleccionar columnas relevantes (basadas en el objetivo del proyecto: 13)
    columnas = ['AAGE', 'ACLSWKR', 'AHGA', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 
                'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'income']
    df_raw = df_raw[columnas]

    # Obtener duplicados antes del tratamiento (solamente para visualizaciónm, no relevante para el procesamiento)
    df_duplicados_raw = df_raw[df_raw.duplicated(keep=False)].copy()

    # LIMPIEZA DE DATOS -------------------------------------
        # -> Eliminar filas con valores nulos y filas duplicadas
    df_clean = df_raw.dropna().drop_duplicates().copy()

    # TRANSFORMACIÓN DE DATOS -------------------------------
        # -> Binarizar columna 'income': 1 si el ingreso es mayor o igual a 50k, 0 si es menor
    df_clean['income'] = df_clean['income'].apply(lambda x: 1 if str(x).strip() == '50000+.' else 0)

        # -> Codificar variables categóricas (10)
    label_cols = ['ACLSWKR', 'AHGA', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 
                'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP']

        # Se utiliza LabelEncoder para transformar cada categoría en un número entero único
    for col in label_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

        # -> Escalar los datos
        # La escala de las variables numéricas puede variar, MinMaxScaler normaliza las características para que estén entre 0 y 1, 
        # ayuda aque todas las características tengam el mismo rango de influencia en el clustering.
    scaler = MinMaxScaler()

        # Aplicar el escalado a los datos limpios
    df_scaled_array = scaler.fit_transform(df_clean)

        # Convertir el arreglo escalado de nuevo en un dataFrame
    df_final = pd.DataFrame(df_scaled_array, columns=df_clean.columns)
    # --------------------------------------------------------
    # Obtener duplicados después del tratamiento (solamente para visualización, no relevante para el procesamiento)
    df_duplicados_final = df_final[df_final.duplicated(keep=False)].copy()

    # Información final
    print(f"Shape inicial: {df_raw.shape}")
    print(df_raw.head())
    print("Duplicados antes del tratamiento:", len(df_duplicados_raw))

    print(f"Shape final: {df_final.shape}")
    print(df_final.head())
    print("Duplicados después del tratamiento:", len(df_duplicados_final))

    return df_final