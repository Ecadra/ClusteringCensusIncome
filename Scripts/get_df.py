import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_df():
    # Cargar el CSV
    df = pd.read_csv('./Data/census_income_kdd_filtrado.csv')

    # Contar registros (filas)
    total_registros = len(df)
    print(f"Total de registros en el dataset: {total_registros}")
    print("------------------------------------------------------")

    # Contar cuántas veces aparece " Not in universe or children" en la columna AMJIND
    conteo = (df['AMJIND'] == " Not in universe or children").sum()
    print(f"Número de ocurrencias de 'Not in universe or children' en AMJIND (Total de personas desempleadas mayores de 21 años): {conteo}")
    print("------------------------------------------------------")

    conteoCincoMil = (df['income'] == " 50000+.").sum()
    print(f"Total de personas que ganan más de 5000: {conteoCincoMil}")
    print("------------------------------------------------------")

    columnas = ['AAGE', 'ACLSWKR', 'AHGA', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 
                'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'income']

    # Reinstancia del df, para que solo contenga las columnas deseadas
    df = df[columnas]
    # Comprobar que las columnas del df fueron las definidas
    print(df.columns)
    # Comprobar que no se alteró el # de registros
    new_length = len(df)
    print(f"Número de registros después de la selección de columnas: {new_length}")
    print("------------------------------------------------------")

    """
    Variables a vectorizar (Son categóricas)
    ACLSWKR, AHGA, AMJIND, AMJOCC, ARACE, ASEX, PEFNTVTY, PEMNTVTY, PENATVTY, PRCITSHP
    """

    # Binarizar la variable 'income'
    df['income'] = df['income'].apply(lambda x: 1 if str(x).strip() == '50000+.' else 0)

    # Verificar valores únicos después de binarizar
    print("Valores únicos en 'income' después de binarizar:", df['income'].unique())

    nuevaCuentaCincoMil = (df['income'] == 1).sum()
    print(f"Comprobacion de que la cuenta de más de 5000 es la misma después de ser binarizada: {nuevaCuentaCincoMil}")
    print("------------------------------------------------------")

    # Definir variables
    variables_categoricas = ['ACLSWKR', 'AHGA', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX',
                             'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP']
    variables_numericas = ['AAGE', 'SEOTR', 'income']

    # One-Hot Encoding para categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categoricas = encoder.fit_transform(df[variables_categoricas])
    df_categoricas = pd.DataFrame(df_categoricas, columns=encoder.get_feature_names_out(variables_categoricas))

    # Escalar variables numéricas (incluyendo income ya binarizado)
    scaler = StandardScaler()
    df_numericas = scaler.fit_transform(df[variables_numericas])
    df_numericas = pd.DataFrame(df_numericas, columns=variables_numericas)

    # Concatenar
    df = pd.concat([df_categoricas, df_numericas], axis=1)

    # Comprobaciones finales
    print(f"Shape del DataFrame transformado: {df.shape}")
    print(df.head())

    return df