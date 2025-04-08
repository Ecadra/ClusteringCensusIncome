import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def read_names(path_names:str) -> list[str]:
    """Lee de manera automática el nombre de las columnas desde el diccionario de datos

    Args:
        path_names (str): Ruta al archivo .names del dataset

    Returns:
        list[str]: lista con los nombres de las diferentes columnas del dataset, legibles y entendibles.
    """
    with open(path_names,'r') as file:
        read = file.readlines()
        column_names:list[str] = []#Se inicializa la lista para guardar los nombres
        for line in read:
            if line.startswith('|') or line.startswith('\n') :
                #ignorar comentarios y nuevas lineas que contiene el archivo
                continue
            else:
                if ":" in line:
                    column_parts = line.split(':')
                    column_names.append(column_parts[0].strip()) #La estructura del archivo define que el nombre va siempre a la izquierda, por lo que se guarda esa parte
    return column_names

def filter_dataset(raw_csv_input:str,original_column_names:list[str],filtered_csv_output_path:str,filtered_column_names:list[str]) -> pd.DataFrame:
    """Carga un dataset y lo filtra con base en la lista de columnas recibida

    Args:
        raw_csv_input (str): Ruta al archivo csv sin procesar
        original_column_names (list[str]): Lista de nombres de columnas del dataset
        filtered_csv_output_path (str): Ruta donde se guardará el archivo csv filtrado
        filtered_column_names (list[str]): Lista de las columnas que se quieren selecionar

    Returns:
        pd.DataFrame: Data Frame de pandas con el dataset cargado y formateado
    """
    if os.path.exists(filtered_csv_output_path):
        os.remove(filtered_csv_output_path)  # Eliminar el archivo existente
    #* Creacion del df original con todos los datos
    data:pd.DataFrame = pd.read_csv(raw_csv_input, 
                            header=None, 
                            sep = ",",
                            engine = 'python',
                            na_values=[' ?',""," Not in universe"," Not in universe or children"," Not in universe under 1 year old"])
    data.columns = original_column_names
    #Filtrado de las columnas que nos interesan
    data = data[filtered_column_names]
    data = data.drop_duplicates()
    data = data[data['age']>=21]
    data = data.dropna()
    data.to_csv(filtered_csv_output_path, index=False)
    return data

def normalize_dataset(filtered_df:pd.DataFrame, output_path:str) -> pd.DataFrame:
    """Recibe un dataframe filtrado pero no normalizado y lo guarda como csv para futuras referencias

    Args:
        filtered_df (pd.DataFrame): Data Frame de pandas que tienen la información seleccionada
        output_path (str): ruta donde se guardará el archivo

    Returns:
        pd.DataFrame: Data Frame de pandas normalizado con minmax
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    le = LabelEncoder()
    scaler = MinMaxScaler()
    categorical_columns:pd.Index[str] = filtered_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        filtered_df[col] = le.fit_transform(filtered_df[col])
    numeric_columns:pd.Index[str] = filtered_df.select_dtypes(include=['int64','float64']).columns
    filtered_df[numeric_columns] = scaler.fit_transform(filtered_df[numeric_columns])
    filtered_df.to_csv(output_path,index = False)
    return filtered_df

path_data = "./Data/extracted_data/tarFile/census-income.data"
path_names = "./Data/extracted_data/tarFile/census-income.names"
path_test = "./Data/extracted_data/tarFile/census-income.test"

path_filtered = "./Data/extracted_data/census-filtered.csv"
selected_columns =["instance weight","age","class of worker","education","major industry code","major occupation code","race","sex","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed","yearly income"]

df = filter_dataset(path_data,read_names(path_names),path_filtered,selected_columns)

print(df.head())

#Comprobación de que ya no hay nulos:
print(f'El total de valores nulos por columna es: {df.isnull().sum()}')

path_normalized_df = "./Data/extracted_data/census-normalized.csv"
df = df.sample(n=20000, weights="instance weight",random_state=42)
df = df.drop(columns=["instance weight"])
df = normalize_dataset(df,path_normalized_df)
print(df.head())
print(df.describe)

print(df.dtypes)
