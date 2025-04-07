import os
from numpy import size
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def leer_nombres(ruta_nombres):

    with open(ruta_nombres,'r') as file:
        lectura = file.readlines()
        column_names = []
        for line in lectura:
            if line.startswith('|') or line.startswith('\n') :
                #ignorar comentarios y nuevas lineas
                continue
            else:
                if ":" in line:
                    column_parts = line.split(':')
                    column_names.append(column_parts[0].strip())
    return column_names

def filtrar_dataset(ruta_data,ruta_csv,nombres_columnas,columnas_deseadas):
    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv,header = 0,sep = ",")
    else:
        nombres_columnas = leer_nombres(ruta_name)
        data = pd.read_csv(ruta_data, 
                                header=None, 
                                sep = ",",
                                engine = 'python',
                                na_values=[' ?',""," Not in universe"," Not in universe or children"," Not in universe under 1 year old"])
        data.columns = nombres_columnas
        #Filtrado de las columnas que nos interesan
        data = data[columnas_deseadas]
        data = data.drop_duplicates()
        data = data.dropna()
        data.to_csv(ruta_csv,index=False)
    return data

def normalizar_dataset(data_frame, ruta_salida):
    if os.path.exists(ruta_salida):
        return pd.read_csv(ruta_salida,header = 0,sep = ",")
    else:
        numeric_columns = data_frame.select_dtypes(include=['int64','float64']).columns
        categorical_columns = data_frame.select_dtypes(include=['object']).columns
        #scaler = StandardScaler()
        #StandardScaler utiliza la media y la desviación estándar, por lo que la normalización puede involucrar numeros negativos
        #? Para K-means, el adecuado a utilizar es MinMaxScaler, porque simplifica el procesado de clusterización.
        scaler = MinMaxScaler()
        data_frame[numeric_columns] = scaler.fit_transform(data_frame[numeric_columns])
        le = LabelEncoder()
        for col in categorical_columns:
            data_frame[col] = le.fit_transform(data_frame[col])

        data_frame.to_csv(ruta_salida,index = False)
    return data_frame

ruta_data = "./Data/extracted_data/tarFile/census-income.data"
ruta_name = "./Data/extracted_data/tarFile/census-income.names"
ruta_test = "./Data/extracted_data/tarFile/census-income.test"

ruta_csv_filtrado = "./Data/extracted_data/census-filtered.csv"
lista_columnas_deseadas =["age","class of worker","education","major industry code","major occupation code","race","sex","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed","yearly income"]

data_frame = filtrar_dataset(ruta_data,ruta_csv_filtrado,leer_nombres(ruta_name),lista_columnas_deseadas)

print(data_frame.head())

#Comprobación de que ya no hay nulos:
print(f'El total de valores nulos por columna es: {data_frame.isnull().sum()}')

ruta_dataset_normalizado = "./Data/extracted_data/census-normalized.csv"
data_frame = normalizar_dataset(data_frame,ruta_dataset_normalizado)

print(data_frame.head())
print(data_frame.describe)


