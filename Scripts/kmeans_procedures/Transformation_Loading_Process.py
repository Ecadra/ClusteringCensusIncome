import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def read_names(path_names:str) -> list[str]:   
    with open(path_names,'r') as file:
        read = file.readlines()
        column_names:list[str] = []
        for line in read:
            if line.startswith('|') or line.startswith('\n') :
                #ignorar comentarios y nuevas lineas
                continue
            else:
                if ":" in line:
                    column_parts = line.split(':')
                    column_names.append(column_parts[0].strip())
    return column_names

def filter_dataset(raw_csv_input:str,original_column_names:list[str],filtered_csv_output_path:str,filtered_column_names:list[str]) -> pd.DataFrame:
    if os.path.exists(filtered_csv_output_path):
        filtered_df:pd.DataFrame = pd.read_csv(filtered_csv_output_path,header = 0,sep = ",")
        return filtered_df
    else:
        original_column_names = read_names(path_names)
        data:pd.DataFrame = pd.read_csv(raw_csv_input, 
                                header=None, 
                                sep = ",",
                                engine = 'python',
                                na_values=[' ?',""," Not in universe"," Not in universe or children"," Not in universe under 1 year old"])
        data.columns = original_column_names
        #Filtrado de las columnas que nos interesan
        data = data[filtered_column_names]
        data = data.drop_duplicates()
        data = data.dropna()
        data.to_csv(filtered_csv_output_path,index=False)
    return data

def normalize_dataset(filtered_df:pd.DataFrame, output_path:str) -> pd.DataFrame:
    if os.path.exists(output_path):
        return pd.read_csv(output_path,header = 0,sep = ",")
    else:
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
selected_columns =["age","class of worker","education","major industry code","major occupation code","race","sex","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed","yearly income"]

df = filter_dataset(path_data,read_names(path_names),path_filtered,selected_columns)

print(df.head())

#Comprobación de que ya no hay nulos:
print(f'El total de valores nulos por columna es: {df.isnull().sum()}')

path_normalized_df = "./Data/extracted_data/census-normalized.csv"
df = normalize_dataset(df,path_normalized_df)

print(df.head())
print(df.describe)

print(df.dtypes)
