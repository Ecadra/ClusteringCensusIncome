import tarfile
import os

ruta_archivo_tar = './Data/data_source/census.tar.gz'
ruta_extraccion = './Data/extracted_data/tarFile'

os.makedirs(ruta_extraccion, exist_ok = True)

with tarfile.open(ruta_archivo_tar, "r:gz") as tar:
    tar.extractall(path=ruta_extraccion)

print("Extracción completada")