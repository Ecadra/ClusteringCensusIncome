import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


#Este script es el tetativo para la demostración de 
#Cargar el dataset
df = pd.read_csv('./Data/extracted_data/census-normalized.csv',header=0)#*Carga del dataset

#print(df)

clustering = AgglomerativeClustering(metric='euclidean', linkage='ward', n_clusters=2) #utilizando el objeto de sklearn AgglomerativeClustering
labels = clustering.fit_predict(df)

#Usar el metodo linkage para mostrar como se combinaron los puntos

print("Asinación de clusters: ", labels)

