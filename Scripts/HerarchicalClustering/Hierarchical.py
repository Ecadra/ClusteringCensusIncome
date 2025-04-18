import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score


#Este script es el tetativo para la demostración de 
#Cargar el dataset
df = pd.read_csv('./Data/CSVs/census_income_30k_transformed.csv',header=0)#*Carga del dataset

#print(df)

clustering = AgglomerativeClustering(metric='euclidean', linkage='complete', n_clusters=2) #utilizando el objeto de sklearn AgglomerativeClustering
#Para poder hacer más o menos cluster se tiene que modificar el número de n_clusters
labels = clustering.fit_predict(df)

df['cluster']=labels
#print(df.head)

cluster0 = df[df['cluster']==0]
print('Objetos del cluster 0: ')
print(len(cluster0.index))

cluster1 = df[df['cluster']==1]
print('Objetos del cluster 1: ')
print(len(cluster1.index))

cluster2 = df[df['cluster']==2]
print('Objetos del cluster 2: ')
print(len(cluster2.index))

cluster3 = df[df['cluster']==3]
print('Objetos del cluster 3: ')
print(len(cluster3.index))

cluster4 = df[df['cluster']==4]
print('Objetos del cluster 4: ')
print(len(cluster4.index))

cluster5 = df[df['cluster']==5]
print('Objetos del cluster 5: ')
print(len(cluster5.index))

cluster6 = df[df['cluster']==6]
print('Objetos del cluster 6: ')
print(len(cluster6.index))

cluster7 = df[df['cluster']==7]
print('Objetos del cluster 7: ')
print(len(cluster7.index))


print("Total de objetos asignados a clusters:", df['cluster'].value_counts().sum())
print("Total de registros en el DataFrame:", len(df))

#Indice de calinski.Harabasz

ch_score = calinski_harabasz_score(df.drop(columns=['cluster']), labels)
print("Índice de validación (Calinski-Harabasz):", ch_score)

