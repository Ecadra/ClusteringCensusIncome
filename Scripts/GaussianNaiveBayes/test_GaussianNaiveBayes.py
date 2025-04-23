from sklearn.datasets import load_iris #*dataset de prueba
from sklearn.model_selection import train_test_split #*dividir el dataser en train y test
from sklearn.naive_bayes import GaussianNB #*algoritmo de Naive Bayes tipo Gaussiano


X, y = load_iris(return_X_y=True)
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
'''
En la líena de arriba se esta dividiendo el dataset en dos partes, una para entrenar y otra para probar el modelo.
X son las variables independientes 
y es la valiable objetivo
test_size=0.5 significa que el 50% de los datos se usaran para entrenar y el otro 50% para probar el modelo.
Existe el parametros train_size que indica el porcentaje de datos que se usaran para entrenar el modelo.
random_state=0 es una semilla para que los resultados sean reproducibles.
Si no se pone este parametro, cada vez que se ejecute el programa se obtendran resultados diferentes.
'''
gnb = GaussianNB()#*Objetivo de la clase GaussianNB
y_pred = gnb.fit(X_train, y_train).predict(X_test)
'''
.fit(X_train, y_train) entrena el modelo con los datos de entrenamiento.
.predict(X_test) generea las predicciones para los datos de prueba.
y_pred es un array con las predicciones del modelo para los datos de prueba.
'''
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
'''
'''