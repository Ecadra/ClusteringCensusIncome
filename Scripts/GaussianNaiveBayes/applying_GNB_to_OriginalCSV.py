import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

df= pd.read_csv('./Data/CSVs/census_income_first_classification.csv')
df_copy = df.copy()

X_train,X_test,y_train,y_test = train_test_split(df_copy.drop(columns=['income']),
                                                 df_copy['income'],test_size=0.3,random_state=42,train_size=0.7)

'''
En la líena de arriba se esta dividiendo el dataset en dos partes, una para entrenar y otra para probar el modelo.
X son las variables independientes 
y es la valiable objetivo
test_size=0.5 significa que el 50% de los datos se usaran para entrenar y el otro 50% para probar el modelo.
Existe el parametros train_size que indica el porcentaje de datos que se usaran para entrenar el modelo.
random_state=0 es una semilla para que los resultados sean reproducibles.
Si no
'''
gnd=GaussianNB()

results=cross_validate(gnd,X_train,y_train,cv=5,return_train_score=True, scoring=['accuracy','f1_macro','precision_macro','recall_macro'])

print("-Precisión promedio: ", results['test_accuracy'].mean())
print("-F1 promedio: ", results['test_f1_macro'].mean())
#?y_pred=gnd.fit(X_train,y_train).predict(X_test)

'''
.fit(X_train, y_train) entrena el modelo con los datos de entrenamiento.
.predict(X_test) generea las predicciones para los datos de prueba.
y_pred es un array con las predicciones del modelo para los datos de prueba.
'''

#?print("\n-Precisión usando Naive Bayes: ", accuracy_score(y_test,y_pred))