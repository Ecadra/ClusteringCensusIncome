import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

df = pd.read_csv('./Data/CSVs/census_income_first_classification.csv')
df_copy = df.copy()

X_train, X_test, y_train, y_test = train_test_split(df_copy.drop(columns=['income']),
                                                     df_copy['income'], test_size=0.3, random_state=42, train_size=0.7)

# Configuración por default
mlp = MLPClassifier(max_iter=1000)

results = cross_validate(mlp, X_train, y_train, cv=5, return_train_score=True,
                         scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])

# Resultados
print("-Precisión promedio: ", results['test_accuracy'].mean())
print("-F1 promedio: ", results['test_f1_macro'].mean())


"""
-Precisión promedio:  0.9015714285714285
-F1 promedio:  0.715712504580398
"""