import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# carregar o arquivo csv
url = 'Url do arquivo csv'
delimiter = ';'
dataset = pd.read_csv(url, delimiter=delimiter)

# Separate features e o target
X = dataset.drop('Nome do dataset target', axis=1) 
y = dataset['Nome do dataset target']

print(X)
print (y)

# identificar as colunas numericas e categoricas
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Estou trocando o valor de valores nas colunas desconhecidos pela média de valor daquela coluna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# Crie uma instância da classe StandardScaler e one hot encoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X = ct.fit_transform(X)

# Codifica o target (se for categórico)
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
  
#Divida o conjunto de dados em um conjunto de treinamento-teste de 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplique o StandardScaler nas características do conjunto de treinamento e faça a transformação 
sc = StandardScaler()
num_columns = len(categorical_cols)
X_train[:, num_columns:] = sc.fit_transform(X_train[:, num_columns:])
X_test[:, num_columns:] = sc.transform(X_test[:, num_columns:])

# Imprima os conjuntos de dados de treinamento e teste escalados se quiser
print("Scaled training set:\n", X_train) 
print("Scaled test set:\n", X_test)


