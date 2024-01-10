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

# Estou trocando o valor de valores nas colunas desconhecidos pela média de valor daquela coluna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

#Divida o conjunto de dados em um conjunto de treinamento-teste de 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie uma instância da classe StandardScaler
sc = StandardScaler()

# Aplique o StandardScaler nas características do conjunto de treinamento e faça a transformação 
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Imprima os conjuntos de dados de treinamento e teste escalados
print("Scaled training set:\n", X_train_scaled) 
print("Scaled test set:\n", X_test_scaled)


