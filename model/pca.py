# La primera parte tiene por objetivo hacer un preprocesamiento de los datos para asegurarse estén limpios y en un formato adecuado para el análisis

# from fim import *
# import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# from scipy.sparse import csr_matrix, vstack
# from sklearn.decomposition import PCA

df1_positivos = pd.read_csv("/content/drive/MyDrive/Tesis_Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepurado.csv", sep=',')

df = df1_positivos[['RegionName_EN','Confirmed','Deaths','Recovered','Tested','NewAdded']]

# Separar las variables numéricas y categóricas
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

data = df

# Estandarizar las variables numéricas utilizando la estandarización Z-score
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Manejar los valores perdidos de las variables categóricas , con la moda
imputer = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = imputer.fit_transform(data[categorical_cols])


# crear un codificador one-hot
encoder = OneHotEncoder()

# codificar variables categóricas
encoded_cols = encoder.fit_transform(df[categorical_cols])

# obtener los nombres de las características
feature_names = encoder.get_feature_names_out(categorical_cols)

# crear un DataFrame con las características codificadas y sus nombres
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=feature_names)

# combinar el DataFrame codificado con el DataFrame original
df_encoded = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Calcular la matriz de covarianza
# matriz_covarianza = np.cov(df_encoded.T)

# Imprimir la matriz de covarianza
# print(matriz_covarianza)

# """Cálculo de valores y vectores propios"""

# Calcular los valores y vectores propios
# valores_propios, vectores_propios = np.linalg.eig(matriz_covarianza)

# Ordenar los valores y vectores propios de mayor a menor
# indices_ordenados = np.argsort(valores_propios)[::-1]
# valores_propios_ord = valores_propios[indices_ordenados]
# vectores_propios_ord = vectores_propios[:,indices_ordenados]

# Imprimir los valores propios ordenados
# print(valores_propios_ord)

# Imprimir los vectores propios ordenados
# print(vectores_propios_ord)

# """Selección de componentes principales"""

# Calcular la varianza acumulada explicada por cada componente principal
# varianza_explicada = np.cumsum(valores_propios_ord) / np.sum(valores_propios_ord)

# Encontrar el número de componentes principales que explican al menos el 95% de la varianza
# n_componentes = np.argmax(varianza_explicada >= 0.95) + 1

# Imprimir el número de componentes principales seleccionados
# print("Número de componentes principales seleccionados:", n_componentes)

# Crear un objeto PCA con el número de componentes principales seleccionados
# pca = PCA(n_components=n_componentes)

# Ajustar el PCA a los datos preprocesados y transformar los datos
# datos_transformados = pca.fit_transform(df_encoded)

# Obtener los nombres de las variables originales
# nombres_variables = list(df_encoded.columns)  # se tiene un DataFrame con columnas nombradas

# Ahora Vamos a guardar el df actualizado
df_encoded.to_csv("/content/drive/MyDrive/Tesis_Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepuradoEncoded.csv", sep=",")

#Crear un DataFrame para visualizar las proporciones de varianza explicada y los nombres de los componentes
# df_componentes = pd.DataFrame({'Varianza Explicada': varianza_explicada,
#                                'Nombres Componentes': nombres_variables})    #nombres_componentes


# Obtener los vectores de carga de las variables originales en cada componente principal
# vectores_carga = pca.components_

# Crear un DataFrame para visualizar los nombres de las variables originales y los vectores de carga
# df_carga = pd.DataFrame(vectores_carga, columns=nombres_variables)

# Imprimir el DataFrame
# print(df_carga)