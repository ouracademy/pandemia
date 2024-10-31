import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

df_positivos = pd.read_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSaudita.csv", sep=',')

def corregir_tested_incorrecto(v):
  c = 0               #contador de tested negativos
  x = 0     # promedio de tested

  for i in range (0, len(v)):
    if v[i] <0:             #si la Tested es negativo
      v[i]= x
      c=c+1    #aumenta el contador
  
  return c

tested = df_positivos['Tested']
nro_valores_corregidos = corregir_tested_incorrecto(tested)
print('N° de valores negativos reemplazados =', nro_valores_corregidos)

df_positivos.to_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepurado.csv", sep=',')

# -------------

df1_positivos = pd.read_csv("/content/drive/MyDrive/Tesis_Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepurado.csv", sep=',')

df = df1_positivos[['RegionName_EN','Confirmed','Deaths','Recovered','Tested','NewAdded']]

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

data = df

scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

imputer = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = imputer.fit_transform(data[categorical_cols])


encoder = OneHotEncoder()

encoded_cols = encoder.fit_transform(df[categorical_cols])
feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=feature_names)
df_encoded = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

df_encoded.to_csv("/content/drive/MyDrive/Tesis_Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepuradoEncoded.csv", sep=",")

