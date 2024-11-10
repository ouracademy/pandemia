import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


df1_positivos = pd.read_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSaudita.csv", sep=',')

def corregir_tested_incorrecto(v):
  x = 0     # promedio de tested

  for i in range (0, len(v)):
    if v[i] <0:             #si la Tested es negativo
      v[i]= x

tested = df1_positivos['Tested']
corregir_tested_incorrecto(tested)

columns_for_pca = ['Confirmed', 'Deaths', 'Recovered', 'Tested', 'NewAdded']
df = df1_positivos[['RegionName_EN','Confirmed','Deaths','Recovered','Tested','NewAdded']]
X = df[columns_for_pca]

n_components = 5

pipeline = make_pipeline(
  StandardScaler(),
  PCA(n_components=n_components)
)

# full_pipeline = ColumnTransformer([
#   ('num', num_pipeline, numeric_cols),
#   ('cat', cat_pipeline, categorical_cols)
# ])

# df_encoded = full_pipeline.fit_transform()


# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# imputer = SimpleImputer(strategy="most_frequent")
# df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
# encoder = OneHotEncoder()
# encoded_cols = encoder.fit_transform(df[categorical_cols])

#region TODO
# feature_names = encoder.get_feature_names_out(categorical_cols)
# encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=feature_names)
# df_encoded = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
#endregion 

# df_encoded.to_csv("/content/drive/MyDrive/Tesis_Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepuradoEncoded.csv", sep=",")