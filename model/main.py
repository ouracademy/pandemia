import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


def corregir_tested_incorrecto(v):
  x = 0     # promedio de tested

  for i in range (0, len(v)):
    if v[i] <0:             #si la Tested es negativo
      v[i]= x


n_components = 5

pipeline = make_pipeline(
  StandardScaler(),
  PCA(n_components=n_components)
)

# full_pipeline = ColumnTransformer([
#   ('num', num_pipeline, numeric_cols),
#   ('cat', cat_pipeline, categorical_cols)
# ])

# df_encoded = pipeline.fit_transform()


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


df1_positivos = pd.read_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSaudita.csv", sep=',')
corregir_tested_incorrecto(df1_positivos['Tested'])

df1_positivos['Confirmed'] = df1_positivos['Confirmed'].astype(int) # from hibrido.ipynb
columns_for_pca = ['Confirmed', 'Deaths', 'Recovered', 'Tested', 'NewAdded']
data = df1_positivos[columns_for_pca]

X = data.drop('Confirmed', axis=1)
y = data['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_hyperparameters = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale"
}

# Hiperparámetros para Random Forest
random_forest_hyperparameters = {
    "n_estimators": 100,
    "max_depth": 10,
    #"min_samples_split": 2,
    #"min_samples_leaf": 1
}

# Hiperparámetros para XGBoost
xgboost_hyperparameters = {
    "learning_rate": 0.01,
    "n_estimators": 200,
    "max_depth": 7
}
# Hiperparámetros para DT
DT_hyperparameters = {
    'max_depth': 15
}
# Hiperparámetros para GB
GB_hyperparameters = {
   'subsample': 1.0,
    "n_estimators": 200,
    "max_depth": 7,
   "learning_rate": 0.01
}
# Hiperparámetros para LR
LR_hyperparameters = {
'solver': 'saga',
 'C': 0.1
}

# Entrenar modelos individuales
## Entrena un modelo RF
rf_model = RandomForestClassifier(**random_forest_hyperparameters)
rf_model.fit(X_train, y_train)

# Entrena un modelo SVM
svm_model = SVC(**svm_hyperparameters)
svm_model.fit(X_train, y_train)

# Entrena un modelo GBM
gbm_model = GradientBoostingClassifier(**GB_hyperparameters)
gbm_model.fit(X_train, y_train)

# Entrena un modelo Naive Bayes- No necesita Hiperparámetros
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Entrena un modelo DT
dt_model = DecisionTreeClassifier(**DT_hyperparameters)
dt_model.fit(X_train, y_train)

# Entrena el modelo XGBoost
unique_classes = np.unique(y_train)
print("Clases únicas en y_train:", unique_classes)
# Obtener un mapeo de las clases originales a las nuevas clases consecutivas
class_mapping = {original_class: new_class for new_class, original_class in enumerate(unique_classes)}
# Aplicar el mapeo a las etiquetas de entrenamiento
y_train_consecutive = np.array([class_mapping[cls] for cls in y_train])
#Utilizar y_train_consecutive en lugar de y_train para entrenar el modelo:
xgb_model = xgb.XGBClassifier(**xgboost_hyperparameters)
xgb_model.fit(X_train, y_train_consecutive)

# Crear y ajustar un modelo de regresión logística
lr_model = LogisticRegression(**LR_hyperparameters)
lr_model.fit(X_train, y_train)

# Realizar predicciones con los datos de entrenamiento
rf_predictions_train = rf_model.predict(X_train)
svm_predictions_train = svm_model.predict(X_train)
gbm_predictions_train = gbm_model.predict(X_train)
nb_predictions_train = naive_bayes.predict(X_train)
dt_predictions_train = dt_model.predict(X_train)
xgb_predictions_train = xgb_model.predict(X_train)
lr_predictions_train = lr_model.predict(X_train)

predictions = {}

# Realizar predicciones con los datos de prueba
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
gbm_predictions = gbm_model.predict(X_test)
nb_predictions = naive_bayes.predict(X_test)
dt_predictions = dt_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Almacenar las predicciones en el diccionario
predictions['Random Forest'] = rf_predictions
predictions['SVM'] = svm_predictions
predictions['Decision Tree'] = dt_predictions
predictions['Gradient Boosting'] = gbm_predictions
predictions['XGBoost'] =xgb_predictions
predictions['Naive Bayes'] = nb_predictions
predictions['Logistic Regression'] = lr_predictions
# Imprimir el diccionario para verificar
print(predictions)

# Función para calcular y reportar las métricas
# def report_metrics(y_true, y_pred, model_name):
    # model_metrics = {
    #     "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
    #     "Accuracy": accuracy_score(y_true, y_pred),
    #     "F1 Score": f1_score(y_true, y_pred, average='weighted'),
    #     "Recall": recall_score(y_true, y_pred, average='weighted'),
    #     "MAE": mean_absolute_error(y_true, y_pred),
    #     "MSE": mean_squared_error(y_true, y_pred),
    #     "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    #     "R2 Score": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None  # R2 solo si hay variación
    # }

    # Imprimir las métricas
    # print(f"\nMétricas para el modelo {model_name}:")
    # for metric_name, metric_value in model_metrics.items():
    #     if metric_value is not None:  # Mostrar solo las métricas calculadas
    #         print(f"{metric_name}: {metric_value:.4f}")

# Iterar sobre el diccionario de predicciones y calcular las métricas
# for model_name, y_pred in predictions.items():
#     report_metrics(y_test, y_pred, model_name)

# Combina las salidas de los 7 modelos y utiliza una capa LSTM
combined_features = np.hstack((rf_predictions.reshape(-1, 1), svm_predictions.reshape(-1, 1), gbm_predictions.reshape(-1, 1),
                               nb_predictions.reshape(-1, 1), dt_predictions.reshape(-1, 1),
                               xgb_predictions.reshape(-1, 1),lr_predictions.reshape(-1, 1)))

