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

# lstm_model = Sequential()
# lstm_model.add(LSTM(32, input_shape=(7, 1)))  # La entrada es la salida combinada de los 7 modelos   3
# lstm_model.add(Dense(1, activation='sigmoid'))
# lstm_model.compile(optimizer='adam', loss='binary_crossentropy')

# Entrena la capa LSTM con las salidas combinadas de los tres modelos
# lstm_model.fit(combined_features, y_test, epochs=10)   # y_train

# Realizar predicciones en el conjunto de prueba
# y_pred = lstm_model.predict(combined_features)

# Calcular el Error Cuadrático Medio (MSE)
# mse = mean_squared_error(y_test, y_pred)
# Calcular el Error Cuadrático Medio Raíz (RMSE)
# rmse = np.sqrt(mse)
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')

USANDO RF:Obtener la longitud de y_train
n_samples_train = len(y_train)

Crea una lista vacía para almacenar las predicciones redimensionadas
X_stacked_list = []

Itera sobre las predicciones de cada modelo y redimensiona si es necesario
for model_name, predictions_array in predictions.items():
    Si la longitud de las predicciones no coincide con la de y_train, redimensiona
    if len(predictions_array) != n_samples_train:
        Ajusta el método de redimensionamiento según sea necesario (e.g., repetir, rellenar con ceros, etc.)
        En este ejemplo, se repite la predicción para que coincida con la longitud de y_train
        predictions_array = np.repeat(predictions_array, n_samples_train // len(predictions_array) + 1)[:n_samples_train]

    X_stacked_list.append(predictions_array)

Apila las predicciones redimensionadas para formar X_stacked
X_stacked = np.column_stack(X_stacked_list)


Entrenar un modelo Random Forest con las nuevas características
rf_stacked = RandomForestClassifier(n_estimators=100, random_state=42)
rf_stacked.fit(X_stacked, y_train)  # Asegúrate de que y_train sea el objetivo real

Hacer nuevas predicciones con el modelo Random Forest
Create X_test_stacked using the test set predictions
X_test_stacked = np.column_stack([predictions[model_name] for model_name in predictions])

Make predictions with rf_stacked
y_pred_stacked = rf_stacked.predict(X_test_stacked)

Calcular las métricas de desempeño
auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
precision = precision_score(y_test, y_pred_stacked, average='weighted')
accuracy = accuracy_score(y_test, y_pred_stacked)
f1 = f1_score(y_test, y_pred_stacked, average='weighted')
recall = recall_score(y_test, y_pred_stacked, average='weighted')
mae = mean_absolute_error(y_test, y_pred_stacked)
mse = mean_squared_error(y_test, y_pred_stacked)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_stacked)

Mostrar las métricas
print("Métricas para el modelo Random Forest con predicciones apiladas:")
print("AUC:", auc)
print(f"Precisión: {precision:.4f}")
print(f"Exactitud: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

USANDO XGBOOST:Crear un conjunto de datos con las predicciones como características
Crear un conjunto de datos con las predicciones como características
X_stacked = np.column_stack(list(predictions.values()))

Crear X_test_stacked usando las predicciones de test
X_test_stacked = np.column_stack([predictions[model_name] for model_name in predictions])

Asegurarte de que y_train tenga la misma longitud que las predicciones
if len(y_train) != len(predictions['XGBoost']):  # Utiliza RF como referencia
    y_train_sliced = y_train[:len(predictions['XGBoost'])]
else:
    y_train_sliced = y_train  # Usa y_train original si las longitudes coinciden

--- Asegúrate de que y_train_sliced solo contiene 0 y 1 ---
y_train_sliced = y_train_sliced[y_train_sliced.isin([0, 1])]

Reset the index of y_train_sliced to start from 0
y_train_sliced = y_train_sliced.reset_index(drop=True)

Reindex X_stacked to match y_train_sliced
X_stacked = X_stacked[y_train_sliced.index]


Convertir a DMatrix de XGBoost
dtrain = xgb.DMatrix(X_stacked, label=y_train_sliced)  # Usa el y_train cortado

Definir los parámetros para el modelo XGBoost
params = {
    'objective': 'binary:logistic',  # Para clasificación binaria
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 100
}

Entrenar el modelo XGBoost
xgboost_model = xgb.train(params, dtrain)

Hacer nuevas predicciones con el modelo XGBoost
dtest = xgb.DMatrix(X_test_stacked)
y_pred_probs = xgboost_model.predict(dtest)  # Obtener probabilidades predichas

Asegúrate de que y_test sea un array de enteros
y_test = y_test.astype(int)  # Convertir a array de enteros

Calcular AUC para clasificación binaria
try:
    auc = roc_auc_score(y_test, y_pred_probs)
    print(f"AUC: {auc:.4f}")
except ValueError as e:
    print("Error al calcular AUC:", e)

Calcular otras métricas
y_pred = (y_pred_probs >= 0.5).astype(int)  # Convertir probabilidades a clases binarias


Calcular las métricas de desempeño
precision = precision_score(y_test, y_pred_stacked, average='weighted')
accuracy = accuracy_score(y_test, y_pred_stacked)
f1 = f1_score(y_test, y_pred_stacked, average='weighted')
recall = recall_score(y_test, y_pred_stacked, average='weighted')
mae = mean_absolute_error(y_test, y_pred_stacked)
mse = mean_squared_error(y_test, y_pred_stacked)
r2 = r2_score(y_test, y_pred_stacked)
# Mostrar las métricas
print("Métricas para el modelo XGBoost con predicciones apiladas:")
print("AUC:", auc)
print(f"Precisión: {precision:.4f}")
print(f"Exactitud: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")



Se utiliza las salidas de los 7 algoritmos  como entrada para un clasificador SVM
svm_input = np.column_stack((rf_predictions.reshape(-1, 1), svm_predictions.reshape(-1, 1), gbm_predictions.reshape(-1, 1),
                               nb_predictions.reshape(-1, 1), dt_predictions.reshape(-1, 1),
                               xgb_predictions.reshape(-1, 1),lr_predictions.reshape(-1, 1)))
svm_classifier = SVC(kernel='linear', probability=True)  # Se puedes ajustar el kernel de  acuerdo a lo que se requiere
svm_classifier.fit(svm_input, y_test)

Hacer predicciones con el clasificador SVM
svm_predictions = svm_classifier.predict(svm_input)

Obtener probabilidades predichas usando 'predict_proba'
y_probs = svm_classifier.predict_proba(svm_input)



Calcular las métricas de evaluación
auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
precision = precision_score(y_test, svm_predictions, average='macro')
accuracy = accuracy_score(y_test, svm_predictions)
f1 = f1_score(y_test, svm_predictions, average='macro')
recall = recall_score(y_test, svm_predictions,average='macro')
mse = mean_squared_error(y_test, svm_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, svm_predictions)

Imprimir los resultados
print("Métricas de evaluación:")
print("AUC:", auc)
print("Precision:", precision)
print("Accuracy:", accuracy)
print("F1 score:", f1)
print("Recall:", recall)
print("MSE:", mse)
print("RMSE:", rmse)
print(f"R2 Score: {r2:.4f}")

Binarizar la salida
y_test_bin = label_binarize(y_test, classes=np.unique(svm_predictions))
n_classes = y_test_bin.shape[1]

Calcular la curva ROC y el área ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    # Use roc_auc_score instead of the variable 'auc' to calculate the AUC
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_probs[:, i])

Calcular la curva ROC micropromedio y el área ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
Use roc_auc_score instead of the variable 'auc' to calculate the AUC
roc_auc["micro"] = roc_auc_score(y_test_bin.ravel(), y_probs.ravel())

