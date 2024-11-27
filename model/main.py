import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# import numpy as np
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib

# def corregir_tested_incorrecto(v):
#   x = 0     # promedio de tested

#   for i in range (0, len(v)):
#     if v[i] <0:             #si la Tested es negativo
#       v[i]= x


# TODO: check
n_components = 5

svm_hyperparameters = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale"
}

random_forest_hyperparameters = {
    "n_estimators": 100,
    "max_depth": 10,
}

xgboost_hyperparameters = {
    "learning_rate": 0.01,
    "n_estimators": 200,
    "max_depth": 7
}

DT_hyperparameters = {
    'max_depth': 15
}

GB_hyperparameters = {
   'subsample': 1.0,
    "n_estimators": 200,
    "max_depth": 7,
   "learning_rate": 0.01
}

LR_hyperparameters = {
'solver': 'saga',
 'C': 0.1
}

estimators = [
    ("rf", RandomForestClassifier(**random_forest_hyperparameters)),
    ("svc", SVC(**svm_hyperparameters)),
    ("gb", GradientBoostingClassifier(**GB_hyperparameters)),
    ("gnb", GaussianNB()),
    ("dt", DecisionTreeClassifier(**DT_hyperparameters)),
    ("lr", LogisticRegression(**LR_hyperparameters))
]

pipeline = make_pipeline(
  StandardScaler(),
#   PCA(n_components=n_components),
    # RandomForestClassifier(**random_forest_hyperparameters)
  StackingClassifier(estimators=estimators, final_estimator=SVC(kernel='linear', probability=True))
)


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


df = pd.read_csv("daily_cases_ksa.csv", sep=',')
promedio_tested = 0
df['Tested'] = df['Tested'].where(df['Tested'] >= 0, promedio_tested)
df['Confirmed'] = df['Confirmed'].astype(int) # from hibrido.ipynb
df['Confirmed'] = df['Confirmed'].where(df['Confirmed'] <= 0, 1) # TODO: evaluate this
columns_for_pca = ['Confirmed', 'Deaths', 'Recovered', 'Tested', 'NewAdded']
data = df[columns_for_pca]

X = data.drop('Confirmed', axis=1)
y = data['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("fit")
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model.joblib")

print("predict")
y_test_pred = pipeline.predict(X_test)

# Obtener probabilidades predichas usando 'predict_proba'
# y_test_pred_probs = pipeline.predict_proba(X_test)

print(y_test_pred)

# Calcular las métricas de evaluación
# auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
# precision = precision_score(y_test, svm_predictions, average='macro')
# accuracy = accuracy_score(y_test, svm_predictions)
# f1 = f1_score(y_test, svm_predictions, average='macro')
# recall = recall_score(y_test, svm_predictions,average='macro')
# mse = mean_squared_error(y_test, svm_predictions)
# rmse = mse ** 0.5
# r2 = r2_score(y_test, svm_predictions)

# # Imprimir los resultados
# print("Métricas de evaluación:")
# print("AUC:", auc)
# print("Precision:", precision)
# print("Accuracy:", accuracy)
# print("F1 score:", f1)
# print("Recall:", recall)
# print("MSE:", mse)
# print("RMSE:", rmse)
# print(f"R2 Score: {r2:.4f}")

# Binarizar la salida
# y_test_bin = label_binarize(y_test, classes=np.unique(svm_predictions))
# n_classes = y_test_bin.shape[1]

# # Calcular la curva ROC y el área ROC para cada clase
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
#     # Use roc_auc_score instead of the variable 'auc' to calculate the AUC
#     roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_probs[:, i])

# # Calcular la curva ROC micropromedio y el área ROC
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
# # Use roc_auc_score instead of the variable 'auc' to calculate the AUC
# roc_auc["micro"] = roc_auc_score(y_test_bin.ravel(), y_probs.ravel())

