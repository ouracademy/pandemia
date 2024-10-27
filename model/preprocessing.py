import pandas as pd

"""#### DATASET: Se utilizará el data set de casos de Covid en Arabia Saudita tomado de https://kaggle.com"""

df_positivos = pd.read_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSaudita.csv", sep=',')

#En el caso de las Tested incorrectos se reemplazarán con el promedio
prom= df_positivos['Tested'].mean()  # prom aprox = 1
prom

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

# Ahora Vamos a guardar el df actualizado
df_positivos.to_csv("/content/drive/MyDrive/Desarrollo/DataPorPais/daily_cases_ksa_covid19ArabiaSauditaDepurado.csv", sep=',')