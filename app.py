import streamlit as st
import pandas as pd
import joblib

st.title("Modelo para pandemia COVID 19")

with st.sidebar:
    st.write("Requerimientos de datos")
    st.caption("Para inferir el modelo, debe cargar un archivo en formato csv con las siguientes columnas")
    with st.expander("Formato de datos"):
      st.markdown("- Deaths (Muertos)")
      st.markdown("- Recovered (Recuperados)")
      st.markdown("- Tested (Probados)")
      st.markdown("- NewAdded (Agregados recientemente)")
      # agregar resto de campos

uploaded_file = st.file_uploader("Selecciona un archivo", type="csv")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write("Una muestra de lo subido:")
  st.write(df.sample(5))
  st.write(f"Número de filas: {len(df)}")
  

  model = joblib.load('model.joblib')
  promedio_tested = 0
  df['Tested'] = df['Tested'].where(df['Tested'] >= 0, promedio_tested)

  pred = model.predict(df[['Deaths', 'Recovered', 'Tested', 'NewAdded']])
  pred = pd.DataFrame(pred, columns = ['ok'])
  st.header('Valores predecidos')
  st.write(pred.head())
  pred = pred.to_csv(index=False).encode('utf-8')
  st.download_button('Descargar predicción',
                  pred,
                  'prediction.csv',
                  'text/csv',
                  key='download-csv')