import streamlit as st
import pandas as pd

st.title("Modelo para pandemia COVID 19")

with st.sidebar:
    st.write("Requirimientos de datos")
    st.caption("Para inferir el modelo, debe cargar un archivo en formato csv con las siguientes columnas")
    with st.expander("Formato de datos"):
      st.markdown("- Deaths (Muertos)")
      st.markdown("- Recovered (Recuperados)")
      st.markdown("- Tested (Probados)")
      st.markdown("- NewAdded (Agregados recientemente)")
      # agregar resto de campos

uploaded_file = st.file_uploader("Selecciona un archivo", type="csv")
if uploaded_file is not None:
  dataframe = pd.read_csv(uploaded_file)
  st.write("Una muestra de lo subido:")
  st.write(dataframe.sample(5))
  st.write(f"Número de filas: {len(dataframe)}")
  

  # model = joblib.load('model.joblib')
  # pred = model.predict_proba(df)
  # pred = pd.DataFrame(pred, columns = ['setosa_probability', 'versicolor_probability', 'virginica_probability'])
  pred = dataframe # TODO: delete this
  st.header('Predicted values')
  st.write(pred.head())
  pred = pred.to_csv(index=False).encode('utf-8')
  st.download_button('Download prediction',
                  pred,
                  'prediction.csv',
                  'text/csv',
                  key='download-csv')