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