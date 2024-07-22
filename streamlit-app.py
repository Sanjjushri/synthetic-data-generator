import streamlit as st
import pandas as pd 
from ctgan import CTGAN 

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:

    number = st.number_input('Number of rows', min_value=0, step=1000)

    df = pd.read_csv(uploaded_file)

    categorical_features= df.select_dtypes(exclude="number").columns.tolist()

    ctgan = CTGAN(epochs=5)

    ctgan.fit(df, categorical_features)

    synthetic_df = ctgan.sample(number)

    st.write(synthetic_df)