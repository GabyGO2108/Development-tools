import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("your_dataset.csv")  # Replace with your actual file name

# App content
st.header("My Interactive Data App")

if st.button("Show Histogram"):
    fig = px.histogram(df, x="odometer", title="Histogram of Odometer")
    st.plotly_chart(fig)

st.header("Car Data Analysis")