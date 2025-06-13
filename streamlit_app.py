import streamlit as st
import pandas as pd
import numpy as np

st.title("🚀 My First Streamlit App")

st.write("Welcome! Here's a simple demo using pandas + Streamlit.")

# Generate example data
df = pd.DataFrame({
    "x": np.arange(10),
    "y": np.random.randn(10)
})

# Show data
st.subheader("📊 Data Table")
st.dataframe(df)

# Plot
st.subheader("📈 Line Chart")
st.line_chart(df.set_index("x"))

# User input
st.subheader("🔧 Try it yourself")
name = st.text_input("Enter your name")
if name:
    st.success(f"Hello, {name}!")
