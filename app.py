import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Predictor")
st.write("This is a test deployment. Your model integration goes here.")

# Simple inputs
year = st.slider("Year", 2000, 2024, 2015)
km = st.number_input("Kilometers Driven", 10000)

if st.button("Predict"):
    st.success("🚀 This is where the predicted price would show.")


