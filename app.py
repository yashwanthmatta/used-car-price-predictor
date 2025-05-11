
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Used Car Price Predictor + GPT", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.markdown("Predict used car prices and receive AI-powered explanations.")

# Load model and scaler
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("ridge_model.pkl")
    return scaler, model

scaler, model = load_model()

# Input fields
st.subheader("Enter Car Details:")
year = st.slider("Year", 2000, 2024, 2018)
km_driven = st.number_input("Kilometers Driven", min_value=5000, max_value=300000, value=50000, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200)
power = st.number_input("Power (bhp)", min_value=30.0, max_value=400.0, value=90.0)
seats = st.slider("Seats", 2, 10, 5)

# Prediction
if st.button("Predict Price"):

st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Predictor")
st.write("This is a test deployment. Your model integration goes here.")

# Simple inputs
year = st.slider("Year", 2000, 2024, 2015)
km = st.number_input("Kilometers Driven", 10000)

if st.button("Predict"):
    st.success("🚀 This is where the predicted price would show.")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Used Car Price Predictor + GPT", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.markdown("Predict used car prices and receive AI-powered explanations.")

# Load model and scaler
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("ridge_model.pkl")
    return scaler, model

scaler, model = load_model()

# Input fields
st.subheader("Enter Car Details:")
year = st.slider("Year", 2000, 2024, 2018)
km_driven = st.number_input("Kilometers Driven", min_value=5000, max_value=300000, value=50000, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200)
power = st.number_input("Power (bhp)", min_value=30.0, max_value=400.0, value=90.0)
seats = st.slider("Seats", 2, 10, 5)

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Year": year,
        "Kilometers_Driven": km_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Engine": engine,
        "Power": power,
        "Seats": seats
    }])

    input_df = pd.get_dummies(input_df)
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    X_scaled = scaler.transform(input_df)
    predicted_price = model.predict(X_scaled)[0]

    st.success(f"💰 Estimated Price: ₹ {predicted_price:.2f} lakhs")
    st.info(f"🔍 Based on: {year}, {km_driven} km, {fuel_type}, {transmission}, {owner_type}, {engine}cc, {power}bhp, {seats} seats.")



