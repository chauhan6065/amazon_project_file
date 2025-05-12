import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# Load the trained model
model = joblib.load("best_model.pkl")  # make sure this path is correct

# Title
st.title("Amazon Delivery Time Predictor")

# Inputs from user
agent_age = st.number_input("Agent Age", min_value=18, max_value=65, step=1)
agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, step=0.1)

store_lat = st.number_input("Store Latitude")
store_long = st.number_input("Store Longitude")
drop_lat = st.number_input("Drop Latitude")
drop_long = st.number_input("Drop Longitude")

weather = st.selectbox("Weather Conditions", ['Sunny', 'Stormy', 'Cloudy', 'Fog', 'Windy', 'Sandstorms'])
traffic = st.selectbox("Traffic Conditions", ['Low', 'Medium', 'High', 'Jam'])
vehicle = st.selectbox("Vehicle Used", ['Bike', 'Car', 'Scooter', 'Truck'])
area = st.selectbox("Area Type", ['Urban', 'Metropolitan'])
category = st.selectbox("Product Category", ['Grocery', 'Clothing', 'Electronics', 'Home Decor', 'Pharmacy'])

# Feature Engineering: Calculate distance
distance = geodesic((store_lat, store_long), (drop_lat, drop_long)).km

# Convert inputs to DataFrame
input_df = pd.DataFrame({
    'Agent_Age': [agent_age],
    'Agent_Rating': [agent_rating],
    'Distance': [distance],
    'Weather': [weather],
    'Traffic': [traffic],
    'Vehicle': [vehicle],
    'Area': [area],
    'Category': [category]
})

# Encode categorical features (make sure this matches your training encoding)
input_encoded = pd.get_dummies(input_df)

# Ensure the input has all columns model was trained on
model_columns = joblib.load("model_columns.pkl")
for col in model_columns:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} hours")

    joblib.dump(model, "best_model.pkl")
