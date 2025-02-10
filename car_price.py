import pandas as pd
import numpy as np
import pickle as pk
import base64
import streamlit as st

# Load the model
try:
    model = pk.load(open('c:/Users/vinuv/car_price_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure the file path is correct.")
    st.stop()

# Load the dataset
try:
    cars_data = pd.read_csv("C:/Users/vinuv/Downloads/Cardetails.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the file path is correct.")
    st.stop()

# Helper function to extract car brand
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Create tabs
tab1, tab2 = st.tabs(["🏠 Home", "📈 Car Price Prediction"])

# Home tab
with tab1:
    st.header("🚗 Welcome to Car Price Prediction")
    st.write("### 🌟 Revolutionizing the Way You Evaluate Car Prices!")
    st.write(
        "Car price prediction is a cutting-edge solution that combines the power of **machine learning** "
        "and real-world data to provide accurate and reliable car valuations. Whether you're a car enthusiast, "
        "buyer, or seller, this tool empowers you to make smarter decisions."
    )
    st.write(
        "🔍 **Key Highlights:**"
        "\n- Estimate car prices based on brand, mileage, year, and other features."
        "\n- Avoid overpaying or underselling with fair market value predictions."
        "\n- Save time and effort by using a reliable AI-driven model."
    )
    st.write("🌟 Explore the next tab to predict car prices with just a few clicks! 🚀")

# Car Price Prediction tab
with tab2:
    st.header("📈 Car Price Prediction ML Model")

    # Input fields
    name = st.selectbox('Select Car Brand', cars_data['name'].unique())
    year = st.number_input('Car Manufactured Year', 1994, 2024, step=1)
    km_driven = st.number_input('Number of Kilometers Driven', 11, 200000, step=100)
    fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
    transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
    owner = st.selectbox('Owner Type', cars_data['owner'].unique())
    mileage = st.number_input('Car Mileage (km/l)', 10.0, 40.0, step=0.1)
    engine = st.number_input('Engine Capacity (CC)', 700, 5000, step=50)
    max_power = st.number_input('Maximum Power (HP)', 0, 200, step=10)
    seats = st.number_input('Number of Seats', 2, 10, step=1)

    # Prediction button
    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )

        # Mapping categorical values
        mappings = {
            'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 
                      'Fourth & Above Owner': 4, 'Test Drive Car': 5},
            'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
            'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
            'transmission': {'Manual': 1, 'Automatic': 2},
            'name': {brand: i+1 for i, brand in enumerate(cars_data['name'].unique())}
        }

        for col, mapping in mappings.items():
            input_data[col].replace(mapping, inplace=True)

        try:
            # Predicting car price
            car_price = model.predict(input_data)
            st.success(f'💰 Predicted Car Price: ₹ {car_price[0]:,.2f}')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
