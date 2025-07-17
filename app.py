import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import os

# ✅ Load Pre-trained Model
@st.cache_resource
def load_model():
    with open('aqi_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ✅ Function to Get Pollutants from OpenWeather
def get_pollutants(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['list'][0]['components']
    else:
        st.error(f"❌ Failed to fetch API data! Status: {response.status_code}")
        return None

# ✅ Predict AQI using the model
def predict_aqi(comp_dict, hour, model):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

# ✅ Streamlit App UI
st.title("🌐 Real-Time AQI Analyzer & 5-Hour Forecast")


# ✅ Input Section
lat = st.number_input("Enter Latitude:", value=12.9169)
lon = st.number_input("Enter Longitude:", value=77.6247)


if st.button("🔮 Predict AQI"):
    api_key=os.getenv('api_key')
    if api_key:
        comp = get_pollutants(lat, lon, api_key)
        if comp:
            cur_hour = datetime.now().hour
            current_aqi = predict_aqi(comp, cur_hour, model)
            st.success(f"✅ Current AQI Category: {current_aqi}")

            st.subheader("📈 Forecast for Next 5 Hours:")
            for i in range(1, 6):
                future_hour = (cur_hour + i) % 24
                future_aqi = predict_aqi(comp, future_hour, model)
                st.write(f"Hour +{i} (Hour {future_hour}): AQI Category = {future_aqi}")
    else:
        st.warning("Please enter your OpenWeather API Key.")
