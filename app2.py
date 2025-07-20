import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# ✅ Set Page Config — Keep this first!
st.set_page_config(page_title="Real-Time AQI Analyzer", page_icon="🌐", layout="centered")

# ✅ Custom CSS for Background and White Overlay
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .overlay {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Environment Variables
load_dotenv()

# ✅ Load Pre-trained Model
@st.cache_resource
def load_model():
    with open('aqi_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ✅ Get API Key from .env
api_key = os.getenv('API_KEY')

# ✅ Function to fetch pollutants from OpenWeather API
def get_pollutants(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['list'][0]['components']
    else:
        return None

# ✅ AQI Prediction Function
def predict_aqi(comp_dict, hour):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

# ✅ App Header Section
st.markdown("""
    <div class="overlay">
        <h1>🌐 Real-Time AQI Analyzer & 5-Hour Forecast</h1>
        <h4>🚀 Enter coordinates to fetch current AQI and predict next 5 hours.</h4>
    </div>
""", unsafe_allow_html=True)

# ✅ User Input
lat = st.number_input("📍 Enter Latitude:", value=12.9169, format="%.6f")
lon = st.number_input("📍 Enter Longitude:", value=77.6247, format="%.6f")

# ✅ Prediction Button
if st.button("🔮 Predict AQI"):
    if api_key:
        pollutants = get_pollutants(lat, lon, api_key)
        if pollutants:
            current_hour = datetime.now().hour
            forecast = []
            for i in range(6):  # Current hour + next 5 hours
                future_hour = (current_hour + i) % 24
                predicted = predict_aqi(pollutants, future_hour)
                forecast.append({"Hour": future_hour, "Predicted AQI": int(predicted)})

            df_forecast = pd.DataFrame(forecast)

            st.markdown(f"""
                <div class="overlay">
                    <h3>✅ Current AQI Category: {df_forecast.iloc[0]['Predicted AQI']}</h3>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Forecast for Current + Next 5 Hours")
            st.dataframe(df_forecast, use_container_width=True)

            # ✅ Forecast Chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_forecast['Hour'], df_forecast['Predicted AQI'], marker='o', color='#FF5733', linewidth=2)
            ax.set_title('AQI Forecast', fontsize=14)
            ax.set_xlabel('Hour of the Day', fontsize=12)
            ax.set_ylabel('Predicted AQI Category', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.error("❌ Failed to fetch API data. Please check your coordinates or API Key.")
    else:
        st.warning("⚠️ API Key not set. Please check your .env file.")
