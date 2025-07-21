import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# ✅ Set page config at top
st.set_page_config(page_title="Real-Time AQI Analyzer", page_icon="🌐", layout="centered")

# ✅ Custom CSS
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

# ✅ Load Model
@st.cache_resource
def load_model():
    with open('aqi_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ✅ Get API Key
api_key = os.getenv('API_KEY')

# ✅ Fetch Pollutants
def get_pollutants(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['list'][0]['components']
    else:
        return None

# ✅ Predict AQI
def predict_aqi(comp_dict, hour):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

# ✅ App Header
st.markdown("""
    <div class="overlay">
        <h1>🌐 Real-Time AQI Analyzer & Next 5-Hour Forecast</h1>
        <h4>🚀 Enter coordinates to fetch current pollutants and predict AQI for next 5 hours.</h4>
    </div>
""", unsafe_allow_html=True)

# ✅ Input Section
lat = st.number_input("📍 Enter Latitude:", value=12.9169, format="%.6f")
lon = st.number_input("📍 Enter Longitude:", value=77.6247, format="%.6f")

# ✅ On Predict
if st.button("🔮 Predict AQI for Next 5 Hours"):
    if api_key:
        pollutants = get_pollutants(lat, lon, api_key)
        if pollutants:
            current_hour = datetime.now().hour
            forecast = []
            for i in range(1, 6):  
                future_hour = (current_hour + i) % 24
                predicted = predict_aqi(pollutants, future_hour)
                forecast.append({"Hour": future_hour, "Predicted AQI": int(predicted)})

            df_forecast = pd.DataFrame(forecast)

            st.markdown("""
                <div class="overlay">
                    <h3>✅ AQI Forecast for Next 5 Hours</h3>
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(df_forecast, use_container_width=True)

            # ✅ Line Chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_forecast['Hour'], df_forecast['Predicted AQI'], marker='o', color='#FF5733', linewidth=2)
            ax.set_title('Next 5 Hours AQI Forecast', fontsize=14)
            ax.set_xlabel('Hour of the Day', fontsize=12)
            ax.set_ylabel('Predicted AQI', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.error("❌ Failed to fetch pollutants. Check coordinates or API Key.")
    else:
        st.warning("⚠️ API Key not set in environment variables. Please check your .env file.")
