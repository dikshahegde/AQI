import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# ✅ Always at the top!
st.set_page_config(page_title="Real-Time AQI Analyzer", page_icon="🌐", layout="centered")

# ✅ Background CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load .env locally (optional)
load_dotenv()

# ✅ Load Pre-trained Model
@st.cache_resource
def load_model():
    with open('aqi_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ✅ Function to fetch pollutants from OpenWeather
def get_pollutants(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['list'][0]['components']
    else:
        return None

# ✅ Predict AQI for given pollutants & hour
def predict_aqi(comp_dict, hour, model):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

# ✅ App Header
st.title("🌐 Real-Time AQI Analyzer & 5-Hour Forecast")
st.markdown("""
            <div style='background-color: white; padding: 15px; border-radius: 10px;'>
                <h3 style='color: black;'><b>🚀 Enter coordinates to fetch current AQI and predict next 5 hours.</b></h3>
            </div>
            """, unsafe_allow_html=True)
# ✅ Input
lat = st.number_input("📍 Enter Latitude:", value=12.9169, format="%.6f")
lon = st.number_input("📍 Enter Longitude:", value=77.6247, format="%.6f")

# ✅ Fetch API Key
api_key = os.getenv('api_key')
if not api_key:
    st.warning("⚠️ API Key not set in environment. Please check your .env file or Streamlit Secrets.")

# ✅ On Button Click
if st.button("🔮 Predict AQI"):
    if api_key:
        comp = get_pollutants(lat, lon, api_key)
        if comp:
            cur_hour = datetime.now().hour
            forecast = []
            for i in range(6):  # current + 5 hours
                hour = (cur_hour + i) % 24
                predicted_aqi = predict_aqi(comp, hour, model)
                forecast.append({"Hour": hour, "Predicted AQI": predicted_aqi})
            
            df_forecast = pd.DataFrame(forecast)
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 10px;'>
            <h3><b>✅ Current AQI Category: {df_forecast.iloc[0]['Predicted AQI']}</b></h3>
            </div>
            """, unsafe_allow_html=True)


            st.subheader("📊 Forecast for Next 5 Hours")
            st.dataframe(df_forecast)

            # ✅ Line Chart
            fig, ax = plt.subplots()
            ax.plot(df_forecast['Hour'], df_forecast['Predicted AQI'], marker='o', color='orange')
            ax.set_title('AQI Forecast - Next 5 Hours')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Predicted AQI')
            ax.grid(True)
            st.pyplot(fig)

        else:
            st.error("❌ Failed to fetch API data. Please check your coordinates or API Key.")
    else:
        st.warning("⚠️ Please set your API Key in environment variables.")

