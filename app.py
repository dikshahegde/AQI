import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')

# ✅ Load Trained Model
@st.cache_resource
def load_model():
    with open('aqi_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ✅ Get Pollutants from OpenWeather
def get_pollutants(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['list'][0]['components']
        else:
            st.error(f"❌ API Error! Status: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

# ✅ Predict AQI using the model
def predict_aqi(comp_dict, hour, model):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

# ✅ Streamlit UI
st.title("🌐 Real-Time AQI Analyzer & 5-Hour Forecast (with Secure API Key)")

lat = st.number_input("📍 Enter Latitude:", value=12.9169, format="%.6f")
lon = st.number_input("📍 Enter Longitude:", value=77.6247, format="%.6f")

if st.button("🔮 Predict Current AQI & Forecast"):
    if API_KEY:
        comp = get_pollutants(lat, lon)
        if comp:
            cur_hour = datetime.now().hour

            st.success(f"✅ AQI Prediction for Hour {cur_hour}:00")
            current_aqi = predict_aqi(comp, cur_hour, model)
            st.markdown(f"### 🌟 **Current AQI Category: {current_aqi}**")

            st.subheader("📈 Forecast for Next 5 Hours")
            forecast_data = []
            for i in range(1, 6):
                future_hour = (cur_hour + i) % 24
                future_aqi = predict_aqi(comp, future_hour, model)
                forecast_data.append({'Hour': f"+{i} (Hour {future_hour})", 'Predicted AQI': future_aqi})

            forecast_df = pd.DataFrame(forecast_data)
            st.table(forecast_df)

            # ✅ Plotting the Forecast
            fig, ax = plt.subplots()
            ax.plot([f"+{i}" for i in range(1, 6)], forecast_df['Predicted AQI'], marker='o')
            ax.set_xlabel('Next Hours')
            ax.set_ylabel('Predicted AQI')
            ax.set_title('AQI Forecast for Next 5 Hours')
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Could not fetch pollutant data.")
    else:
        st.warning("⚠️ API Key not set in environment. Please check your .env file.")
