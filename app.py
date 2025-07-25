import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

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
google_api_key = os.getenv("GOOGLE_API_KEY")

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


def get_coordinates(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    headers = {"User-Agent": "AQIApp/1.0"}
    response = requests.get(url, headers=headers)
    try:
        res = response.json()
        if res:
            lat = float(res[0]["lat"])
            lon = float(res[0]["lon"])
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None, None



# Function to get nearby parks using Overpass API
def get_nearby_parks(lat, lon):
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          node["leisure"="park"](around:3000,{lat},{lon});
          way["leisure"="park"](around:3000,{lat},{lon});
          relation["leisure"="park"](around:3000,{lat},{lon});
        );
        out center;
        """
        response = requests.post(overpass_url, data=overpass_query, headers={"User-Agent": "AQIApp/1.0"})
        data = response.json()
        parks = []

        for element in data["elements"]:
            name = element["tags"].get("name", "Unnamed Park")
            if "lat" in element and "lon" in element:
                lat, lon = element["lat"], element["lon"]
            elif "center" in element:
                lat, lon = element["center"]["lat"], element["center"]["lon"]
            else:
                continue
            parks.append({"name": name, "lat": lat, "lon": lon})

        return parks
    except Exception as e:
        st.error(f"Error fetching parks: {e}")
        return []


# Function to get AQI from OpenWeatherMap
def get_aqi(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    res = requests.get(url).json()
    try:
        aqi = res["list"][0]["main"]["aqi"]
        return aqi
    except:
        return None


# Function to display parks with the best AQI
def display_cleanest_parks(parks):
    sorted_parks = sorted([p for p in parks if p["aqi"] is not None], key=lambda x: x["aqi"])
    st.subheader("🌿 Cleanest Parks Nearby (Based on AQI)")
    for park in sorted_parks[:5]:  # top 5 parks with lowest AQI
        st.markdown(f"**{park['name']}** - AQI Level: {park['aqi']}  \nLocation: ({park['lat']:.4f}, {park['lon']:.4f})")


# ✅ App Header
st.markdown("""
    <div class="overlay">
        <h1>🌐 Smart AQI Assistant</h1>
        <h4>🚀 Enter your location (City, Area):.</h4>
    </div>
""", unsafe_allow_html=True)


city = st.text_input("Enter City:")
area = st.text_input("Enter Area:")
if city and area:
    location_input = f"{area}, {city}"
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(location_input, timeout=10)

    if location:
        lat, lon = location.latitude, location.longitude
        st.success(f"📍 Location detected: {location.address}")
# ✅ Input Section

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

location = st.text_input("Enter your Area or City (e.g., Jayanagar, Bangalore)")

if st.button("🏞️ Find Cleanest Parks Nearby"):
    if location.strip() == "":
        st.warning("⚠️ Please enter a valid location.")
    else:
        with st.spinner("🔍 Fetching parks and air quality data..."):
            lat, lon = get_coordinates(location)
            if lat is None or lon is None:
                st.error("❌ Failed to find location coordinates. Please check the area name.")
            else:
                parks = get_nearby_parks(lat, lon)
                for park in parks:
                    park["aqi"] = get_aqi(park["lat"], park["lon"], api_key)
                if parks:
                    display_cleanest_parks(parks)
                else:
                    st.info("ℹ️ No parks found nearby.")