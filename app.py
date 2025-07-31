import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime
import pytz
import os
import time
import folium
from folium import CircleMarker, PolyLine
from streamlit_folium import st_folium
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import openrouteservice
import matplotlib.pyplot as plt


# ✅ Set page config
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
    label[for^="text_input"] {
        display: none !important;
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

# ✅ API Keys
api_key = os.getenv('API_KEY')  # OpenWeatherMap
ors_key = os.getenv("ORS_API_KEY")  # OpenRouteService

@st.cache_data(ttl=3600)
def geocode_location(location):
    try:
        geolocator = Nominatim(user_agent="AQIApp/1.0")
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except:
        return None, None, None
    return None, None, None

def get_pollutants(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['list'][0]['components']
    else:
        return None

def predict_aqi(comp_dict, hour):
    features = [
        comp_dict['co'], comp_dict['no'], comp_dict['no2'], comp_dict['o3'],
        comp_dict['so2'], comp_dict['pm2_5'], comp_dict['pm10'], comp_dict['nh3'], hour
    ]
    return model.predict([features])[0]

def get_nearby_parks(lat, lon):
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
            plat, plon = element["lat"], element["lon"]
        elif "center" in element:
            plat, plon = element["center"]["lat"], element["center"]["lon"]
        else:
            continue
        parks.append({"name": name, "lat": plat, "lon": plon})
    return parks

def get_aqi(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    res = requests.get(url).json()
    try:
        return res["list"][0]["main"]["aqi"]
    except:
        return None

def aqi_to_color(aqi):
    if aqi is None:
        return "gray"
    if aqi <= 2:
        return "green"
    elif aqi <= 4:
        return "orange"
    else:
        return "red"

def get_routes_ors(start_coords, end_coords, ors_api_key):
    client = openrouteservice.Client(key=ors_api_key)
    try:
        route = client.directions(
            coordinates=[start_coords[::-1], end_coords[::-1]],
            profile='driving-car',
            format='geojson'
        )
        return [route['features'][0]]
    except openrouteservice.exceptions.ApiError as e:
        st.error(f"OpenRouteService Error: {e}")
        return []

def sample_route_coords(route_coords, step=10):
    return route_coords[::step] if len(route_coords) > step else route_coords

def compute_avg_aqi(coords, api_key):
    total_aqi, count = 0, 0
    for lat, lon in coords:
        time.sleep(1)
        aqi = get_aqi(lat, lon, api_key)
        if aqi is not None:
            total_aqi += aqi
            count += 1
    return total_aqi / count if count > 0 else None


# ---------- UI Layout ----------
st.markdown("""
    <div class="overlay">
        <h1>🌐 Smart AQI Assistant</h1>
        <h4>Predict AQI, find clean parks and safest routes visually 🌿</h4>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="overlay">📍 Enter your Area / City</div>', unsafe_allow_html=True)
location_input = st.text_input(label="")
lat, lon, full_addr = geocode_location(location_input)

if lat and lon:
    st.success(f"Detected: {full_addr}")

# ---------- AQI Forecast ----------
if st.button("🔮 Predict AQI for Next 5 Hours"):
    if api_key:
        pollutants = get_pollutants(lat, lon, api_key)
        if pollutants:
            local_tz = pytz.timezone('Asia/Kolkata')  # Change this to your timezone
            current_hour = datetime.now(local_tz).hour
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

# ---------- Cleanest Parks ----------
if st.button("🏞️ Find Cleanest Parks Nearby"):
    if not api_key:
        st.warning("⚠️ OpenWeatherMap API_KEY missing")
    elif lat and lon:
        with st.spinner("Fetching parks and AQI data..."):
            parks = get_nearby_parks(lat, lon)
            for park in parks:
                park["aqi"] = get_aqi(park["lat"], park["lon"], api_key)
            cleanest = sorted([p for p in parks if p["aqi"] is not None], key=lambda x: x["aqi"])[:5]
            st.subheader("🌿 Cleanest Parks")
            for p in cleanest:
                st.markdown(f"**{p['name']}** - AQI: {p['aqi']}")

            fmap = folium.Map(location=[lat, lon], zoom_start=13)
            folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="blue")).add_to(fmap)
            for p in parks:
                color = aqi_to_color(p["aqi"])
                CircleMarker(location=[p["lat"], p["lon"]], radius=7,
                             color=color, fill=True, popup=f"{p['name']} (AQI: {p['aqi']})").add_to(fmap)
            st.subheader("🗺️ Nearby Parks Map")
            st_folium(fmap, width=700, height=500)
    else:
        st.error("❌ Invalid location")

# ---------- Cleanest Route ----------
st.markdown("### 🛣️ Cleanest Route Based on AQI")
source = st.text_input("Source (e.g., MG Road, Bangalore)")
destination = st.text_input("Destination (e.g., Indiranagar, Bangalore)")

if st.button("🚦 Show Cleanest Route"):
    if not api_key or not ors_key:
        st.warning("⚠️ Set both API_KEY and ORS_API_KEY in .env")
    else:
        src_lat, src_lon, src_addr = geocode_location(source)
        dst_lat, dst_lon, dst_addr = geocode_location(destination)
        if None in (src_lat, src_lon, dst_lat, dst_lon):
            st.error("❌ Could not geocode source/destination.")
        else:
            with st.spinner("Calculating cleanest route..."):
                routes = get_routes_ors((src_lat, src_lon), (dst_lat, dst_lon), ors_key)
                route_data = []
                for route in routes:
                    coords = [(lat, lon) for lon, lat in route['geometry']['coordinates']]
                    sampled = sample_route_coords(coords)
                    avg_aqi = compute_avg_aqi(sampled, api_key)
                    if avg_aqi:
                        distance = route['properties']['summary']['distance'] / 1000
                        route_data.append((coords, avg_aqi, distance))
                if route_data:
                    clean = min(route_data, key=lambda x: x[1])
                    st.success(f"Cleanest route found!\nDistance: {clean[2]:.2f} km\nAvg AQI: {clean[1]:.2f}")
                    route_map = folium.Map(location=[(src_lat + dst_lat) / 2, (src_lon + dst_lon) / 2], zoom_start=13)
                    PolyLine(clean[0], color="blue", weight=5).add_to(route_map)
                    folium.Marker([src_lat, src_lon], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
                    folium.Marker([dst_lat, dst_lon], popup="End", icon=folium.Icon(color="red")).add_to(route_map)
                    st_folium(route_map, width=700, height=500)
                else:
                    st.error("❌ Could not compute AQI for route.")
