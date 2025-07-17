
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('aqi_data_fetched1.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['hour'] = df['datetime'].dt.hour

features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'hour']
X = df[features]
y = df['aqi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model using pickle
with open('aqi_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained & saved successfully.")
