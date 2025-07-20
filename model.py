import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# ✅ Step 1 — Load Data
df = pd.read_csv('aqi_data_fetched1.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['hour'] = df['datetime'].dt.hour

# ✅ Step 2 — Prepare Features & Target
features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'hour']
X = df[features]
y = df['aqi'].astype(int)


# ✅ Step 3 — Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ✅ Step 4 — Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 5 — Make Predictions
y_pred = model.predict(X_test)

# ✅ Step 6 — Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%\n")

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Step 7 — Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Step 8 — Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette='viridis')
plt.title("Feature Importance in Random Forest")
plt.show()

# ✅ Step 9 — Save the Trained Model
with open('aqi_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained, evaluated & saved successfully.")


scores = cross_val_score(model, X, y, cv=5)
print(f"✅ Cross-Validation Accuracy: {scores.mean()*100:.2f}%")
