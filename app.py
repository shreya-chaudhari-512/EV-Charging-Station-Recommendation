# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EV Charger Availability Prediction", layout="centered")

st.title("⚡ EV Charging Station - Availability Prediction")

# Step 0: Load your dataset
try:
    df1 = pd.read_csv('final_cleaned.csv')  # <-- Correct relative path
    st.success("✅ Data Loaded Successfully.")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Step 1: Drop unnecessary text-heavy columns
cols_to_drop = ['uid', 'name', 'vendor_name', 'address', 'city', 'country',
                'open', 'close', 'logo_url', 'payment_modes', 'contact_numbers']
df1_model = df1.drop(columns=cols_to_drop, errors='ignore')

# Step 2: Encode categorical columns
cols_to_encode = ['power_type', 'type', 'vehicle_type', 'zone', 'station_type', 'staff']
le = LabelEncoder()
for col in cols_to_encode:
    if col in df1_model.columns:
        df1_model[col] = le.fit_transform(df1_model[col].astype(str))

# Step 3: Clean 'cost_per_unit' column
def clean_cost(x):
    if isinstance(x, str):
        x = x.replace('₹', '').replace('per unit', '').strip()
    try:
        return float(x)
    except:
        return 0.0

if 'cost_per_unit' in df1_model.columns:
    df1_model['cost_per_unit'] = df1_model['cost_per_unit'].apply(clean_cost)

# Step 4: Clean 'duration' column
def clean_duration(x):
    if isinstance(x, str) and 'days' in x:
        return pd.to_timedelta(x).total_seconds()
    try:
        return float(x)
    except:
        return 0.0

if 'duration' in df1_model.columns:
    df1_model['duration'] = df1_model['duration'].apply(clean_duration)

# Step 5: Define X (features) and y (target)
selected_features = ['latitude', 'longitude', 'vehicle_type', 'duration']
X = df1_model[selected_features]

# Target variable
y = df1_model['available']

# Step 6: Drop rows where y is NaN
valid_rows = ~y.isna()
X = X[valid_rows]
y = y[valid_rows]

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Train KNN Model
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred_knn = knn_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_knn)
report = classification_report(y_test, y_pred_knn)

st.subheader("🔎 Model Evaluation Results")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.text("Classification Report:")
st.text(report)

# Step 10: Prediction Form
st.subheader("🚗 Make a Prediction")

with st.form("prediction_form"):
    user_latitude = st.number_input("Enter Latitude:", format="%.6f")
    user_longitude = st.number_input("Enter Longitude:", format="%.6f")
    user_vehicle_type = st.number_input("Enter Vehicle Type (encoded integer):", step=1, format="%d")
    user_duration = st.number_input("Enter Duration (in seconds):", format="%.2f")

    submitted = st.form_submit_button("Predict Availability")

if submitted:
    try:
        user_input = np.array([[user_latitude, user_longitude, user_vehicle_type, user_duration]])
        user_prediction = knn_clf.predict(user_input)

        if user_prediction[0] == 1:
            st.success("✅ Prediction: Charging Station Likely Available")
        else:
            st.error("🚫 Prediction: Charging Station Likely NOT Available")
    except Exception as e:
        st.error(f"Prediction error: {e}")
