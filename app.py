# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 🎨 Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EV Charger Availability Prediction",
    page_icon="🔋",
    layout="wide"
)

# ---------------------------------------------------------------
# 📥 Load Dataset
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('final_cleaned.csv')  # <- your correct path!

try:
    df1 = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ---------------------------------------------------------------
# 🛠️ Data Preprocessing
# ---------------------------------------------------------------
def preprocess_data(df):
    cols_to_drop = ['uid', 'name', 'vendor_name', 'address', 'city', 'country',
                    'open', 'close', 'logo_url', 'payment_modes', 'contact_numbers']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    cols_to_encode = ['power_type', 'type', 'vehicle_type', 'zone', 'station_type', 'staff']
    le = LabelEncoder()
    for col in cols_to_encode:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    def clean_cost(x):
        if isinstance(x, str):
            x = x.replace('₹', '').replace('per unit', '').strip()
        try:
            return float(x)
        except:
            return 0.0

    if 'cost_per_unit' in df.columns:
        df['cost_per_unit'] = df['cost_per_unit'].apply(clean_cost)

    def clean_duration(x):
        if isinstance(x, str) and 'days' in x:
            return pd.to_timedelta(x).total_seconds()
        try:
            return float(x)
        except:
            return 0.0

    if 'duration' in df.columns:
        df['duration'] = df['duration'].apply(clean_duration)

    return df

# Preprocess
df1_model = preprocess_data(df1)

# Feature Selection
selected_features = ['latitude', 'longitude', 'vehicle_type', 'duration']
X = df1_model[selected_features]
y = df1_model['available']

# Remove NaN
valid_rows = ~y.isna()
X = X[valid_rows]
y = y[valid_rows]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 🧭 Sidebar Navigation
# ---------------------------------------------------------------
st.sidebar.title("🔋 EV Charging App")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "About"])

# ---------------------------------------------------------------
# 🏠 Home Page
# ---------------------------------------------------------------
if page == "Home":
    st.title("🔋 EV Charging Station Optimization System")
    st.markdown("""
        Welcome to the **EV Charging Station Optimization System**! 🚗⚡  
        This app helps predict whether installing a new EV charging station at a given location would be beneficial or not.
        
        ### How it works:
        - 📍 Input latitude & longitude
        - 🚗 Select vehicle type (encoded)
        - ⏳ Specify expected charging duration
        - 🔮 Get an instant prediction whether the station would be **available** or **not**!

        ---
    """)
    with st.expander("🔎 Model Performance"):
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        st.text("Classification Report:")
        st.code(report, language='text')

# ---------------------------------------------------------------
# 🚗 Prediction Page
# ---------------------------------------------------------------
elif page == "Make Prediction":
    st.title("🚗 Predict EV Charger Availability")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_latitude = st.number_input("Enter Latitude:", format="%.6f")
            user_longitude = st.number_input("Enter Longitude:", format="%.6f")
        with col2:
            user_vehicle_type = st.number_input("Enter Vehicle Type (encoded integer):", step=1, format="%d")
            user_duration = st.number_input("Enter Charging Duration (in seconds):", format="%.2f")

        submitted = st.form_submit_button("Predict Availability 🚀")

    if submitted:
        try:
            user_input = np.array([[user_latitude, user_longitude, user_vehicle_type, user_duration]])
            user_prediction = knn_model.predict(user_input)

            st.subheader("🎯 Prediction Result:")
            if user_prediction[0] == 1:
                st.success("✅ Likely Available for EV Charging!")
            else:
                st.error("🚫 Likely NOT Available for EV Charging.")

            # 🎯 Show on a Map
            st.subheader("📍 Location on Map:")
            map_data = pd.DataFrame({
                'latitude': [user_latitude],
                'longitude': [user_longitude]
            })
            st.map(map_data)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------------------------------------------------------
# ℹ️ About Page
# ---------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
        This project is developed to assist urban planners and EV companies in optimizing the location of EV Charging Stations. 
        
        **Key Features:**
        - Location-based prediction
        - Vehicle-type wise adjustment
        - KNN-based machine learning model
        - Simple, user-friendly interface

        Developed with ❤️ by [Your Name].
        
        GitHub Repo: [🔗 Click Here](https://github.com/shreya-chaudhari-512/EV-Charging-Station-Optimization)

        ---
    """)
