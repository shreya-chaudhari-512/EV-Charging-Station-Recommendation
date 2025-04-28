# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# 🎨 Page Configuration + Global Background
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EV Charger Availability Prediction",
    page_icon="🔋",
    layout="wide"
)

# --- Global Background Image + White Overlay ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
                    url('https://images.unsplash.com/photo-1549921296-3a5458a92c24?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------
# 📥 Load Dataset
# ---------------------------------------------------------------
# @st.cache_data
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
st.sidebar.title("🔋 EV Charging Recommendation Web-App")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "About"])

# ---------------------------------------------------------------
# 🏠 Home Page
# ---------------------------------------------------------------
if page == "Home":
    st.title("🔋 EV Charging Station Optimization System")
    st.markdown("""
        <h3 style='color:#4CAF50;'>Welcome to the Future of EV Infrastructure! 🚗⚡</h3>
        <p>This app helps predict whether installing a new EV charging station at a given location would be beneficial or not.</p>
        
        <hr>

        ### 🔥 How It Works:
        - 📍 **Input** latitude & longitude
        - 🚗 **Select** vehicle type (encoded)
        - ⏳ **Specify** expected charging duration
        - 🔮 **Get an Instant Prediction** whether the station would be **available** or **not**!

        ### 📈 Why it Matters:
        - Expand EV networks efficiently
        - Support green mobility revolution
        - Enhance user satisfaction with well-placed stations

        ---
    """, unsafe_allow_html=True)

    with st.expander("📊 Model Performance Metrics"):
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        st.metric(label="✅ Model Accuracy", value=f"{accuracy:.2%}")
        st.text("Detailed Classification Report:")
        st.code(report, language='text')

# 🚗 Prediction Page
elif page == "Make Prediction":
    st.title("🚗 Can You Install an EV Charging Station Here?")

    col1, col2 = st.columns(2)

    # --- 📋 Left Column: Reference Locations (Maharashtra) ---
    with col1:
        st.subheader("🛣️ Reference Locations (Maharashtra)")
        reference_data = pd.DataFrame({
            "Location": [
                "Marine Drive, Mumbai", 
                "FC Road, Pune", 
                "Nagpur Railway Station",
                "Aurangabad Caves",
                "Nashik Road"
            ],
            "Latitude": [18.9430, 18.5293, 21.1466, 19.9126, 19.9425],
            "Longitude": [72.8238, 73.8446, 79.0849, 75.8370, 73.8087]
        })
        st.table(reference_data)

    # --- 🚗 Right Column: Vehicle Type Encoding Reference ---
    with col2:
        st.subheader("📋 Vehicle Type Encoding Reference")
        vehicle_mapping = {
            0: 'Two Wheeler',
            1: 'Three Wheeler',
            2: 'Passenger Car',
            3: 'Light Commercial Vehicle',
            4: 'Heavy Commercial Vehicle'
        }
        vehicle_df = pd.DataFrame(list(vehicle_mapping.items()), columns=['Encoded Value', 'Vehicle Type'])
        st.table(vehicle_df)

    st.markdown("---")

    # --- 🔮 User Input Form for Prediction ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_latitude = st.number_input("Enter Latitude:", format="%.6f")
            user_longitude = st.number_input("Enter Longitude:", format="%.6f")
        with col2:
            user_vehicle_type = st.number_input("Enter Vehicle Type (encoded integer):", step=1, format="%d")
            user_duration = st.number_input("Enter Charging Duration (in seconds):", format="%.2f")

        submitted = st.form_submit_button("🔮 Predict")

    # --- 🎯 Prediction Result ---
    if submitted:
        try:
            user_input = np.array([[user_latitude, user_longitude, user_vehicle_type, user_duration]])
            user_prediction = knn_model.predict(user_input)

            st.subheader("🎯 Prediction Result:")
            if user_prediction[0] == 1:
                st.success("✅ Yes, you can install a station here!")
            else:
                st.error("🚫 Likely not a suitable place.")

            # 📍 Show Location on Map
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
    st.header("About Our Project")

    st.write("""
        ## 🚀 Problem Statement
        
        Electric Vehicles (EVs) are becoming increasingly popular, but inadequate charging infrastructure remains a hurdle.
        
        This project identifies the **optimal locations** for EV charging stations based on:
        - Population density
        - Traffic flow
        - Existing infrastructure
        - Power availability

        ## 🎯 Aim
        - **Strategic Placement**
        - **Data-Driven Decisions**
        - **Sustainable Growth**
        - **User Satisfaction**

        ## 🧾 Data Dictionary
        - *Latitude*: Geographical latitude
        - *Longitude*: Geographical longitude
        - *Vehicle Type (Encoded)*: 0 (Two Wheeler), 1 (Three Wheeler), etc.
        - *Duration*: Expected charging time
        - *Score*: Suitability for placing new station

        ## 📊 Key Insights
        - High-density zones + heavy traffic flow → High potential
        - Infrastructure gaps exist even in major cities
        - Longer durations common in suburban areas
        - Passenger vehicles and LCVs have highest charging needs

        ## 🛠️ Model Performance
        - **Accuracy**: ~85%
        - KNN model effective for neighbor-based spatial predictions

    """)

    st.subheader("👩‍💻 Team Members")
    st.write("""
        - Shreya Chaudhari (221061013)
        - Nithya Cherala (221061014)
    """)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """*Team Members:*  
    Shreya Chaudhari  
    Nithya Cherala"""
)
