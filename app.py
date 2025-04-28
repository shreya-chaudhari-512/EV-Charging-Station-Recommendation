# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# 🎨 Page Configuration + Global Background
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EV Charger Availability Prediction",
    page_icon="🔋",
    layout="wide"
)

# --- Apply Fonts and Button Styling ---
st.markdown(
    """
    <style>
    /* Global Font Settings */
    body, h1, h2, h3, h4, p {
        font-family: 'Poppins', sans-serif;
    }
    section[data-testid="stSidebar"] {
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    .main p {
        color: #333333;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    a {
        color: #4CAF50;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# 📥 Load and Preprocess Dataset
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('final_cleaned.csv')  # <-- correct your path here if needed

try:
    df1 = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

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

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 🧭 Sidebar Navigation
# ---------------------------------------------------------------
st.sidebar.title("🔋 EV Charging Recommendation Web-App")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Make Prediction", "About"])

# ---------------------------------------------------------------
# 🏠 Home Page
# ---------------------------------------------------------------
if page == "Home":
    st.title("EV Charging Station Optimization System")
    st.markdown("""
        <h3 style='color:#4CAF50;'>Welcome to the Future of Electric Vehicle Infrastructure! ⚡</h3>
        <p>This web app leverages data-driven insights to predict the viability of installing EV charging stations.</p>

        ### How It Works:
        - **Enter** latitude, longitude, vehicle type, and estimated charging duration
        - **Predict** feasibility instantly!

        ### Why It Matters:
        - **Accelerating EV adoption**
        - **Environment-friendly growth**
        - **Enhanced user satisfaction**
    """, unsafe_allow_html=True)

    with st.expander("Model Performance Metrics"):
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        st.text("Detailed Classification Report:")
        st.code(report, language='text')

# ---------------------------------------------------------------
# 📊 EDA Page
# ---------------------------------------------------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Cleaning for EDA
    df1_cleaned = df1.copy()
    non_numeric_columns = df1_cleaned.select_dtypes(exclude=[np.number]).columns
    df1_cleaned[non_numeric_columns] = df1_cleaned[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    df1_cleaned = df1_cleaned.fillna(df1_cleaned.mean())

    # Overview
    st.subheader("First Few Rows")
    st.write(df1_cleaned.head())

    st.subheader("Missing Data")
    st.bar_chart(df1.isnull().sum())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df1_cleaned.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Numerical Features")
    numerical_cols = df1_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df1_cleaned[col], kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

    st.subheader("Boxplots for Numerical Features")
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df1_cleaned, x=col, color="orange", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------
# 🚗 Make Prediction Page
# ---------------------------------------------------------------
elif page == "Make Prediction":
    st.title("🚗 Can You Install an EV Charging Station Here?")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛣️ Reference Locations (Maharashtra)")
        reference_data = pd.DataFrame({
            "Location": ["Marine Drive, Mumbai", "FC Road, Pune", "Nagpur Railway Station", "Aurangabad Caves", "Nashik Road"],
            "Latitude": [18.9430, 18.5293, 21.1466, 19.9126, 19.9425],
            "Longitude": [72.8238, 73.8446, 79.0849, 75.8370, 73.8087]
        })
        st.table(reference_data)

    with col2:
        st.subheader("📋 Vehicle Type Encoding Reference")
        vehicle_mapping = {
            0: 'Two Wheeler',
            1: 'Three Wheeler',
            2: 'Passenger Car',
            3: 'Light Commercial Vehicle',
            4: 'Heavy Commercial Vehicle'
        }
        st.table(pd.DataFrame(vehicle_mapping.items(), columns=['Encoded Value', 'Vehicle Type']))

    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_lat = st.number_input("Enter Latitude:", format="%.6f")
            user_lon = st.number_input("Enter Longitude:", format="%.6f")
        with col2:
            user_vehicle = st.number_input("Enter Vehicle Type (encoded integer):", step=1, format="%d")
            user_duration = st.number_input("Enter Charging Duration (in seconds):", format="%.2f")

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        try:
            user_input = np.array([[user_lat, user_lon, user_vehicle, user_duration]])
            prediction = knn_model.predict(user_input)

            st.subheader("🎯 Prediction Result:")
            if prediction[0] == 1:
                st.success("✅ Yes, you can install a station here!")
            else:
                st.error("🚫 Likely not a suitable place.")

            st.subheader("📍 Location on Map:")
            st.map(pd.DataFrame({'latitude': [user_lat], 'longitude': [user_lon]}))
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------------------------------------------------------
# ℹ️ About Page
# ---------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About Our EV Charging Station Optimization Project")
    st.write("""
        ## Problem Statement
        We aim to address the demand-supply gap in EV infrastructure using data-driven approaches.

        ## Goals:
        - Strategic placement of charging stations
        - Enhance EV adoption and accessibility

        ## Data Dictionary:
        - Latitude, Longitude
        - Vehicle Type (encoded)
        - Charging Duration
        - Target: Station Install Feasibility

        ## Key Insights:
        - High-density zones show best potential
        - Passenger cars & LCVs dominate charging needs
    """)

    st.subheader("Team Members")
    st.write("""
        - Shreya Chaudhari (221061013)
        - Nithya Cherala (221061014)
    """)
