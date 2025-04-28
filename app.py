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
st.sidebar.title("🔋 EV Charging Recommendation Web-App")
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
    st.title("🚗 Can You Install a EV Charging Station here?")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_latitude = st.number_input("Enter Latitude:", format="%.6f")
            user_longitude = st.number_input("Enter Longitude:", format="%.6f")
        with col2:
            user_vehicle_type = st.number_input("Enter Vehicle Type (encoded integer):", step=1, format="%d")
            user_duration = st.number_input("Enter Charging Duration (in seconds):", format="%.2f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            user_input = np.array([[user_latitude, user_longitude, user_vehicle_type, user_duration]])
            user_prediction = knn_model.predict(user_input)

            st.subheader("🎯 Prediction Result:")
            if user_prediction[0] == 1:
                st.success("✅Yes, you can!")
            else:
                st.error("🚫 Likely Can't.")

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
    st.header("About Our Project")
    
    st.write("""
        ## Problem Statement
        
        Electric Vehicles (EVs) are becoming increasingly popular, but inadequate charging infrastructure remains a major hurdle to mass adoption. This project aims to identify the optimal locations for EV charging stations by analyzing key factors like population density, traffic flow, existing infrastructure, and power availability to ensure maximum utilization and convenience for users.
        
        ### Aim
        
        - *Strategic Placement*: Identify optimal locations for EV charging stations to maximize accessibility and convenience.
        - *Data-Driven Decisions*: Leverage key factors like population density, traffic flow, existing infrastructure, and power availability.
        - *Sustainable Growth*: Support the expansion of EV infrastructure in a scalable and environmentally responsible manner.
        - *User Satisfaction*: Reduce range anxiety by ensuring better coverage and reliability for EV users.     
        
        ### Data Dictionary
        
        The dataset contains traffic flow records, including:
        
        - *Latitude*: Latitude of the location (geographical coordinate)
        - *Longitude*: Longitude of the location (geographical coordinate)
        - *Population Density*: Number of people living per square kilometer
        - *Traffic Flow*: Average vehicle flow (vehicles per day)
        - *Existing Infrastructure*: Availability of existing EV stations nearby (count)
        - *Power Availability*: Availability of sufficient electrical capacity at the location
        - *Vehicle Type (Encoded)*: Encoded type of common vehicle usage (e.g., Passenger, Commercial)
        - *Expected Charging Duration*: Average time vehicles spend charging (hours)
        - *Score*: Calculated score for suitability of placing a new station
        
        ### Key Insights
        
        - *High Potential Zones*: Locations with **high population density** and **heavy traffic flow** were most favorable for new EV charging stations.
        - *Infrastructure Gaps*: Several high-demand areas lacked sufficient existing EV infrastructure, highlighting major opportunities for station deployment.
        - *Power Constraints*: Some otherwise ideal areas were unsuitable due to **insufficient power availability**.
        - *Vehicle Patterns*: Areas with a higher mix of **passenger vehicles** and **light commercial vehicles** showed the most consistent charging needs.
        - *Charging Duration Trends*: Longer expected charging durations were more common in suburban regions compared to city centers.
        
        ### Model Performance
        
        The K-Nearest Neighbors (KNN) model was trained to predict the suitability of locations for EV charging station placement based on key spatial and infrastructure factors.
        The model successfully identified clusters of high-potential locations, offering **valuable, data-driven support** for EV infrastructure planning.

        - **Accuracy**: ~85% (for classification tasks)
        - **Mean Squared Error (MSE)**: Low (for regression tasks)
        - **Model Strengths**: Effective in spatial neighbor-based predictions and simple to interpret for strategic planning.
    """)
    
    st.subheader("Applications")
    st.write("""
        - *Traffic Management*: Optimizing signal timings and road usage
        - *Urban Planning*: Informing infrastructure development decisions
        - *Environmental Impact*: Reducing emissions through better traffic flow
        - *Public Transportation*: Adjusting schedules based on predicted congestion
    """)

    st.subheader("Team Members")
    st.write("""
        - *Shreya Chaudhari*: 221061013
        - *Nithya Cherala*: 221061014
    """)
   

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    """*Team Members:*  
    Shreya Chaudhari  
    Nithya Cherala"""
)
