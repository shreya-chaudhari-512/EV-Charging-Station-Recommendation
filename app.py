# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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

    /* Sidebar Font */
    section[data-testid="stSidebar"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Button Styling (Rounded + Shadow) */
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }

    /* Button Hover Effects */
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }

    /* Text for Sections */
    .main p {
        color: #333333;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Info Boxes (Card-Like Design) */
    .st-expanderHeader {
        background-color: #f4f4f4;
        color: #333333;
        font-weight: bold;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }

    /* Enhance Links */
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
page = st.sidebar.radio("Go to", ["Home", "EDA" , "Make Prediction", "About"])

# ---------------------------------------------------------------
# 🏠 Home Page
# ---------------------------------------------------------------
if page == "Home":
    st.title("EV Charging Station Optimization System")
    st.markdown("""
        <h3 style='color:#4CAF50;'>Welcome to the Future of Electric Vehicle Infrastructure! ⚡</h3>
        <p>This web app leverages data-driven insights to predict the viability of installing EV charging stations in your area. By analyzing key factors like population density, traffic patterns, and existing infrastructure, we empower decision-makers to expand the EV network in the most optimal locations.</p>

        <hr>

        ### How It Works:
        -  **Enter** latitude and longitude of a location
        -  **Select** the vehicle type (encoded for simplicity)
        -  **Provide** an estimated charging duration
        -  **Instant Prediction** on whether installing a station is feasible or not

        ### Why It Matters:
        - **Accelerating the EV transition** with well-placed stations
        - **Minimizing environmental impact** by improving green mobility
        - **Improving user satisfaction** through better accessibility

        ---
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
if page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Show the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.write(df1.head())

    # Show dataset info (column types, non-null values)
    st.subheader("Data Summary")
    st.write(df1.describe())

    # Missing Data Heatmap
    st.subheader("Missing Data Visualization")
    missing_data = df1.isnull().sum()
    st.bar_chart(missing_data)

    # Show a correlation heatmap between numeric columns
    st.subheader("Correlation Heatmap")
    correlation_matrix = df1.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Show Distribution of `vehicle_type`
    st.subheader("Distribution of Vehicle Types")
    vehicle_type_dist = df1['vehicle_type'].value_counts()
    fig = px.pie(values=vehicle_type_dist, names=vehicle_type_dist.index, title="Vehicle Type Distribution")
    st.plotly_chart(fig)

    # Distribution of `duration`
    st.subheader("Charging Duration Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df1['duration'], kde=True, ax=ax)
    st.pyplot(fig)

    # Show Target Distribution (Install/Don't Install)
    st.subheader("Target Variable Distribution (Install/Don't Install)")
    install_dist = df1['install'].value_counts()
    fig = px.pie(values=install_dist, names=install_dist.index, title="Install/Don't Install Distribution")
    st.plotly_chart(fig)

    # Boxplot of Charging Duration by Vehicle Type
    st.subheader("Charging Duration by Vehicle Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='vehicle_type', y='duration', data=df1, ax=ax)
    st.pyplot(fig)

    # Label Encoding Process
    st.subheader("Label Encoding Process")
    st.write("For categorical columns like `vehicle_type`, `power_type`, etc., label encoding was applied to transform them into numerical values. Here's an example for `vehicle_type`:")
    st.write(df1[['vehicle_type']].head())

# 🚗 Prediction Page
elif page == "Make Prediction":
    st.title("Can You Install an EV Charging Station Here?")

    col1, col2 = st.columns(2)

    # --- 📋 Left Column: Reference Locations (Maharashtra) ---
    with col1:
        st.subheader("Reference Locations (Maharashtra)")
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
        st.subheader("Vehicle Type Encoding Reference")
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

        submitted = st.form_submit_button("Predict")

    # --- 🎯 Prediction Result ---
    if submitted:
        try:
            user_input = np.array([[user_latitude, user_longitude, user_vehicle_type, user_duration]])
            user_prediction = knn_model.predict(user_input)

            st.subheader("Prediction Result:")
            if user_prediction[0] == 1:
                st.success("Yes, you can install a station here!")
            else:
                st.error("Likely not a suitable place.")

            # 📍 Show Location on Map
            st.subheader("Location on Map:")
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
    st.header("About Our EV Charging Station Optimization Project")

    st.write("""
        ## Problem Statement
        
        The rise of electric vehicles (EVs) is creating a massive demand for more charging infrastructure. Yet, inadequate and poorly located charging stations remain a significant challenge for EV adoption. Our project uses advanced data science techniques to identify **optimal locations** for new EV charging stations based on the following criteria:
        - **Population density** in the surrounding area
        - **Traffic flow** for higher station usage
        - **Existing infrastructure** for easier integration
        - **Power availability** for efficient operation

        ## Project Goals:
        - **Strategic Placement** of EV stations for high usage and efficiency
        - **Data-Driven Decisions** based on real-world factors
        - **Sustainable Growth** of EV networks
        - **Enhanced User Experience** with well-placed, accessible stations

        ## Data Dictionary:
        - *Latitude*: The geographical latitude of the location
        - *Longitude*: The geographical longitude of the location
        - *Vehicle Type (Encoded)*: 0 (Two-Wheeler), 1 (Three-Wheeler), 2 (Passenger Car), etc.
        - *Duration*: Estimated time a vehicle will need to charge at the station
        - *Availability*: Whether installing a station at this location is feasible (1) or not (0)

        ## Key Insights:
        - **High-density zones** with heavy traffic flow show the greatest potential for new stations.
        - **Suburban areas** tend to require more infrastructure to meet the increasing demand.
        - Passenger vehicles and light commercial vehicles (LCVs) have the **highest demand** for charging.

        ## Model Overview:
        - **Accuracy**: ~91% for the KNN model, effective for neighbor-based predictions in spatial data.
    """)

    st.subheader("Team Members")
    st.write("""
        - Shreya Chaudhari (221061013)
        - Nithya Cherala (221061014)
    """)
