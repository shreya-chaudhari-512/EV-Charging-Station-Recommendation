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
st.sidebar.title("EV Charging Recommendation Web-App")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Make Prediction", "About"])

# ---------------------------------------------------------------
# 🏠 Home Page
# ---------------------------------------------------------------
if page == "Home":
    st.title("EV Charging Station Optimization System")
    st.markdown("""
        <h3 style='color:#4CAF50;'>Welcome to the Future of Electric Vehicle Infrastructure! ⚡</h3>
        <p style='font-size:16px;'>This intelligent system leverages <b>machine learning</b> to recommend the best locations for EV charging station installations based on real-world data.</p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### How It Works")
    st.markdown("""
    - 🔍 **Input**: Enter the latitude, longitude, vehicle type, and estimated charging duration.
    - 🧠 **Predict**: Our KNN-based model predicts the feasibility of installing an EV charger at that location.
    - 📈 **Decision Support**: Helps urban planners and businesses make smarter infrastructure investments.
    """)

    st.divider()

    st.markdown("### Why It Matters")
    st.success("""
    - Accelerates electric vehicle adoption across cities.
    - Promotes sustainable and eco-friendly transportation.
    - Reduces "range anxiety" among EV users.
    - Optimizes public and private investments.
    """)

    st.divider()

    with st.expander("Model Performance Metrics"):
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.metric(label=" Model Accuracy", value=f"{accuracy:.2%}")
        st.text("Detailed Classification Report:")
        st.code(report, language='text')

    st.divider()

    st.markdown("### Try It Out!")
    st.info("""
    Ready to explore?  
    ➡ Head over to the **Predict** page and test different coordinates and vehicle types!  
    See where the future of EV charging is headed.
    """)

    st.balloons()  # 🎈 small animation when user opens the home page


# ---------------------------------------------------------------
# 📊 EDA Page
# ---------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Cleaning for EDA
    df1_cleaned = df1.copy()
    non_numeric_columns = df1_cleaned.select_dtypes(exclude=[np.number]).columns
    df1_cleaned[non_numeric_columns] = df1_cleaned[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    df1_cleaned = df1_cleaned.fillna(df1_cleaned.mean())

    # Overview
    st.subheader("🧹 First Few Rows of Cleaned Data")
    st.write(df1_cleaned.head())

    st.divider()

    st.subheader("🛠️ Missing Data Overview")
    st.bar_chart(df1.isnull().sum())

    st.divider()

    st.subheader("🔗 Top Feature Correlations")

    # Calculate correlations
    corr_matrix = df1_cleaned.corr()

    # Take the absolute value and unstack
    corr_pairs = corr_matrix.abs().unstack()

    # Remove self-correlations
    corr_pairs = corr_pairs[corr_pairs != 1]

    # Sort correlations and take the top 10
    top_corr = corr_pairs.sort_values(ascending=False).drop_duplicates().head(10)

    # Plot the top correlated feature pairs
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df1_cleaned[top_corr.index.get_level_values(0).unique()].corr(),
                annot=True, cmap="YlGnBu", linewidths=0.5, cbar_kws={"shrink": 0.75}, square=True, fmt=".2f")
    ax.set_title("Top Correlated Features Heatmap", fontsize=16, fontweight='bold')
    st.pyplot(fig)

    st.info("✅ Only the strongest feature correlations are shown to avoid clutter!")

# ---------------------------------------------------------------
# 🚗 Make Prediction Page
# ---------------------------------------------------------------
elif page == "Make Prediction":
    st.title("Can You Install an EV Charging Station Here?")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Locations (Maharashtra)")
        reference_data = pd.DataFrame({
            "Location": ["Marine Drive, Mumbai", "FC Road, Pune", "Nagpur Railway Station", "Aurangabad Caves", "Nashik Road"],
            "Latitude": [18.9430, 18.5293, 21.1466, 19.9126, 19.9425],
            "Longitude": [72.8238, 73.8446, 79.0849, 75.8370, 73.8087]
        })
        st.table(reference_data)

    with col2:
        st.subheader("Vehicle Type Encoding Reference")
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

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.success("Yes, you can install a station here!")
            else:
                st.error("Likely not a suitable place.")

            st.subheader("Location on Map:")
            st.map(pd.DataFrame({'latitude': [user_lat], 'longitude': [user_lon]}))
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------------------------------------------------------
# ℹ️ About Page
# ---------------------------------------------------------------

elif page == "About":
    st.title("About Our EV Charging Station Optimization Project")

    st.write("""
    ## Problem Statement
    With the global push towards electric vehicle (EV) adoption, the lack of accessible and well-distributed EV charging infrastructure presents a significant bottleneck. Our project aims to close the demand-supply gap by identifying optimal locations for installing new EV charging stations through data-driven analysis and machine learning techniques.

    ## Project Objectives
    - **Strategic Placement:** Recommend the best locations for installing EV charging stations based on multiple factors including vehicle density, traffic patterns, and existing infrastructure.
    - **Promote EV Adoption:** Support the transition to electric vehicles by reducing range anxiety and improving charging accessibility.
    - **Efficient Resource Allocation:** Help governments, municipalities, and private companies invest wisely in EV infrastructure.

    ## Data Dictionary
    Our model was trained on a cleaned and processed dataset with the following key features:
    - **Latitude** *(float)*: Geographical latitude of the potential installation site.
    - **Longitude** *(float)*: Geographical longitude of the site.
    - **Vehicle Type** *(integer encoded)*: Encoded values representing different categories of vehicles (e.g., Passenger Car, Light Commercial Vehicle).
    - **Expected Charging Duration** *(float)*: Anticipated duration (in hours) an EV would occupy the charging station.
    - **Target (Install/Don't Install)** *(binary)*: Model output (1 = Feasible for Installation, 0 = Not Feasible).

    ## Methodology Overview
    - **Data Preprocessing:** Label encoding, feature scaling, and cleaning were performed.
    - **Model Selection:** K-Nearest Neighbors (KNN) was chosen after comparing multiple models due to its simplicity, robustness, and strong performance with spatial data.
    - **Model Evaluation:** K-Fold Cross-Validation and extensive testing ensured model reliability.
    - **Insights Extraction:** Areas with high traffic flow and passenger vehicle dominance emerged as prime locations.

    ## Key Insights
    - **High-Density Zones:** Urban areas with dense vehicle presence offer the highest ROI for installing new charging stations.
    - **Dominant Vehicle Types:** Passenger cars and Light Commercial Vehicles (LCVs) account for the majority of predicted charging demand.
    - **Charging Duration Patterns:** Most users prefer medium to short-duration charging, influencing the type of chargers to be deployed.

    ## Future Enhancements
    - Integration of real-time traffic and power grid data.
    - Adoption of advanced models like Random Forest or XGBoost for improved prediction.
    - Dynamic recommendation system based on live data feeds.

    ## Technologies Used
    - Python (Pandas, NumPy, Scikit-learn)
    - Streamlit (for frontend deployment)
    - Machine Learning (KNN classifier)
    - Data Visualization (Matplotlib, Seaborn)

    ---
    """)

    st.subheader("Meet Our Team")
    st.write("""
    **Project Contributors:**
    - Shreya Chaudhari (221061013)
    - Nithya Cherala (221061014)

    We are passionate about using data science and machine learning to solve real-world challenges and contribute to the future of sustainable transportation.
    """)

