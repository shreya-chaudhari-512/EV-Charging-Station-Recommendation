# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EV Charging Station Optimization",
    page_icon="⚡",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #ffffff, #e0f7ff);
        padding: 2rem;
        border-radius: 8px;
        color: #000080;
    }
    h1, h2, h3, h4, h5, h6, p, li, div {
        color: #000080;
        font-family: 'Segoe UI', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #ffffff, #f9f9f9);
        color: #000080;
    }
    div.stButton > button {
        background-color: white;
        color: #004080;
        border: 1px solid #004080;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #e6f0ff;
    }
    button[data-testid="baseButton-secondary"] {
        background-color: white;
        color: #004080;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #e6f0ff;
    }
    .stDataFrame {
        background-color: white;
        border: 1px solid #cce6ff;
        border-radius: 8px;
    }

    <style>
/* Make selectbox label white */
div.stSelectbox label {
    color: #ffffff;
}

/* Make inside selected value and options white */
div[data-baseweb="select"] > div {
    color: #ffffff;
}
div[data-baseweb="select"] span {
    color: #000000;
}

/* Optional: if you want the dropdown background to be dark too */
div[data-baseweb="select"] {
    background-color: #000000; /* Deep blue */
}
</style>

""", unsafe_allow_html=True)

# --- LOAD MODEL AND DATA ---
@st.cache_resource
def load_model():
    model = joblib.load('app/knn_model.pkl')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/final_cleaned.csv')
    return df

model = load_model()
data = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Home", "About", "Find Places", "Find EV Charge Point", "Dataset", "How We Made It"])

# --- HOME PAGE ---
if page == "Home":
    st.title("EV Charging Station Optimization")
    st.write("""
        Welcome to the Electric Vehicle Charging Station Optimization system. 
        This application assists in identifying optimal locations for setting up EV charging stations 
        and predicting the availability based on location and vehicle parameters.
    """)
    st.subheader("Project Objectives")
    st.write("""
    - Geospatial and demographic analysis
    - Predictive modeling using K-Nearest Neighbors (KNN)
    - Visualization of locations using an interactive map
    """)

# --- ABOUT PAGE ---
elif page == "About":
    st.title("About This Project")
    st.write("""
        This project leverages data science to optimize the planning of EV charging stations 
        by predicting demand and visualizing spatial distributions.
        
        *Features used for prediction:*
        - Latitude
        - Longitude
        - Vehicle Type (encoded)
        - Charging Duration (seconds)

        *Developed By:*
        - NITHYA CHERALA 
        - SHREYA CHAUDHARI
        
        Built using Streamlit and Scikit-Learn.
    """)

# --- FIND PLACES PAGE (MAP) ---
elif page == "Find Places":
    st.title("Find Places on Map")

    st.subheader("Charging Station Locations")
    midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=data,
                get_position='[longitude, latitude]',
                get_color='[0, 128, 255, 160]',
                get_radius=300,
            ),
        ],
    ))

# --- FIND EV CHARGE POINT PAGE (PREDICTION) ---
elif page == "Find EV Charge Point":
    st.title("Find EV Charge Point - Prediction")

    st.subheader("Input Parameters")

    with st.form(key='prediction_form'):
        latitude = st.number_input('Latitude', format="%.6f")
        longitude = st.number_input('Longitude', format="%.6f")
        vehicle_type = st.selectbox('Vehicle Type', options=[0, 1, 2, 3])
        duration = st.number_input('Charging Duration (seconds)', min_value=0)
        
        submit_button = st.form_submit_button(label='Predict Availability')

    if submit_button:
        with st.spinner('Generating Prediction...'):
            input_features = pd.DataFrame([{
                'latitude': latitude,
                'longitude': longitude,
                'vehicle_type': vehicle_type,
                'duration': duration
            }])
            prediction = model.predict(input_features)[0]
        st.success(f"Predicted Available Charging Slots: {int(prediction)}")

# --- DATASET PAGE ---
elif page == "Dataset":
    st.title("Dataset Overview")

    st.subheader("Selected Dataset Columns")
    selected_columns = ['latitude', 'longitude', 'vehicle_type', 'available']
    st.dataframe(data[selected_columns], use_container_width=True)

    st.subheader("Download Full Dataset")
    st.download_button(
        label="Download CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name='final_cleaned.csv',
        mime='text/csv'
    )

# --- HOW WE MADE IT PAGE ---
# --- HOW WE MADE IT PAGE ---
elif page == "How We Made It":
    st.title("How We Made This Project")

    st.write("""
        Our EV Charging Station Optimization project was developed through a systematic approach, combining data science and visualization techniques to create a meaningful solution.

        We began with *Data Collection*, gathering comprehensive datasets related to EV charging stations, traffic patterns, and geographical information. 

        This was followed by *Data Cleaning*, where we handled missing values, removed irrelevant attributes, and ensured consistency across datasets.

        In the *Feature Engineering* phase, categorical variables were encoded, and key features like vehicle type and charging duration were refined to enhance model performance.

        We then moved on to *Model Building*, where a K-Nearest Neighbors (KNN) model was trained to predict the availability of charging slots based on location and user parameters.

        After achieving satisfactory model performance, we focused on *Model Deployment*. We saved the trained model using Joblib and built an interactive, user-friendly web application using Streamlit.

        Finally, we emphasized *Visualization*, using Pydeck for interactive mapping and Seaborn/Matplotlib for data exploration. The result is a powerful tool that provides deep insights and predictive capabilities for EV infrastructure planning.

        ---
        ### Development Highlights
        - Streamlined data pipelines
        - Lightweight and responsive Streamlit app
        - Interactive geospatial visualizations
        - Ready for cloud deployment
    """)


















