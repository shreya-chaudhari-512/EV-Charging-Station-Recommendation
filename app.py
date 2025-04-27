# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Set page configuration
st.set_page_config(
    page_title="EV Charging Station Optimization",
    page_icon="🔋",
    layout="wide"
)

# Application title
st.title("🔋 EV Charging Station Optimization System")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Exploration", "Make Prediction", "About"])

# Sidebar info
st.sidebar.info(
    "This project helps predict whether an EV charging station should be installed "
    "based on geographic and vehicle data.\n\nBuilt with ❤️ using Python, Streamlit, and Scikit-learn."
)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Process dataset
@st.cache_data
def process_data(df):
    try:
        le_vehicle_type = LabelEncoder()
        df['vehicle_type'] = le_vehicle_type.fit_transform(df['vehicle_type'])
        df['target'] = df['target'].map({'Install': 1, 'Don\'t Install': 0})
        df.dropna(inplace=True)
        return df, le_vehicle_type
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return df, None

# Train and return model
@st.cache_resource
def get_model(df):
    try:
        X = df[['latitude', 'longitude', 'vehicle_type', 'duration']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        return model, scaler, accuracy
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

# Load and prepare data
df = load_data()
if df is not None:
    df, le_vehicle_type = process_data(df)
    model, scaler, model_accuracy = get_model(df)

# Define prediction function
def make_prediction(model, scaler, le_vehicle_type):
    st.subheader("⚡ Enter Charging Station Details")

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=19.0760, format="%.6f")
        longitude = st.number_input("Longitude", value=72.8777, format="%.6f")
        vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Bike', 'Truck', 'Bus'])
        encoded_vehicle_type = le_vehicle_type.transform([vehicle_type])[0]

    with col2:
        duration = st.number_input("Expected Charging Duration (minutes)", min_value=5, max_value=720, value=30, step=5)

    input_data = np.array([[latitude, longitude, encoded_vehicle_type, duration]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.success("✅ Prediction: Install EV Charging Station at this location!")
    else:
        st.warning("⚠️ Prediction: Not Recommended to Install EV Charging Station.")

# Page routing
if page == "Home":
    st.header("🏠 Welcome")
    st.write("""
        This application predicts whether an EV charging station should be installed 
        at a given location based on various parameters like latitude, longitude, 
        vehicle type, and charging duration.

        The aim is to help urban planners and private companies optimize EV infrastructure!
    """)
    
    if df is not None:
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Data Points", df.shape[0])
            st.metric("Average Latitude", round(df['latitude'].mean(), 4))
            st.metric("Average Longitude", round(df['longitude'].mean(), 4))

        with col2:
            st.metric("Vehicle Type Average Code", int(df['vehicle_type'].mean()))
            st.metric("Average Charging Duration (min)", int(df['duration'].mean()))
            st.metric("Install EV Charger Ratio", round(df['target'].mean(), 2))

        st.dataframe(df.head())

        if model_accuracy:
            st.success(f"Model Accuracy on Test Set: {model_accuracy*100:.2f}%")
    else:
        st.error("Dataset not available. Please upload 'final_cleaned.csv'.")

elif page == "Data Exploration":
    st.header("📈 Data Exploration")

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Vehicle Type Distribution", "Duration Analysis", "Geospatial Analysis"])

        with tab1:
            st.subheader("Vehicle Type Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='vehicle_type', data=df, ax=ax)
            ax.set_xlabel("Vehicle Type (Encoded)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with tab2:
            st.subheader("Duration by Vehicle Type")
            fig, ax = plt.subplots()
            sns.boxplot(x='vehicle_type', y='duration', data=df, ax=ax)
            ax.set_xlabel("Vehicle Type (Encoded)")
            ax.set_ylabel("Charging Duration (minutes)")
            st.pyplot(fig)

        with tab3:
            st.subheader("Location Scatter Plot")
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(x='longitude', y='latitude', hue='target', palette='coolwarm', data=df, ax=ax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(title='Install Station (1=Yes, 0=No)')
            st.pyplot(fig)
    else:
        st.error("Dataset not available to explore.")

elif page == "Make Prediction":
    if model is not None and scaler is not None and le_vehicle_type is not None:
        make_prediction(model, scaler, le_vehicle_type)
    else:
        st.error("Model not available. Please check if dataset and model are properly loaded.")

elif page == "About":
    st.header("📖 About This Project")
    st.write("""
        This system is developed to aid in the strategic installation of Electric Vehicle (EV) charging stations. 
        By using machine learning models, we can predict where stations are most needed based on 
        geographic and vehicle usage data.

        **Built With:**  
        - Python 🐍
        - Streamlit 🌟
        - Scikit-learn ⚡
        - Pandas, Matplotlib, Seaborn 📊
    """)
