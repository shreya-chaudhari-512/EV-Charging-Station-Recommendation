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

# Application title and description
st.title("EV Charging Station Optimization System")
st.markdown("""
    This application predicts whether an EV charging station should be installed based on location coordinates, vehicle type, and expected charging duration.
    Use the sidebar to navigate through different sections of the app.
""")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Data Exploration", "Make Prediction", "About"])

# Create a sample dataframe for demonstration
# Load your dataset
# Process data function
@st.cache_data
def load_data():
    try:
        # Load the data from the repository's relative path
        df = pd.read_csv('final_cleaned.csv')  # Relative path to the dataset
        st.sidebar.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Process data function
@st.cache_data
def process_data(df):
    try:
        # Encode 'vehicle_type' using LabelEncoder
        le_vehicle_type = LabelEncoder()
        df['vehicle_type'] = le_vehicle_type.fit_transform(df['vehicle_type'])
        
        # Encode the 'target' column (target) as 1 or 0
        df['target'] = df['target'].map({'Install': 1, 'Don\'t Install': 0})
        
        # Drop any rows with missing values
        df.dropna(inplace=True)
        
        return df, le_vehicle_type
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return df, None

# Train or load the model
@st.cache_resource
def get_model(df):
    try:
        # Select features and target
        X = df[['latitude', 'longitude', 'vehicle_type', 'duration']]
        y = df['target']  # Use the 'target' column here
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train KNN model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None


# Load and process the data
df = load_data()
if df is not None:
    df, le_vehicle_type = process_data(df)

    # Train or load the model
    model, scaler = get_model(df)

# Prediction function
def make_prediction(model, scaler, le_vehicle_type):
    if model is not None and scaler is not None:
        st.subheader("Enter EV Charging Station Parameters")
        
        # Input fields for user
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Latitude", value=19.0760)
            longitude = st.number_input("Longitude", value=72.8777)
            vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Bike', 'Truck', 'Bus'])
            encoded_vehicle_type = le_vehicle_type.transform([vehicle_type])[0]
        
        with col2:
            duration = st.number_input("Expected Charging Duration (in minutes)", value=30)
        
        # Prepare input data for prediction
        input_data = np.array([[latitude, longitude, encoded_vehicle_type, duration]])
        
        # Scale the input data
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display prediction result
        if prediction == 1:
            st.success("Prediction: Install EV Charging Station")
        else:
            st.warning("Prediction: Don't Install EV Charging Station")
    
# Run the prediction page
if st.button("Make Prediction"):
    make_prediction(model, scaler, le_vehicle_type)

# HOME PAGE
if page == "Home":
    st.header("Welcome to EV Charging Station Optimization")

    # Display project overview
    st.subheader("Project Overview")
    st.write("""
        This project aims to identify optimal locations for Electric Vehicle (EV) charging stations based on factors such as 
        population density, traffic flow, existing infrastructure, and power availability. The goal is to predict where charging 
        stations should be installed to maximize efficiency and accessibility, enabling the transition to sustainable transportation.
    """)

    # Display summary statistics if data is available
    if df is not None:
        st.subheader("Dataset Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", df.shape[0])
            st.metric("Average Latitude", float(df['latitude'].mean()))
            st.metric("Average Longitude", float(df['longitude'].mean()))
        with col2:
            st.metric("Average Vehicle Type", int(df['vehicle_type'].mean()))
            st.metric("Average Duration", int(df['duration'].mean()))
            st.metric("Install EV Charger (1/0)", int(df['target'].mean()))  # Average of the target

        # Display a sample of the data
        st.write("Sample Data:")
        st.write(df.head().to_html(), unsafe_allow_html=True)
    else:
        st.warning("No data available. Please check if the dataset is correctly loaded.")

# DATA EXPLORATION PAGE
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if df is not None:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Vehicle Distribution", "Duration by Vehicle Type", "Geospatial Analysis"])
        
        with tab1:
            st.subheader("Vehicle Distribution by Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x='vehicle_type', ax=ax)
            plt.title('Vehicle Distribution by Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            st.write("""
                This count plot shows the distribution of different vehicle types in the dataset. 
                It helps to understand which vehicle types are more prevalent, which may influence charging station location decisions.
            """)
        
        with tab2:
            st.subheader("Charging Duration by Vehicle Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='vehicle_type', y='duration', ax=ax)
            plt.title('Charging Duration by Vehicle Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Charging Duration (Seconds)')
            st.pyplot(fig)
            
            st.write("""
                This boxplot shows how charging durations vary by vehicle type. Longer charging times could be an indicator of the need 
                for more chargers or faster charging infrastructure for certain vehicle types.
            """)
        
        with tab3:
            st.subheader("Geospatial Analysis: Locations of Potential EV Chargers")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='longitude', y='latitude', hue='vehicle_type', palette='Set1', ax=ax)
            plt.title('Geospatial Distribution of EV Charging Locations')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            st.pyplot(fig)
            
            st.write("""
                This scatter plot shows the distribution of data points based on the geographical location of potential EV charging stations.
                Understanding the spatial patterns helps identify areas with high demand for chargers.
            """)
    else:
        st.warning("No data available for exploration. Please check if the dataset is correctly loaded.")

# PREDICTION PAGE
elif page == "Make Prediction":
    st.header("EV Charging Station Installation Prediction")
    
    if df is not None and model is not None:
        # Create input form for prediction
        st.subheader("Enter Location and Vehicle Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Latitude input
            latitude = st.number_input("Latitude", format="%.6f")
            
            # Longitude input
            longitude = st.number_input("Longitude", format="%.6f")
        
        with col2:
            # Vehicle Type input (encoded manually)
            vehicle_type_encoded = st.number_input("Vehicle Type (Encoded)", min_value=0, step=1)
            
            # Expected Charging Duration (in seconds)
            duration_seconds = st.number_input("Expected Charging Duration (Seconds)", min_value=0.0, step=3600.0)
        
        # Show entered data
        st.metric("Charging Duration (Hours)", round(duration_seconds / 3600, 2))
        
        # Show Vehicle Type Encoding Reference
        st.subheader("Vehicle Type Encoding Reference")
        st.write("""
        - 0: Two-Wheeler
        - 1: Three-Wheeler
        - 2: Four-Wheeler
        - 3: Heavy Vehicle
        """)
        
        # Prediction button
        if st.button("Predict Installation Decision"):
            # Validate inputs
            if latitude == 0.0 or longitude == 0.0:
                st.warning("⚠️ Please enter valid latitude and longitude values.")
            else:
                # Prepare input data
                input_data = np.array([[latitude, longitude, vehicle_type_encoded, duration_seconds]])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.success("✅ Recommendation: Install EV Charging Point!")
                else:
                    st.error("❌ Recommendation: Do NOT Install EV Charging Point.")
                
                # Show prediction probabilities if available
                if hasattr(model, "predict_proba"):
                    st.subheader("Prediction Confidence")
                    
                    prediction_proba = model.predict_proba(input_data)[0]
                    
                    proba_df = pd.DataFrame({
                        'Decision': ['Do NOT Install', 'Install'],
                        'Probability': prediction_proba
                    })
                    
                    # Sort by probability
                    proba_df = proba_df.sort_values('Probability', ascending=False)
                    
                    # Display top 2 outcomes
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=proba_df, x='Decision', y='Probability', ax=ax, palette="viridis")
                    plt.title('Installation Decision Confidence')
                    plt.ylabel('Probability')
                    plt.ylim(0, 1)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    else:
        st.warning("Model or data not available for prediction. Please check if everything is correctly loaded.")
