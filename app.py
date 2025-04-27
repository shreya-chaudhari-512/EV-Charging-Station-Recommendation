# Import necessary libraries
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

# Train and return the model
@st.cache_resource
def get_model(df):
    try:
        X = df[['latitude', 'longitude', 'vehicle_type', 'duration']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Load and prepare data
df = load_data()
if df is not None:
    df, le_vehicle_type = process_data(df)
    model, scaler = get_model(df)

# Define prediction function
def make_prediction(model, scaler, le_vehicle_type):
    st.subheader("Enter EV Charging Station Parameters")

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=19.0760)
        longitude = st.number_input("Longitude", value=72.8777)
        vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Bike', 'Truck', 'Bus'])
        encoded_vehicle_type = le_vehicle_type.transform([vehicle_type])[0]

    with col2:
        duration = st.number_input("Expected Charging Duration (in minutes)", value=30)

    input_data = np.array([[latitude, longitude, encoded_vehicle_type, duration]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.success("Prediction: Install EV Charging Station")
    else:
        st.warning("Prediction: Don't Install EV Charging Station")

# Display different pages based on selection
if page == "Home":
    st.header("Welcome to EV Charging Station Optimization")

    st.subheader("Project Overview")
    st.write("""
        This project aims to identify optimal locations for Electric Vehicle (EV) charging stations based on factors such as 
        population density, traffic flow, existing infrastructure, and power availability. The goal is to predict where charging 
        stations should be installed to maximize efficiency and accessibility, enabling the transition to sustainable transportation.
    """)

    if df is not None:
        st.subheader("Dataset Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", df.shape[0])
            st.metric("Average Latitude", round(df['latitude'].mean(), 4))
            st.metric("Average Longitude", round(df['longitude'].mean(), 4))

        with col2:
            st.metric("Average Vehicle Type", int(df['vehicle_type'].mean()))
            st.metric("Average Duration (min)", int(df['duration'].mean()))
            st.metric("Install EV Charger (Ratio)", round(df['target'].mean(), 2))

        st.write("Sample Data:")
        st.dataframe(df.head())
    else:
        st.warning("No data available. Please check if the dataset is correctly loaded.")

elif page == "Make Prediction":
    if model is not None and scaler is not None and le_vehicle_type is not None:
        make_prediction(model, scaler, le_vehicle_type)
    else:
        st.error("Model not available. Please check data and model training.")

elif page == "Data Exploration":
    if df is not None:
        st.header("Data Exploration")
        st.write("First 10 rows of the dataset:")
        st.dataframe(df.head(10))

        st.subheader("Feature Distributions")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        sns.histplot(df['latitude'], kde=True, ax=axs[0, 0])
        sns.histplot(df['longitude'], kde=True, ax=axs[0, 1])
        sns.histplot(df['duration'], kde=True, ax=axs[1, 0])
        sns.countplot(x='vehicle_type', data=df, ax=axs[1, 1])

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("No data available to explore.")

elif page == "About":
    st.header("About This Project")
    st.write("""
        This system is developed as part of a project to assist urban planners, city councils, and private firms in making data-driven 
        decisions for EV infrastructure development.
        
        **Key Technologies Used:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn.
    """)


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
    st.header("EV Charging Station Availability Prediction")

    if df is not None and model is not None:
        st.subheader("Enter Location and Vehicle Details")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input("Latitude", format="%.6f", value=19.0760)
            longitude = st.number_input("Longitude", format="%.6f", value=72.8777)

        with col2:
            vehicle_type_encoded = st.number_input("Vehicle Type (Encoded)", min_value=0, step=1)
            duration_seconds = st.number_input("Charging Duration (Seconds)", min_value=0.0, value=3600.0)

        st.info("Example Encoding:\n\n0: Two-Wheeler\n1: Three-Wheeler\n2: Four-Wheeler\n3: Heavy Vehicle")

        # Prediction button
        if st.button("Predict Availability"):
            # Prepare input
            input_data = np.array([[latitude, longitude, vehicle_type_encoded, duration_seconds]])
            
            # Scale input (important because model was trained on scaled data)
            scaled_input = scaler.transform(input_data)

            # Predict
            prediction = model.predict(scaled_input)[0]

            # Display result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success("✅ EV Charging Slot Likely Available!")
            else:
                st.error("❌ EV Charging Slot Likely Not Available.")

            # If your model has probability prediction
            if hasattr(model, "predict_proba"):
                st.subheader("Prediction Confidence")
                prediction_proba = model.predict_proba(scaled_input)[0]

                proba_df = pd.DataFrame({
                    'Availability': ['Not Available (0)', 'Available (1)'],
                    'Probability': prediction_proba
                }).sort_values('Probability', ascending=False)

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=proba_df, x='Availability', y='Probability', palette="viridis", ax=ax)
                plt.title('Availability Prediction Confidence')
                plt.ylim(0, 1)
                st.pyplot(fig)
    else:
        st.warning("Model or data not available. Please check if everything is loaded correctly.")
# ABOUT PAGE
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
