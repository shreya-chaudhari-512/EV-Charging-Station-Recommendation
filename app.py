import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set up page configuration
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

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Prediction", "Model Performance"])

# Function to clean 'cost_per_unit' column
def clean_cost(x):
    if isinstance(x, str):
        x = x.replace('₹', '').replace('per unit', '').strip()
    try:
        return float(x)
    except:
        return 0.0

# Function to clean 'duration' column
def clean_duration(x):
    if isinstance(x, str) and 'days' in x:
        return pd.to_timedelta(x).total_seconds()
    try:
        return float(x)
    except:
        return 0.0

# Load and process data for the model
def load_data():
    # Load your dataframe (replace with actual path)
    df1 = pd.read_csv("your_data.csv")  # Adjust with your actual data source
    cols_to_drop = ['uid', 'name', 'vendor_name', 'address', 'city', 'country',
                    'open', 'close', 'logo_url', 'payment_modes', 'contact_numbers']
    df1_model = df1.drop(columns=cols_to_drop, errors='ignore')

    # Encode categorical columns
    cols_to_encode = ['power_type', 'type', 'vehicle_type', 'zone', 'station_type', 'staff']
    le = LabelEncoder()
    for col in cols_to_encode:
        df1_model[col] = le.fit_transform(df1_model[col])

    # Clean 'cost_per_unit' and 'duration'
    df1_model['cost_per_unit'] = df1_model['cost_per_unit'].apply(clean_cost)
    df1_model['duration'] = df1_model['duration'].apply(clean_duration)

    # Define features and target variable
    selected_features = ['latitude', 'longitude', 'vehicle_type', 'duration']
    X = df1_model[selected_features]
    y = df1_model['available']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the KNN model
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)

    return knn_clf

# Prediction page
def predict_availability(knn_clf):
    st.title("EV Station Availability Prediction")
    st.write("Please enter the details of the EV station:")

    # User input
    user_latitude = st.number_input("Enter latitude", format="%.6f")
    user_longitude = st.number_input("Enter longitude", format="%.6f")
    user_vehicle_type = st.number_input("Enter vehicle type (encoded integer)", min_value=0, step=1)
    user_duration = st.number_input("Enter duration (in seconds)", format="%.0f")

    # Predict button
    if st.button("Predict Availability"):
        try:
            # Prepare input for prediction
            user_input = [[user_latitude, user_longitude, user_vehicle_type, user_duration]]
            
            # Prediction
            user_prediction = knn_clf.predict(user_input)
            availability = "Available" if user_prediction[0] == 1 else "Not Available"
            st.success(f"Predicted Availability: {availability} (1 = Available, 0 = Not Available)")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Model performance page
def model_performance(knn_clf):
    st.title("Model Performance")
    st.write("Here is the performance evaluation of the KNN classifier on the test data.")

    # Model evaluation
    try:
        df1 = pd.read_csv("your_data.csv")  # Replace with actual data path
        cols_to_drop = ['uid', 'name', 'vendor_name', 'address', 'city', 'country',
                        'open', 'close', 'logo_url', 'payment_modes', 'contact_numbers']
        df1_model = df1.drop(columns=cols_to_drop, errors='ignore')

        # Encode categorical columns
        cols_to_encode = ['power_type', 'type', 'vehicle_type', 'zone', 'station_type', 'staff']
        le = LabelEncoder()
        for col in cols_to_encode:
            df1_model[col] = le.fit_transform(df1_model[col])

        # Clean 'cost_per_unit' and 'duration'
        df1_model['cost_per_unit'] = df1_model['cost_per_unit'].apply(clean_cost)
        df1_model['duration'] = df1_model['duration'].apply(clean_duration)

        # Define features and target variable
        selected_features = ['latitude', 'longitude', 'vehicle_type', 'duration']
        X = df1_model[selected_features]
        y = df1_model['available']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Predict using the trained model
        y_pred_knn = knn_clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred_knn)
        report = classification_report(y_test, y_pred_knn)

        st.write(f"Accuracy: {accuracy:.4f}")
        st.text("Classification Report:")
        st.text(report)

    except Exception as e:
        st.error(f"Error in model evaluation: {e}")

# Main app code
if __name__ == "__main__":
    # Load the model
    knn_clf = load_data()

    # Navigate through pages based on selection
    if page == "Home":
        st.write("Welcome to the EV Charging Station Optimization app!")
    elif page == "Prediction":
        predict_availability(knn_clf)
    elif page == "Model Performance":
        model_performance(knn_clf)
