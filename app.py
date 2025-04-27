import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st

# Assuming df1 is your cleaned dataset
# Step 1: Drop unnecessary text-heavy columns
cols_to_drop = ['uid', 'name', 'vendor_name', 'address', 'city', 'country',
                'open', 'close', 'logo_url', 'payment_modes', 'contact_numbers']
df1_model = df1.drop(columns=cols_to_drop, errors='ignore')

# Step 2: Encode categorical columns
cols_to_encode = ['power_type', 'type', 'vehicle_type', 'zone', 'station_type', 'staff']
le = LabelEncoder()
for col in cols_to_encode:
    df1_model[col] = le.fit_transform(df1_model[col])

# Step 3: Clean 'cost_per_unit' column
def clean_cost(x):
    if isinstance(x, str):
        x = x.replace('₹', '').replace('per unit', '').strip()
    try:
        return float(x)
    except:
        return 0.0

df1_model['cost_per_unit'] = df1_model['cost_per_unit'].apply(clean_cost)

# Step 4: Clean 'duration' column
def clean_duration(x):
    if isinstance(x, str) and 'days' in x:
        return pd.to_timedelta(x).total_seconds()
    try:
        return float(x)
    except:
        return 0.0

df1_model['duration'] = df1_model['duration'].apply(clean_duration)

# Step 5: Confirm data is clean
st.write("✅ Data ready for KNN Classification. Columns:", df1_model.columns.tolist())

# Step 6: Define X (features) and y (target)
selected_features = ['latitude', 'longitude', 'vehicle_type', 'duration']
X = df1_model[selected_features]

# 🎯 Set your classification target
y = df1_model['available']   # 0 or 1

# Step 7: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Train KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Step 9: Predict on Test Set
y_pred_knn = knn_clf.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred_knn)
report = classification_report(y_test, y_pred_knn)

st.write("\n🎯 KNN Classification Model Performance:")
st.write(f"Accuracy: {accuracy:.4f}")
st.write("\nClassification Report:\n", report)

# Step 11: Take user input and make prediction using Streamlit widgets
st.write("\n🔮 Now let's predict availability based on your input!")

try:
    user_latitude = st.number_input("Enter latitude (numeric):", value=0.0)
    user_longitude = st.number_input("Enter longitude (numeric):", value=0.0)
    user_vehicle_type = st.number_input("Enter vehicle type (encoded integer):", value=0, step=1)
    user_duration = st.number_input("Enter duration (in seconds):", value=0.0)
    
    if user_latitude and user_longitude and user_vehicle_type is not None:
        # Prepare input for prediction
        user_input = [[user_latitude, user_longitude, user_vehicle_type, user_duration]]
        
        # Predict
        user_prediction = knn_clf.predict(user_input)
        
        st.write(f"\n✅ Predicted Availability: {int(user_prediction[0])} (1 = Available, 0 = Not Available)")
except Exception as e:
    st.error(f"❌ Error in input or prediction: {e}")
