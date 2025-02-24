import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data (adjust path if needed)
try:
    data = pd.read_csv("AQI_Data.csv")
except FileNotFoundError:
    st.error("Error: AQI_Data.csv not found.  Make sure it's in the same directory or provide the correct path.")
    st.stop() # Prevents further execution

# Data Cleaning and Preprocessing (Handling Missing Values)
data = data.dropna()

# Features and target based on the notebook
X = data[['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']]  # Independent variables
y = data['PM 2.5']  # Dependent variable (what we're predicting)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("AQI Prediction App")

# Display Data Summary
st.header("Data Summary")
st.write("First few rows of the data:")
st.dataframe(data.head())

st.write("Data shape:", data.shape)

# Input Features - Directly using column names from the dataset
st.header("Input Features")
col1, col2 = st.columns(2)
T = col1.number_input("Temperature (T)", value=data['T'].mean())
TM = col2.number_input("Max Temperature (TM)", value=data['TM'].mean())
Tm = col1.number_input("Min Temperature (Tm)", value=data['Tm'].mean())
SLP = col2.number_input("Sea Level Pressure (SLP)", value=data['SLP'].mean())
H = col1.number_input("Humidity (H)", value=data['H'].mean())
VV = col2.number_input("Visibility (VV)", value=data['VV'].mean())
V = col1.number_input("Wind Speed (V)", value=data['V'].mean())
VM = col2.number_input("Max Wind Speed (VM)", value=data['VM'].mean())

# Prediction
if st.button("Predict AQI"):
    input_data = pd.DataFrame([[T, TM, Tm, SLP, H, VV, V, VM]],
                              columns=['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])  # Correct column order
    prediction = model.predict(input_data)[0]
    st.write("Predicted AQI:", prediction)

# Model Evaluation Metrics
st.header("Model Evaluation Metrics")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)
