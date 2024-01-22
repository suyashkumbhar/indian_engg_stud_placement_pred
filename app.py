# Import necessary libraries
import streamlit as st
import joblib
import numpy as np

# Load the trained machine learning model
model = joblib.load('placement_model.pkl')

# Streamlit app
def main():
    # Set the title of the web app
    st.title("Student Placement Predictor")

    # Collect user input features
    features = collect_user_input()

    # Make prediction using the loaded model
    prediction = predict_placement(features)

    # Display the prediction
    display_prediction(prediction)

# Function to collect user input features
def collect_user_input():
    # Create input fields for relevant features
    cgpa = st.slider("Enter CGPA (out of 10)", 0.0, 10.0, 8.0)
    iq = st.slider("Enter IQ", 0, 200, 100)

    # Organize user input into a dictionary
    user_input = {
        'cgpa': cgpa,
        'iq': iq
    }

    return user_input

# Function to make prediction using the loaded model
def predict_placement(features):
    # Prepare input data for prediction
    input_data = np.array([features['cgpa'], features['iq']]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]

# Function to display the prediction
def display_prediction(prediction):
    if prediction == 1:
        st.success("Congratulations! The student is predicted to get placement.")
    else:
        st.error("Sorry, the student is predicted not to get placement.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
