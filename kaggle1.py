#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 02:58:01 2024

@author: ashleymantz
"""
pip install -r requirements.txt
## MUST INSTALL -> to your environment to work conda install scikit-learn=1.3.0 ##

import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Load the model
with open('rf_clf.pkl', 'rb') as file:
    model = pickle.load(file)

# Numeric inputs
st.header("Enter Respondent's Details")

# Input fields for numeric values
age = st.selectbox("What is your current age in years?", ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", 
                                                          "45-49", "50-54", "55-59", "60-69", "70+"])
coding_experience = st.selectbox("For how many years have you been writing code and/or programming?", 
                                options=["<1", "1-3", "3-5", "5-10", "10-20", "20+"])
country = st.selectbox("In which country do you currently reside?", ["United States", "India", "Canada", "United Kingdom", "Germany", "Other"])
company_size = st.selectbox("What is the size of the company where you are employed?", 
                            ["0-49", "50-249", "250-999", "1000-9,999", "10,000+"])
ml_usage = st.selectbox("Does your current employer incorporate machine learning methods into their business?", [
    "We do not use ML methods",
    "Exploring ML methods",
    "Use ML methods for insights",
    "Models in production (<2 years)",
    "Models in production (2+ years)"
])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "In which country do you currently reside?": [country],
    "What is the size of the company where you are employed?": [company_size],
    "Does your current employer incorporate machine learning methods into their business?": [ml_usage],
    "What is your age (# years)?": [age],
    "For how many years have you been writing code and/or programming?": [coding_experience],
})

# One-hot encode the categorical variables to match the model's training data
input_data_encoded = pd.get_dummies(input_data, columns=[
    "For how many years have you been writing code and/or programming?", 
    "In which country do you currently reside?", 
    "What is the size of the company where you are employed?", 
    "Does your current employer incorporate machine learning methods into their business?",
    "What is your age (# years)?"
])

# Ensure all expected columns are present (fill missing columns with 0s)
model_columns = model.feature_names_in_  # Get the feature names used during training
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0  # Add missing column with value 0

# Reorder columns to match the model's training data
input_data_encoded = input_data_encoded[model_columns]

# Define the compensation prediction ranges
compensation_ranges = [f"${int(start)}-{int(end)}" for start, end in zip(np.arange(0, 1000000, 5000), np.arange(5000, 1000001, 5000))]
compensation_ranges.append(">$1,000,000")  # Add a range for values above $1,000,000


if st.button("Evaluate Respondent"):
    # Predict using the loaded model
    raw_prediction = model.predict(input_data_encoded)
    st.write(f"Raw Prediction: {raw_prediction}")
    
    prediction = raw_prediction[0]  # Extract the prediction
    
    # Map the prediction to the corresponding compensation range
    if prediction > 1000000:
        predicted_range = ">$1,000,000"
    else:
        # Iterate through continuous ranges to find the correct range
        for i, (start, end) in enumerate(zip(np.arange(0, 1000000, 5000), np.arange(5000, 1000001, 5000))):
            if start <= prediction < end:
                predicted_range = compensation_ranges[i]
                break

    # Display result
    st.write(f"The predicted yearly compensation range is: **{predicted_range}** 💰")
