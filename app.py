import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes based on health data.")

# Get user input
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose Level", 0, 200, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 122, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 99, 20)
Insulin = st.number_input("Insulin Level", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 1, 120, 33)

if st.button("Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is likely not diabetic.")
