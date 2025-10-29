import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model & scaler
model = pickle.load(open(r"C:\Users\KIIT\tasks\diabetes-predictor\trained_model.sav", "rb"))
scaler = pickle.load(open(r"C:\Users\KIIT\tasks\diabetes-predictor\scaler.sav", "rb"))

def diabetes_prediction(input_data):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    input_df = pd.DataFrame([input_data], columns=columns)
    std_data = scaler.transform(input_df)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        return "✅ The patient is Non-Diabetic"
    else:
        return "⚠️ The patient is Diabetic"


def main():
    st.title("🔬 Diabetes Prediction Web App (SVM)")
    st.write("Enter patient details to predict diabetes")

    # UI inputs
    Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose Level", 0, 300, 100)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.number_input("DP Function", 0.0, 3.0, 0.2)
    Age = st.number_input("Age", 1, 120, 25)

    # Prediction button
    if st.button("Predict"):
        result = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ])
        st.success(result)

if __name__ == '__main__':
    main()
