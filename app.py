import streamlit as st
import numpy as np
import joblib

# Load the trained models
heart_model, diabetes_model = joblib.load("model.pkl")

st.title("🩺 Multi-Disease Prediction Engine")
st.subheader("Heart Disease & Diabetes Risk Predictor")

with st.form("health_form"):
    age = st.number_input("Age", 1, 120, 30)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 22.5)
    systolic = st.number_input("Systolic BP (mm Hg)", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP (mm Hg)", 50, 140, 80)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 180)
    glucose = st.number_input("Glucose Level (mg/dL)", 60, 300, 100)
    smoking = st.radio("Do you smoke?", ("No", "Yes"))
    alcohol = st.radio("Do you consume alcohol?", ("No", "Yes"))
    activity = st.radio("Physically Active?", ("Yes", "No"))
    submitted = st.form_submit_button("Predict")

def encode_yes_no(val):
    return 1 if val == "Yes" else 0

if submitted:
    input_data = np.array([[
        age, bmi, systolic, diastolic, cholesterol, glucose,
        encode_yes_no(smoking), encode_yes_no(alcohol), encode_yes_no(activity)
    ]])

    heart_prob = heart_model.predict_proba(input_data)
    diabetes_prob = diabetes_model.predict_proba(input_data)

    def get_risk(prob):
        if prob >= 0.7:
            return "🔴 High Risk"
        elif prob >= 0.4:
            return "🟠 Moderate Risk"
        else:
            return "🟢 Low Risk"

    st.success("✅ Prediction Complete!")
    st.write(f"**Heart Disease Risk:** {get_risk(heart_prob[0][1])} ({heart_prob[0][1]*100:.2f}%)")
    st.write(f"**Diabetes Risk:** {get_risk(diabetes_prob[0][1])} ({diabetes_prob[0][1]*100:.2f}%)")
