import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
threshold = joblib.load("threshold.pkl")

st.title("🏥 Hospital Readmission Prediction")
st.write("ทำนายความเสี่ยงผู้ป่วยกลับมา admit ภายใน 30 วัน")

st.sidebar.header("Patient Information")

time_in_hospital = st.sidebar.slider("จำนวนวันที่อยู่โรงพยาบาล", 1, 14, 3)
num_medications = st.sidebar.slider("จำนวนยา", 1, 80, 15)
num_lab_procedures = st.sidebar.slider("จำนวน Lab", 1, 132, 40)
num_procedures = st.sidebar.slider("จำนวน Procedures", 0, 6, 1)
number_inpatient = st.sidebar.slider("จำนวนครั้ง admit ก่อนหน้า", 0, 20, 0)

input_data = pd.DataFrame(
    np.zeros((1, len(features))),
    columns=features
)

values = {
    "time_in_hospital": time_in_hospital,
    "num_medications": num_medications,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "number_inpatient": number_inpatient
}

for key, value in values.items():
    if key in input_data.columns:
        input_data[key] = value

if st.button("🔮 Predict"):
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prob >= threshold:
        st.error(f"⚠️ High Risk: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Risk: {prob*100:.2f}%")

    st.write(f"Decision threshold: {threshold}")
    st.progress(float(prob))