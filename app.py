import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

st.title('🏥 Hospital Readmission Prediction')
st.write('ทำนายความเสี่ยงที่คนไข้จะกลับมา admit ซ้ำใน 30 วัน')

st.sidebar.header('ข้อมูลคนไข้')

time_in_hospital = st.sidebar.slider('จำนวนวันที่อยู่โรงพยาบาล', 1, 14, 3)
num_medications = st.sidebar.slider('จำนวนยาที่ได้รับ', 1, 80, 15)
num_lab_procedures = st.sidebar.slider('จำนวนการตรวจ Lab', 1, 132, 40)
num_procedures = st.sidebar.slider('จำนวน Procedures', 0, 6, 1)
number_inpatient = st.sidebar.slider('จำนวนครั้งที่เคย admit', 0, 20, 0)

# สร้าง input dataframe
input_data = pd.DataFrame(
    [[0] * len(features)], 
    columns=features
)

input_data['time_in_hospital'] = time_in_hospital
input_data['num_medications'] = num_medications
input_data['num_lab_procedures'] = num_lab_procedures
input_data['num_procedures'] = num_procedures
input_data['number_inpatient'] = number_inpatient

if st.button('ทำนาย'):
    prob = model.predict_proba(input_data)[0][1]
    
    st.subheader('ผลการทำนาย')
    
    if prob > 0.3:
        st.error(f'⚠️ ความเสี่ยงสูง: {prob*100:.1f}%')
    else:
        st.success(f'✅ ความเสี่ยงต่ำ: {prob*100:.1f}%')
    
    st.progress(float(prob))