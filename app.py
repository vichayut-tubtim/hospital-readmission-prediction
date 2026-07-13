import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def train_model():
    df = pd.read_csv('diabetic_data.csv')
    
    # Clean
    df = df.drop(columns=['weight', 'medical_specialty', 'payer_code'])
    df['race'] = df['race'].replace('?', 'Unknown')
    df = df[~df['diag_1'].isin(['?'])]
    df = df[~df['diag_2'].isin(['?'])]
    df = df[~df['diag_3'].isin(['?'])]
    
    # Target
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'])
    
    # Encode
    le = LabelEncoder()
    for col in df.select_dtypes(include='str').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(columns=['readmitted_binary'])
    y = df['readmitted_binary']
    
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    return model, list(X.columns)

st.title('🏥 Hospital Readmission Prediction')
st.write('ทำนายความเสี่ยงที่คนไข้จะกลับมา admit ซ้ำใน 30 วัน')

with st.spinner('กำลัง train model...'):
    model, features = train_model()

st.sidebar.header('ข้อมูลคนไข้')
time_in_hospital = st.sidebar.slider('จำนวนวันที่อยู่โรงพยาบาล', 1, 14, 3)
num_medications = st.sidebar.slider('จำนวนยาที่ได้รับ', 1, 80, 15)
num_lab_procedures = st.sidebar.slider('จำนวมการตรวจ Lab', 1, 132, 40)
num_procedures = st.sidebar.slider('จำนวน Procedures', 0, 6, 1)
number_inpatient = st.sidebar.slider('จำนวนครั้งที่เคย admit', 0, 20, 0)

input_data = pd.DataFrame([[0] * len(features)], columns=features)
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