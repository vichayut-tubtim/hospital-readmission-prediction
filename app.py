import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥"
)


@st.cache_resource
def load_model():
    return joblib.load("models/model_pipeline.pkl")


model = load_model()

st.title("🏥 Hospital Readmission Prediction")
st.write(
    "Predict the risk of diabetic patients being readmitted within 30 days."
)


st.sidebar.header("Patient Information")


age = st.sidebar.selectbox(
    "Age",
    [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)"
    ]
)


gender = st.sidebar.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


time_in_hospital = st.sidebar.slider(
    "Time in hospital",
    1,
    14,
    3
)


num_lab_procedures = st.sidebar.slider(
    "Number of lab procedures",
    1,
    132,
    40
)


num_medications = st.sidebar.slider(
    "Number of medications",
    1,
    80,
    15
)


num_procedures = st.sidebar.slider(
    "Number of procedures",
    0,
    6,
    1
)


number_inpatient = st.sidebar.slider(
    "Previous inpatient visits",
    0,
    20,
    0
)


number_outpatient = st.sidebar.slider(
    "Previous outpatient visits",
    0,
    40,
    0
)


number_emergency = st.sidebar.slider(
    "Emergency visits",
    0,
    10,
    0
)


number_diagnoses = st.sidebar.slider(
    "Number of diagnoses",
    1,
    16,
    5
)


# Get columns used during training
feature_columns = model.named_steps[
    "preprocessor"
].feature_names_in_


# Create dataframe with all features
input_df = pd.DataFrame(
    [[0] * len(feature_columns)],
    columns=feature_columns
)


# Fill user inputs
input_df["age"] = age
input_df["gender"] = gender
input_df["time_in_hospital"] = time_in_hospital
input_df["num_lab_procedures"] = num_lab_procedures
input_df["num_medications"] = num_medications
input_df["num_procedures"] = num_procedures
input_df["number_inpatient"] = number_inpatient
input_df["number_outpatient"] = number_outpatient
input_df["number_emergency"] = number_emergency
input_df["number_diagnoses"] = number_diagnoses


# Get numeric and categorical columns from pipeline
preprocessor = model.named_steps["preprocessor"]

num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]


# Fix missing values
for col in num_cols:
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        )
        input_df[col] = input_df[col].fillna(0)


for col in cat_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna("Unknown")


st.divider()


if st.button("🔍 Predict"):

    probability = model.predict_proba(
        input_df
    )[0][1]


    st.subheader("Prediction Result")


    if probability >= 0.45:
        st.error(
            f"⚠️ High Risk: {probability:.1%}"
        )
    else:
        st.success(
            f"✅ Low Risk: {probability:.1%}"
        )


    st.progress(
        float(probability)
    )


st.caption(
    "Educational project only. Not for clinical decision-making."
)