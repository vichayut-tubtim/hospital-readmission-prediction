import streamlit as st
import pandas as pd
import joblib


# =====================
# Page Config
# =====================

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


# =====================
# Load Model
# =====================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/model_pipeline.pkl"
    )


model = load_model()



# =====================
# Title
# =====================

st.title(
    "🏥 Hospital Readmission Prediction"
)

st.write(
    "Predict the risk of diabetic patients being readmitted within 30 days."
)


st.sidebar.header(
    "🧑 Patient Information"
)



# =====================
# Sidebar Inputs
# =====================


with st.sidebar.expander(
    "👤 Demographics",
    expanded=True
):

    age = st.selectbox(
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


    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )



with st.sidebar.expander(
    "🏥 Hospital Stay"
):

    time_in_hospital = st.slider(
        "Time in hospital (days)",
        1,
        14,
        3
    )


    num_lab_procedures = st.slider(
        "Number of lab procedures",
        1,
        132,
        40
    )


    num_procedures = st.slider(
        "Number of procedures",
        0,
        6,
        1
    )



with st.sidebar.expander(
    "💊 Medical History"
):

    num_medications = st.slider(
        "Number of medications",
        1,
        80,
        15
    )


    number_inpatient = st.slider(
        "Previous inpatient visits",
        0,
        20,
        0
    )


    number_outpatient = st.slider(
        "Previous outpatient visits",
        0,
        40,
        0
    )


    number_emergency = st.slider(
        "Emergency visits",
        0,
        10,
        0
    )


    number_diagnoses = st.slider(
        "Number of diagnoses",
        1,
        16,
        5
    )



# =====================
# Prepare Input
# =====================


feature_columns = (
    model
    .named_steps["preprocessor"]
    .feature_names_in_
)



input_df = pd.DataFrame(
    [[0] * len(feature_columns)],
    columns=feature_columns
)



input_df["age"] = age
input_df["gender"] = gender

input_df["time_in_hospital"] = (
    time_in_hospital
)

input_df["num_lab_procedures"] = (
    num_lab_procedures
)

input_df["num_medications"] = (
    num_medications
)

input_df["num_procedures"] = (
    num_procedures
)

input_df["number_inpatient"] = (
    number_inpatient
)

input_df["number_outpatient"] = (
    number_outpatient
)

input_df["number_emergency"] = (
    number_emergency
)

input_df["number_diagnoses"] = (
    number_diagnoses
)



# =====================
# Fix Missing Values
# =====================


preprocessor = (
    model
    .named_steps["preprocessor"]
)


num_cols = (
    preprocessor
    .transformers_[0][2]
)


cat_cols = (
    preprocessor
    .transformers_[1][2]
)



for col in num_cols:

    if col in input_df.columns:

        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        )

        input_df[col] = (
            input_df[col]
            .fillna(0)
        )



for col in cat_cols:

    if col in input_df.columns:

        input_df[col] = (
            input_df[col]
            .fillna("Unknown")
        )



# =====================
# Prediction
# =====================


st.divider()


if st.button(
    "🔍 Predict Risk",
    use_container_width=True
):


    probability = (
        model
        .predict_proba(input_df)[0][1]
    )


    st.subheader(
        "Prediction Result"
    )


    col1, col2 = st.columns(2)



    with col1:

        if probability >= 0.45:

            st.error(
                f"⚠️ High Risk\n\n"
                f"{probability:.1%}"
            )

        else:

            st.success(
                f"✅ Low Risk\n\n"
                f"{probability:.1%}"
            )



    with col2:

        st.metric(
            "Readmission Probability",
            f"{probability:.1%}"
        )



    st.progress(
        float(probability)
    )



    # =====================
    # Feature Importance
    # =====================


    # st.subheader(
    #     "📊 Important Risk Factors"
    # )


    # rf = (
    #     model
    #     .named_steps["model"]
    # )


    # importance = (
    #     rf.feature_importances_
    # )


    # feature_names = (
    #     model
    #     .named_steps["preprocessor"]
    #     .get_feature_names_out()
    # )



    # importance_df = pd.DataFrame(
    #     {
    #         "Feature": feature_names,
    #         "Importance": importance
    #     }
    # )


    # importance_df = (
    #     importance_df
    #     .sort_values(
    #         "Importance",
    #         ascending=False
    #     )
    #     .head(10)
    # )


    # st.bar_chart(
    #     importance_df.set_index(
    #         "Feature"
    #     )
    # )



    st.info(
        """
        ℹ️ This prediction is generated by a Random Forest model.
        Feature importance shows which variables influenced the model most during training.
        """
    )



st.caption(
    "Educational project only. Not for clinical decision-making."
)