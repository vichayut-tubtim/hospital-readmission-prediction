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
# Load Feature Importance
# =====================

@st.cache_data
def load_feature_importance():

    return pd.read_csv(
        "models/feature_importance.csv"
    )


feature_importance = load_feature_importance()



# =====================
# Title
# =====================

st.title(
    "🏥 Hospital Readmission Prediction"
)


st.write(
    "Predict the probability of diabetic patient readmission within 30 days."
)



# =====================
# Tabs
# =====================

tab1, tab2 = st.tabs(
    [
        "🔍 Prediction",
        "📊 Model Explainability"
    ]
)



# =====================================================
# Prediction Tab
# =====================================================

with tab1:


    st.sidebar.header(
        "🧑 Patient Information"
    )


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


    input_df.loc[0, "age"] = age
    input_df.loc[0, "gender"] = gender

    input_df.loc[0, "time_in_hospital"] = time_in_hospital
    input_df.loc[0, "num_lab_procedures"] = num_lab_procedures
    input_df.loc[0, "num_medications"] = num_medications
    input_df.loc[0, "num_procedures"] = num_procedures

    input_df.loc[0, "number_inpatient"] = number_inpatient
    input_df.loc[0, "number_outpatient"] = number_outpatient
    input_df.loc[0, "number_emergency"] = number_emergency

    input_df.loc[0, "number_diagnoses"] = number_diagnoses



    # =====================
    # Fix Missing
    # =====================

    preprocessor = (
        model
        .named_steps["preprocessor"]
    )


    num_cols = preprocessor.transformers_[0][2]

    cat_cols = preprocessor.transformers_[1][2]


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



    st.divider()



    if st.button(
        "🔍 Predict Risk",
        width="stretch"
    ):


        try:

            with st.spinner(
                "Analyzing patient information..."
            ):

                probability = (
                    model
                    .predict_proba(input_df)[0][1]
                )



            st.subheader(
                "Prediction Result"
            )



            if probability >= 0.45:

                st.error(
                    f"""
                    ⚠️ High Risk

                    Probability of Readmission:
                    {probability:.1%}
                    """
                )


            else:

                st.success(
                    f"""
                    ✅ Low Risk

                    Probability of Readmission:
                    {probability:.1%}
                    """
                )



            st.progress(
                float(probability)
            )


            st.info(
                """
                ℹ️ Prediction generated by Random Forest model.

                Check Model Explainability tab for important factors.
                """
            )


        except Exception as e:

            st.error(
                "Prediction failed. Please check model compatibility."
            )

            st.exception(e)




# =====================================================
# Explainability Tab
# =====================================================

with tab2:


    st.subheader(
        "📊 Feature Importance (Random Forest)"
    )


    top_features = (
        feature_importance
        .head(10)
        .copy()
    )


    # Convert importance to percentage
    top_features["Importance (%)"] = (
        top_features["Importance"] * 100
    ).round(2)


    # Reverse order for better visualization
    chart_data = (
        top_features
        .sort_values(
            "Importance (%)",
            ascending=True
        )
    )


    st.bar_chart(
        chart_data.set_index(
            "Feature"
        )["Importance (%)"]
    )


    st.write(
        "Top 10 features that contributed most to the Random Forest prediction:"
    )


    display_df = (
        top_features[
            [
                "Feature",
                "Importance (%)"
            ]
        ]
    )


    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


    st.caption(
        "Feature importance shows which variables influenced the model decision most during training. It does not represent causation."
    )