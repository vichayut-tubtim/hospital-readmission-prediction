import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from ml.src.feature_engineering import feature_engineering


st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =====================
# Load Model
# =====================

@st.cache_resource
def load_model():

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "models/catboost_readmission.pkl"
    )

    package = joblib.load(
        MODEL_PATH
    )


    model = package["model"]
    threshold = package["threshold"]
    features = package["features"]


    cat_features = [
        i
        for i in model.get_cat_feature_indices()
    ]


    cat_columns = [
        features[i]
        for i in cat_features
    ]


    return (
        model,
        threshold,
        features,
        cat_columns
    )



model, threshold, feature_columns, cat_features = load_model()



# =====================
# Feature Importance
# =====================

@st.cache_data
def load_importance():

    path = os.path.join(
        BASE_DIR,
        "models/feature_importance.csv"
    )

    return pd.read_csv(path)



importance = load_importance()



# =====================
# Header
# =====================

st.title(
    "🏥 Hospital Readmission Risk Stratification"
)


st.write(
    """
    Upload patient profiles and estimate
    30-day readmission risk score for diabetic patients.

    The score is used for patient risk ranking and
    follow-up prioritization, not as a clinical diagnosis.
    """
)


# =====================
# Upload
# =====================

st.sidebar.header(
    "📂 Patient Data Upload"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Patient Profiles",
    type=["csv"]
)



# =====================
# Sample CSV
# =====================

sample_path = os.path.join(
    BASE_DIR,
    "data/sample_patients.csv"
)


if os.path.exists(sample_path):

    sample = pd.read_csv(
        sample_path
    )


    st.sidebar.download_button(
        label="⬇️ Download Sample CSV",
        data=sample.to_csv(index=False),
        file_name="sample_patients.csv",
        mime="text/csv"
    )



# =====================
# Prediction
# =====================

if uploaded_file:


    raw_df = pd.read_csv(
        uploaded_file
    )


    df = feature_engineering(
        raw_df
    )


    # =====================
    # Match Training Features
    # =====================

    for col in feature_columns:

        if col not in df.columns:

            if col in cat_features:
                df[col] = "Unknown"

            else:
                df[col] = 0



    df = df[
        feature_columns
    ]


    for col in cat_features:

        if col in df.columns:

            if str(df[col].dtype) == "category":
                df[col] = (
                    df[col]
                    .astype(object)
                    .fillna("Unknown")
                )

            else:
                df[col] = (
                    df[col]
                    .fillna("Unknown")
                    .astype(str)
                )



    st.subheader(
        "👥 Uploaded Patients"
    )


    st.dataframe(
        raw_df.head(),
        use_container_width=True
    )



    if st.button(
        "🔍 Predict Readmission Risk Score"
    ):

        with st.spinner(
            "Analyzing patients..."
        ):

            risk_scores = (
                model
                .predict_proba(df)[:, 1]
            )


            result = raw_df.copy()


            # Risk Score (not probability)
            result["Readmission Risk Score"] = risk_scores.astype(float)


            result["Readmission Risk Score"] = pd.to_numeric(
                result["Readmission Risk Score"],
                errors="coerce"
            ).fillna(0)


            result["Risk Level"] = result[
                "Readmission Risk Score"
            ].apply(
                lambda x:
                "⚠️ High Risk"
                if x >= threshold
                else
                "🟢 Lower Risk"
            )


        st.success(
            "Risk assessment completed!"
        )


        st.subheader(
            "📊 Risk Stratification Result"
        )


        st.dataframe(
            result[
                [
                    "Readmission Risk Score",
                    "Risk Level"
                ]
            ],
            use_container_width=True
        )


        st.caption(
            f"High Risk threshold: {threshold:.2f}. "
            "Higher scores indicate higher priority for follow-up review. "
            "This score is a ranking score, not a calibrated probability."
        )


        col1, col2, col3 = st.columns(3)


        with col1:
            st.metric(
                "⚠️ High Risk Patients",
                int(
                    (
                        result["Risk Level"]
                        ==
                        "⚠️ High Risk"
                    ).sum()
                )
            )


        with col2:
            st.metric(
                "Average Risk Score",
                f"{risk_scores.mean():.3f}"
            )


        with col3:
            st.metric(
                "Highest Risk Score",
                f"{risk_scores.max():.3f}"
            )


        st.divider()


        st.subheader(
            "📈 Risk Score Distribution"
        )


        fig, ax = plt.subplots()


        ax.hist(
            risk_scores,
            bins=20
        )


        ax.set_xlabel(
            "Risk Score"
        )


        ax.set_ylabel(
            "Patients"
        )


        st.pyplot(
            fig
        )



else:

    st.info(
        "👈 Upload patient CSV file to start prediction."
    )



# =====================
# Explainability
# =====================

st.divider()


st.subheader(
    "📊 Model Explainability"
)



top = importance.head(10)



st.bar_chart(
    top.set_index(
        "Feature"
    )
)