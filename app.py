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
    "🏥 Hospital Readmission Prediction"
)


st.write(
    """
    Upload patient profiles and predict
    diabetic patient readmission risk within 30 days.
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
        "🔍 Predict Readmission Risk"
    ):


        with st.spinner(
            "Analyzing patients..."
        ):


            probs = (
                model
                .predict_proba(df)[:,1]
            )



            result = raw_df.copy()


            result["Readmission Probability"] = probs.astype(float)


            result["Readmission Probability"] = pd.to_numeric(
                result["Readmission Probability"],
                errors="coerce"
            )


            result["Risk"] = result[
                "Readmission Probability"
            ].apply(
                lambda x:
                "⚠️ High Risk"
                if x >= threshold
                else
                "✅ Low Risk"
            )


        st.success(
            "Prediction completed!"
        )



        st.subheader(
            "📊 Prediction Result"
        )



        st.dataframe(
            result[
                [
                    "Readmission Probability",
                    "Risk"
                ]
            ],
            use_container_width=True
        )



        st.metric(
            "High Risk Patients",
            int(
                (
                    result["Risk"]
                    ==
                    "⚠️ High Risk"
                )
                .sum()
            )
        )
        
        st.metric(
            "Average Risk",
            f"{probs.mean()*100:.2f}%"
        )

        st.metric(
            "Highest Risk",
            f"{probs.max()*100:.2f}%"
        )



        st.divider()



        st.subheader(
            "📈 Risk Distribution"
        )



        fig, ax = plt.subplots()


        ax.hist(
            probs,
            bins=20
        )


        ax.set_xlabel(
            "Probability"
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