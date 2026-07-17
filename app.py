import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


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

    package = joblib.load(
        "models/catboost_readmission.pkl"
    )

    return (
        package["model"],
        package["threshold"]
    )


model, threshold = load_model()



# =====================
# Feature Importance
# =====================

@st.cache_data
def load_importance():

    return pd.read_csv(
        "models/feature_importance.csv"
    )


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
    the probability of diabetic patient
    readmission within 30 days.
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
# Sample Download
# =====================


sample = pd.read_csv(
    "data/sample_patients.csv"
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


    df = pd.read_csv(
        uploaded_file
    )


    st.subheader(
        "👥 Uploaded Patients"
    )


    st.dataframe(
        df.head(),
        use_container_width=True
    )



    if st.button(
        "🔍 Predict Readmission Risk"
    ):


        with st.spinner(
            "Analyzing patients..."
        ):


            probs = model.predict_proba(df)[:,1]


            result = df.copy()


            result["Readmission Probability"] = (
                probs
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
            (result["Risk"]=="⚠️ High Risk").sum()
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

        st.pyplot(fig)



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
    top.set_index("Feature")
)