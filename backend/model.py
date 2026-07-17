import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "../ml/models/model_pipeline.pkl"
)


model = joblib.load(MODEL_PATH)


feature_columns = (
    model
    .named_steps["preprocessor"]
    .feature_names_in_
)


def predict_readmission(data):

    input_df = pd.DataFrame(
        [data]
    )


    # เติม feature ที่ user ไม่ได้ส่งมา
    for col in feature_columns:

        if col not in input_df.columns:

            input_df[col] = "Unknown"



    input_df = input_df[
        feature_columns
    ]


    probability = (
        model
        .predict_proba(input_df)[0][1]
    )


    return {
        "risk":
            "High Risk"
            if probability >= 0.45
            else "Low Risk",

        "probability":
            round(float(probability),4)
    }