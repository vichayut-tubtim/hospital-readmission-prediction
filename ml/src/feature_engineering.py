import pandas as pd
import numpy as np


def feature_engineering(df):

    df = df.copy()

    df = df.replace("?", np.nan)


    # ======================
    # Medication
    # ======================

    med_cols = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]

    med_cols = [
        c for c in med_cols
        if c in df.columns
    ]


    def simplify_med(x):

        if pd.isna(x):
            return "Unknown"

        if x == "No":
            return "No"

        if x == "Steady":
            return "Steady"

        if x in ["Up","Down"]:
            return "Change"

        return x


    for col in med_cols:
        df[col] = df[col].apply(
            simplify_med
        )


    df["diabetes_med_count"] = (
        df[med_cols] != "No"
    ).sum(axis=1)


    df["total_changes"] = (
        df[med_cols] == "Change"
    ).sum(axis=1)



    # ======================
    # Hospital Utilization
    # ======================

    df["total_visits"] = (
        df["number_inpatient"].fillna(0)
        +
        df["number_outpatient"].fillna(0)
        +
        df["number_emergency"].fillna(0)
    )


    df["high_utilizer"] = (
        df["number_inpatient"] >= 2
    ).astype(int)


    df["medication_complexity"] = (
        df["num_medications"].fillna(0)
        /
        (
            df["time_in_hospital"].fillna(0)
            + 1
        )
    )


    df["total_procedures"] = (
        df["num_lab_procedures"].fillna(0)
        +
        df["num_procedures"].fillna(0)
    )


    high_risk_discharge = [
        2,3,4,5,6,
        10,12,14,
        22,23,24,25
    ]


    df["discharge_high_risk"] = (
        df["discharge_disposition_id"]
        .isin(high_risk_discharge)
    ).astype(int)


    df["previous_admission_intensity"] = (
        df["number_inpatient"].fillna(0)
        /
        (
            df["time_in_hospital"].fillna(0)
            +1
        )
    )



    # ======================
    # Age
    # ======================

    age_map = {
        "[0-10)":5,
        "[10-20)":15,
        "[20-30)":25,
        "[30-40)":35,
        "[40-50)":45,
        "[50-60)":55,
        "[60-70)":65,
        "[70-80)":75,
        "[80-90)":85,
        "[90-100)":95
    }


    df["age_numeric"] = (
        df["age"].map(age_map)
    )


    df["age_group"] = pd.cut(
        df["age_numeric"],
        bins=[
            0,
            35,
            50,
            65,
            120
        ],
        labels=[
            "young",
            "middle",
            "senior",
            "elderly"
        ]
    ).astype(object)


    # ======================
    # A1C / Glucose
    # ======================


    df["a1c_measured"] = (
        df["A1Cresult"]
        .fillna("None")
        != "None"
    ).astype(int)



    df["glu_measured"] = (
        df["max_glu_serum"]
        .fillna("None")
        != "None"
    ).astype(int)



    def map_a1c(x):

        if pd.isna(x) or x=="None":
            return 0

        if x=="Norm":
            return 1

        if x==">7":
            return 2

        if x==">8":
            return 3

        return 0



    def map_glucose(x):

        if pd.isna(x) or x=="None":
            return 0

        if x=="Norm":
            return 1

        if x==">200":
            return 2

        if x==">300":
            return 3

        return 0



    df["a1c_level"] = (
        df["A1Cresult"]
        .apply(map_a1c)
    )


    df["glu_level"] = (
        df["max_glu_serum"]
        .apply(map_glucose)
    )


    df["a1c_high"] = (
        df["A1Cresult"]
        .isin([">7",">8"])
    ).astype(int)


    df["glu_high"] = (
        df["max_glu_serum"]
        .isin([">200",">300"])
    ).astype(int)



    return df