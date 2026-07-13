import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


df = pd.read_csv(
    "data/diabetic_data.csv"
)

print("Original shape:", df.shape)


# =====================
# Data Cleaning
# =====================

df = df.drop(columns=[
    "weight",
    "medical_specialty",
    "payer_code"
])


df["race"] = df["race"].replace(
    "?",
    "Unknown"
)


for col in [
    "diag_1",
    "diag_2",
    "diag_3"
]:
    df = df[df[col] != "?"]



# =====================
# Target
# =====================

df["readmitted_binary"] = (
    df["readmitted"] == "<30"
).astype(int)


df = df.drop(columns=[
    "encounter_id",
    "patient_nbr",
    "readmitted"
])


X = df.drop(
    columns=["readmitted_binary"]
)

y = df["readmitted_binary"]



# =====================
# Split Data
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# =====================
# Columns
# =====================

cat_cols = X_train.select_dtypes(
    include="object"
).columns


num_cols = X_train.select_dtypes(
    exclude="object"
).columns



# =====================
# Preprocessing
# =====================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        num_cols
    ),
    (
        "cat",
        categorical_pipeline,
        cat_cols
    )
])



# =====================
# Model
# =====================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)



pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        rf_model
    )
])



# =====================
# Training
# =====================

pipeline.fit(
    X_train,
    y_train
)



# =====================
# Evaluation
# =====================

prob = pipeline.predict_proba(
    X_test
)[:,1]


pred = pipeline.predict(
    X_test
)


print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        prob
    )
)


print(
    confusion_matrix(
        y_test,
        pred
    )
)


print(
    classification_report(
        y_test,
        pred
    )
)



# =====================
# Save Model
# =====================

joblib.dump(
    pipeline,
    "models/model_pipeline.pkl"
)


print(
    "Model saved!"
)