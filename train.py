import os
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


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
    RocCurveDisplay
)



# =====================
# Load Dataset
# =====================

df = pd.read_csv(
    "data/diabetic_data.csv"
)


print(
    "Original shape:",
    df.shape
)



# =====================
# Cleaning
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
    columns=[
        "readmitted_binary"
    ]
)


y = df["readmitted_binary"]



# =====================
# Split
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
    n_jobs=1,
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
# Train
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


roc_auc = roc_auc_score(
    y_test,
    prob
)


print(
    "ROC-AUC:",
    roc_auc
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



# Create folder

os.makedirs(
    "screenshots",
    exist_ok=True
)



# =====================
# Confusion Matrix Image
# =====================

cm = confusion_matrix(
    y_test,
    pred
)


plt.figure(
    figsize=(6,5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Readmission",
        "Readmission"
    ],
    yticklabels=[
        "No Readmission",
        "Readmission"
    ]
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300
)


plt.close()



# =====================
# Classification Report Image
# =====================

report = classification_report(
    y_test,
    pred,
    output_dict=True
)


report_df = pd.DataFrame(
    report
).transpose()


plt.figure(
    figsize=(8,4)
)


sns.heatmap(
    report_df.iloc[:2,:3],
    annot=True,
    fmt=".2f",
    cmap="Blues"
)


plt.title(
    "Classification Report"
)


plt.tight_layout()


plt.savefig(
    "screenshots/classification_report.png",
    dpi=300
)


plt.close()



# =====================
# ROC Curve Image
# =====================

plt.figure(
    figsize=(6,5)
)


RocCurveDisplay.from_predictions(
    y_test,
    prob
)


plt.title(
    f"ROC Curve (AUC={roc_auc:.3f})"
)


plt.tight_layout()


plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300
)


plt.close()



# =====================
# Feature Importance
# =====================

rf = pipeline.named_steps["model"]


feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)



importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }
)



importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)



os.makedirs(
    "models",
    exist_ok=True
)



importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)



print(
    "Feature importance saved!"
)



# =====================
# Feature Importance Image
# =====================

top_features = (
    importance_df
    .head(10)
)


plt.figure(
    figsize=(8,5)
)


sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)


plt.title(
    "Top 10 Feature Importance"
)


plt.tight_layout()


plt.savefig(
    "screenshots/feature_importance.png",
    dpi=300
)


plt.close()



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