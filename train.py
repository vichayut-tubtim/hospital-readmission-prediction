import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score

df = pd.read_csv("diabetic_data.csv")
print("Original shape:", df.shape)

# Cleaning
df = df.drop(columns=["weight", "medical_specialty", "payer_code"], errors="ignore")
df["race"] = df["race"].replace("?", "Unknown")

for col in ["diag_1", "diag_2", "diag_3"]:
    df = df[df[col] != "?"]

# Target
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

df = df.drop(columns=["encounter_id", "patient_nbr", "readmitted"])

# Encoding
encoders = {}
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split
X = df.drop(columns=["readmitted_binary"])
y = df["readmitted_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluation
y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

best_threshold = 0
best_f1 = 0

for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold {threshold}: F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold:", best_threshold)

final_pred = (y_prob >= best_threshold).astype(int)

print(classification_report(y_test, final_pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "features.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(best_threshold, "threshold.pkl")

print("Model saved!")