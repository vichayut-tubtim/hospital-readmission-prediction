# ==========================================
# Diabetes Readmission Prediction
# CatBoost + Domain Feature Engineering
# (leakage-fixed pipeline)
# ==========================================

import os
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from src.feature_engineering import feature_engineering
from catboost import CatBoostClassifier


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/diabetic_data.csv")

print("Original shape:", df.shape)


# ==========================================
# Cleaning
# ==========================================

df = df.replace("?", np.nan)

# Keep latest patient encounter
# (prevents the same patient appearing in both train and test)
if "patient_nbr" in df.columns and "encounter_id" in df.columns:
    df = (
        df
        .sort_values(["patient_nbr", "encounter_id"])
        .drop_duplicates(subset="patient_nbr", keep="last")
    )

# Drop encounters that ended in death / hospice.
# These patients cannot be readmitted, so labeling them "not readmitted"
# is not a valid negative example — it's a different outcome entirely.
expired_or_hospice = [11, 13, 14, 19, 20, 21]
if "discharge_disposition_id" in df.columns:
    before = df.shape[0]
    df = df[~df["discharge_disposition_id"].isin(expired_or_hospice)]
    print(f"Dropped {before - df.shape[0]} expired/hospice encounters")

drop_cols = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "examide",
    "citoglipton"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print("After cleaning:", df.shape)


# ==========================================
# Target
# ==========================================

df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

# ==========================================
# Feature Engineering
# ==========================================

df = feature_engineering(df)

print("Positive rate:", df["readmit_30"].mean())


# ==========================================
# Train / Valid / Test Split
# Done BEFORE any frequency-based grouping (diag codes, medical_specialty)
# so that "top categories" are learned from train only.
# ==========================================

X = df.drop(columns=["readmit_30"])
y = df["readmit_30"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42,
    stratify=y_train_full
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test:", X_test.shape)


# ==========================================
# Diagnosis grouping — fit on TRAIN only
# ==========================================

diag_top_codes = {}

for col in ["diag_1", "diag_2", "diag_3"]:
    X_train[col] = X_train[col].astype(str)
    top_codes = X_train[col].value_counts().head(50).index
    diag_top_codes[col] = set(top_codes)

    X_train[col] = np.where(X_train[col].isin(top_codes), X_train[col], "Other")

    for split_df in (X_valid, X_test):
        split_df[col] = split_df[col].astype(str)
        split_df[col] = np.where(
            split_df[col].isin(diag_top_codes[col]), split_df[col], "Other"
        )


# ==========================================
# Medical specialty — fit on TRAIN only
# ==========================================

if "medical_specialty" in X_train.columns:
    for split_df in (X_train, X_valid, X_test):
        split_df["medical_specialty"] = split_df["medical_specialty"].fillna("Unknown")

    top_specialty = set(
        X_train["medical_specialty"].value_counts().head(10).index
    )

    for split_df in (X_train, X_valid, X_test):
        split_df["medical_specialty"] = np.where(
            split_df["medical_specialty"].isin(top_specialty),
            split_df["medical_specialty"],
            "Other"
        )


# ==========================================
# Categorical Features
# ==========================================

cat_features = (
    X_train.select_dtypes(include=["object", "category"]).columns.tolist()
)

for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("Unknown")
    X_valid[col] = X_valid[col].astype(str).fillna("Unknown")
    X_test[col] = X_test[col].astype(str).fillna("Unknown")

cat_indices = [X_train.columns.get_loc(col) for col in cat_features]

print("Categorical features:", len(cat_features))


# ==========================================
# CatBoost Model
# Single definition. Early stopping uses VALID, never TEST.
# ==========================================

model = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    boosting_type="Ordered",
    l2_leaf_reg=10,
    random_strength=1,
    auto_class_weights="Balanced",
    od_type="Iter",
    od_wait=200,
    cat_features=cat_indices,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)


# ==========================================
# Threshold Optimization — done on VALID, not test
# ==========================================

valid_prob = model.predict_proba(X_valid)[:, 1]

# PR-AUC is a more honest summary metric than ROC-AUC when the positive
# class is rare (~4.6% here) — ROC-AUC looks deceptively high on
# imbalanced problems because it's dominated by the large negative class.
pr_auc_valid = average_precision_score(y_valid, valid_prob)
print("PR-AUC (validation):", round(pr_auc_valid, 4))

# Search the full [0, 1] range with fine steps so the optimum can't be
# cut off by an arbitrary search boundary (this is what happened last run:
# F1 was still climbing at threshold=0.65, the edge of the old range).
threshold_results = []
for threshold in np.arange(0.02, 0.99, 0.02):
    pred = (valid_prob >= threshold).astype(int)
    f1 = f1_score(y_valid, pred, zero_division=0)
    f2 = fbeta_score(y_valid, pred, beta=2, zero_division=0)
    threshold_results.append({
        "threshold": round(threshold, 3),
        "f1_score": f1,
        "f2_score": f2
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)

# --- Why F2 instead of F1, and why threshold is picked on VALID not TEST ---
# In 30-day readmission screening, a missed at-risk patient (false negative)
# usually costs more than a false alarm (false positive): a missed
# readmission can mean a preventable ER visit or complication, while a
# false alarm just costs a care-coordinator follow-up call. F2 weights
# recall twice as much as precision, which better reflects that asymmetry
# than F1's equal weighting. The exact beta should ultimately be set from
# a real cost ratio (cost of a missed readmission vs. cost of a follow-up
# call) if those numbers are available — beta=2 here is a reasonable
# default in the absence of that data, not the "correct" answer.
best_row = threshold_df.sort_values("f2_score", ascending=False).iloc[0]
best_threshold = best_row["threshold"]

print("Best threshold (chosen on validation set, optimizing F2):", best_threshold)
print("Validation F1 at this threshold:", round(best_row["f1_score"], 4))
print("Validation F2 at this threshold:", round(best_row["f2_score"], 4))

if best_threshold in (threshold_df["threshold"].min(), threshold_df["threshold"].max()):
    print(
        "WARNING: best threshold sits at the edge of the search range — "
        "widen the range further and re-check before trusting this value."
    )


# ==========================================
# Capacity-based operating points
# A single "optimal" threshold from a metric formula (F1/F2) assumes the
# hospital can act on however many patients that threshold happens to
# flag. In practice, care-coordination teams can only follow up with a
# limited number of discharges per day. This table answers the more
# useful question: "if we can only intervene on the top K% highest-risk
# patients, what recall/precision do we get?" — so a stakeholder can pick
# an operating point based on their actual team capacity, not just a
# metric that sounds good in isolation.
# Cutoffs are derived from VALID only, exactly like the F2 threshold above.
# ==========================================

capacity_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

capacity_cutoffs = {}
capacity_table = []

for capacity in capacity_levels:
    cutoff = np.quantile(valid_prob, 1 - capacity)
    capacity_cutoffs[capacity] = cutoff

    valid_flag = (valid_prob >= cutoff).astype(int)
    n_flagged = valid_flag.sum()
    n_true_pos = ((valid_flag == 1) & (y_valid.values == 1)).sum()

    recall = n_true_pos / max(y_valid.sum(), 1)
    precision = n_true_pos / max(n_flagged, 1)

    capacity_table.append({
        "capacity_%": int(capacity * 100),
        "threshold": round(cutoff, 4),
        "%_flagged_valid": round(valid_flag.mean() * 100, 1),
        "recall_valid": round(recall, 3),
        "precision_valid": round(precision, 3)
    })

capacity_df = pd.DataFrame(capacity_table)
print("\nCapacity-based operating points (thresholds derived on VALID):")
print(capacity_df)


# ==========================================
# Final Evaluation — TEST set touched exactly once, here
# ==========================================

test_prob = model.predict_proba(X_test)[:, 1]
final_pred = (test_prob >= best_threshold).astype(int)

auc = roc_auc_score(y_test, test_prob)
pr_auc_test = average_precision_score(y_test, test_prob)
print("ROC-AUC (test):", auc)
print("PR-AUC (test):", round(pr_auc_test, 4))

print(classification_report(y_test, final_pred, digits=3))
print(confusion_matrix(y_test, final_pred))


# ==========================================
# Capacity-based operating points — applied to TEST
# Same cutoffs from the VALID table above, touched on TEST exactly once,
# same discipline as the single F2-threshold evaluation.
# ==========================================

test_capacity_table = []
for capacity in capacity_levels:
    cutoff = capacity_cutoffs[capacity]

    test_flag = (test_prob >= cutoff).astype(int)
    n_flagged = test_flag.sum()
    n_true_pos = ((test_flag == 1) & (y_test.values == 1)).sum()

    recall = n_true_pos / max(y_test.sum(), 1)
    precision = n_true_pos / max(n_flagged, 1)

    test_capacity_table.append({
        "capacity_%": int(capacity * 100),
        "threshold": round(cutoff, 4),
        "%_flagged_test": round(test_flag.mean() * 100, 1),
        "recall_test": round(recall, 3),
        "precision_test": round(precision, 3)
    })

test_capacity_df = pd.DataFrame(test_capacity_table)
print("\nCapacity-based operating points (same VALID-derived thresholds, evaluated on TEST):")
print(test_capacity_df)


# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})
importance = importance.sort_values("Importance", ascending=False)

print(importance.head(30))

os.makedirs("models", exist_ok=True)
importance.to_csv("models/feature_importance.csv", index=False)


# ==========================================
# Visualizations
# All computed from the TEST-set predictions already made above (test_prob,
# final_pred) — no extra touches of the test set. Saved as PNGs so they can
# be dropped straight into a README or slide deck.
# ==========================================

os.makedirs("screenshots", exist_ok=True)

sns.set_theme(style="whitegrid")

# --- Confusion matrix ---
cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Readmission", "Readmission"],
    yticklabels=["No Readmission", "Readmission"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (threshold={best_threshold}, test set)")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=300)
plt.close()

# --- ROC curve ---
plt.figure(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title(f"ROC Curve (AUC={auc:.3f}, test set)")
plt.tight_layout()
plt.savefig("screenshots/roc_curve.png", dpi=300)
plt.close()

# --- Precision-Recall curve ---
# More informative than the ROC curve here — positive rate is only ~4.6%,
# so PR-AUC reflects performance on the rare class much more honestly.
plt.figure(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, test_prob)
baseline_rate = y_test.mean()
plt.axhline(baseline_rate, color="gray", linestyle="--",
            label=f"Random baseline ({baseline_rate:.3f})")
plt.legend()
plt.title(f"Precision-Recall Curve (PR-AUC={pr_auc_test:.3f}, test set)")
plt.tight_layout()
plt.savefig("screenshots/precision_recall_curve.png", dpi=300)
plt.close()

# --- Feature importance (top 15) ---
top_features = importance.head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=top_features, x="Importance", y="Feature", color="steelblue")
plt.title("Top 15 Feature Importance (CatBoost)")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=300)
plt.close()

# --- Capacity vs. recall ("lift") curve ---
# The single clearest chart for a non-technical audience: if the care team
# can only follow up with the top K% highest-risk discharges, this shows
# what fraction of true 30-day readmissions get caught — versus what a
# random K% selection would catch (the diagonal-equivalent baseline).
plt.figure(figsize=(7, 5))
capacities_pct = [row["capacity_%"] for row in test_capacity_table]
recalls = [row["recall_test"] for row in test_capacity_table]

plt.plot(capacities_pct, recalls, marker="o", label="Model (risk-ranked)")
plt.plot(
    [0, 100], [0, 1],
    color="gray", linestyle="--", label="Random selection baseline"
)
plt.xlabel("% of discharges flagged (capacity)")
plt.ylabel("Recall (% of true readmissions caught)")
plt.title("Capacity vs. Recall — Model vs. Random Selection (test set)")
plt.legend()
plt.xlim(0, 45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("screenshots/capacity_recall_curve.png", dpi=300)
plt.close()

print("Plots saved to screenshots/")


# ==========================================
# Save Model
# ==========================================

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": list(X.columns),
        "cat_features": cat_features
    },
    "models/catboost_readmission.pkl"
)

print("Model saved!")