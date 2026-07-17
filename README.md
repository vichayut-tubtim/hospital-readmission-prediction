# 🏥 Hospital Readmission Prediction

Machine Learning project for predicting the probability of diabetic patient readmission within 30 days after hospital discharge.

This project demonstrates an end-to-end Machine Learning workflow:

- Data preprocessing (leakage-safe)
- Feature engineering
- Model training (CatBoost)
- Threshold selection & capacity-based decision framing
- Model evaluation
- Model explainability
- Streamlit deployment

> ⚠️ Educational project only. This system is not intended for clinical decision-making.

---

# 🎯 Problem Statement

Hospital readmissions create additional healthcare costs and may indicate potential gaps in patient care.

The goal of this project is to build a model that estimates the likelihood of diabetic patients being readmitted within 30 days after discharge, producing a **risk score** that can help identify higher-risk cases that may need additional follow-up.

---

# 📊 Dataset

**UCI Diabetes 130-US Hospitals Dataset**

Dataset information:

- 101,766 original patient encounters
- 50 original features
- Data collected from 130 US hospitals

## Row-level cleaning

| Step | Rows removed | Rows remaining |
|---|---|---|
| Original | — | 101,766 |
| Keep only latest encounter per patient (dedup by `patient_nbr`) | — | (deduplicated) |
| Drop expired / hospice discharges (`discharge_disposition_id` in `[11,13,14,19,20,21]`) | 2,349 | **69,169** |

Encounters ending in death or hospice discharge are removed because these patients cannot be readmitted — labeling them "not readmitted" would not be a clinically meaningful negative example.

Target:

| Label | Meaning |
|---|---|
| 1 | Readmitted within 30 days (`<30`) |
| 0 | Not readmitted within 30 days (`>30` or `NO`) |

Final positive rate: **4.62%** — a highly imbalanced classification problem.

---

# 📂 Project Structure

```
hospital-readmission-prediction/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── .python-version
├── .gitattributes
│
├── data/
│   └── diabetic_data.csv
│
├── models/
│   ├── catboost_readmission.pkl
│   └── feature_importance.csv
│
├── notebooks/
│   └── EDA.ipynb
│
└── screenshots/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    └── capacity_recall_curve.png
```

---

# 🏗️ System Architecture

```
                 Dataset
                    |
                    v
         Row-level Cleaning
   (dedup patients, drop expired/hospice)
                    |
                    v
       Train / Valid / Test Split
                    |
                    v
  Domain Feature Engineering + Category
  Grouping (diag codes, specialty — fit
           on TRAIN only)
                    |
                    v
        CatBoost Classifier
  (native categorical handling, early
   stopping on VALID, class-balanced)
                    |
                    v
   Threshold & Capacity Selection
        (chosen on VALID only)
                    |
                    v
         Saved Pipeline
     (catboost_readmission.pkl)
                    |
                    v
             Streamlit App
                    |
          +---------+---------+
          |                   |
          v                   v
 Prediction Result     Model Explainability
 (Risk Score)          (Feature Importance)
```

---

# 🔄 Data Preprocessing

## Leakage Prevention

This is the part of the pipeline most portfolio projects on this dataset get wrong, so it's called out explicitly:

- **Train / valid / test split happens first.** Anything that "looks at the data" to make a decision — which diagnosis codes are common, which medical specialties are common — is fit on the **train split only**, then applied unchanged to valid and test.
- `diag_1`, `diag_2`, `diag_3`: the top-50 most frequent codes are learned from train; everything else (in train, valid, and test alike) is bucketed into `"Other"`.
- `medical_specialty`: the top-10 most frequent specialties are learned from train the same way.
- The **test set is only ever used once**, for the final reported metrics. Early stopping and threshold/capacity selection all happen on the validation split.

If the "top-N" categories were instead computed on the full dataset before splitting, information about the test set's distribution would leak into training — a common but subtle mistake.

## Missing Value Handling

- `?` values are converted to `NaN` across the dataset.
- `race`: missing values are not dropped — CatBoost handles missing categorical values natively via `"Unknown"` fill where relevant.
- `diag_1` / `diag_2` / `diag_3`: rows are **not dropped** for missing diagnosis codes (unlike an earlier version of this project) — missing/rare codes are folded into `"Other"` instead, preserving those rows.
- `weight`, `payer_code`: dropped (majority missing, low signal).
- `examide`, `citoglipton`: dropped (near-constant columns).

## Removed Identifier Features

Removed: `encounter_id`, `patient_nbr` (after using them to deduplicate patients).

## Categorical Features

CatBoost consumes categorical columns natively via `cat_features` indices — there is no manual one-hot encoding step. This matters here because several categorical columns (`diag_1/2/3`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`) have high cardinality, which one-hot encoding handles poorly.

---

# 🤖 Machine Learning Model

## CatBoost Classifier

Model configuration:

```
iterations       = 4000
depth            = 6
learning_rate    = 0.02
loss_function    = Logloss
eval_metric      = AUC
boosting_type    = Ordered
l2_leaf_reg      = 10
auto_class_weights = Balanced
od_type          = Iter, od_wait = 200   (early stopping)
```

**Why CatBoost:** the dataset has 33 categorical features, several with high cardinality (medical diagnosis codes), and CatBoost handles these natively without one-hot encoding, along with built-in handling for missing categorical values.

**Why a validation split, not just train/test:** early stopping (`use_best_model=True`) is driven by the **validation** set, never the test set. If the test set were used for early stopping, the reported test metrics would be optimistically biased — the model would have effectively "seen" the test set during training.

The model outputs a **risk score**, described further below.

---

# ⚠️ Risk Score, Not a Calibrated Probability

`auto_class_weights="Balanced"` is used to counteract the 4.6% positive rate. This deliberately shifts the model's output distribution to make the minority class easier to detect — which means `predict_proba()` output should be read as a **risk score for ranking patients**, not as a calibrated probability. A score of 0.7 does not mean "70% chance of readmission."

If a calibrated probability is genuinely needed (e.g. for a downstream cost calculation that requires accurate probabilities), this would need `CalibratedClassifierCV` (Platt scaling or isotonic regression) with calibration quality checked via a calibration curve and Brier score. That is listed under Future Improvements below.

---

# 📈 Model Performance

Evaluation was performed on a held-out test set (20%), touched only once after threshold/capacity selection was finalized on the validation set.

| Metric | Score |
|---|---|
| ROC-AUC | 0.716 |
| PR-AUC | 0.121 |
| Recall (at F2-optimal threshold) | 0.703 |
| Precision (at F2-optimal threshold) | 0.079 |
| F1 (at F2-optimal threshold) | 0.142 |
| F2 (at F2-optimal threshold) | 0.267 |

**On PR-AUC:** with a positive rate of 4.6%, a random model would score a PR-AUC of about 0.046. Our PR-AUC of 0.121 is roughly **2.6x better than random** — a more honest way to communicate model quality here than ROC-AUC alone, which can look deceptively strong on imbalanced data.

**On accuracy:** accuracy is not reported as a headline metric and shouldn't be treated as one — a model that predicts "no readmission" for every single patient would score about 95% accuracy while catching zero at-risk patients. PR-AUC and recall at a chosen operating point are the metrics that actually reflect performance on this problem.

---

# 🎯 Threshold Selection & Capacity Trade-off

Rather than picking a single "best" threshold from a metric formula in isolation, this project frames threshold selection as a **business trade-off**, driven by two considerations:

## 1. Recall matters more than precision here

In 30-day readmission screening, a missed at-risk patient (false negative) is generally more costly than an unnecessary follow-up call (false positive): a missed readmission can mean a preventable ER visit or complication, while a false alarm costs a care-coordinator's time. For that reason, threshold selection optimizes **F2** (recall weighted twice as heavily as precision) rather than F1, using the **validation** set only.

- F2-optimal threshold: **0.46**
- At this threshold: recall 0.703, precision 0.079

## 2. But a hospital's care-coordination team has limited capacity

A metric-optimal threshold assumes the team can act on however many patients that threshold happens to flag. In practice, follow-up capacity is limited. The table below shows recall/precision at several **capacity levels** — "if the team can only follow up with the top K% highest-risk discharges" — so an operating point can be chosen based on actual team capacity rather than a metric alone.

| Capacity | % Flagged (test) | Recall | Precision |
|---|---|---|---|
| 5% | 5.0% | 0.184 | 0.172 |
| 10% | 9.6% | 0.286 | 0.138 |
| 15% | 13.8% | 0.353 | 0.119 |
| 20% | 18.8% | 0.445 | 0.110 |
| 25% | 23.5% | 0.508 | 0.100 |
| 30% | 28.3% | 0.561 | 0.092 |
| 40% | 38.0% | 0.669 | 0.081 |

Notably, the F2-optimal threshold (0.46) lands close to the **40% capacity** row — meaning the F2-optimized threshold implicitly assumes the care team can intervene on roughly 4 in 10 discharged patients, which is a large assumption for most hospital operations. Lower-capacity rows (5–15%) offer a better lift-per-effort trade-off (see chart below) at the cost of catching fewer total at-risk patients. The right row to pick depends on real team capacity and the true cost ratio between a missed readmission and an unnecessary follow-up — not on the metric alone.

![Capacity vs Recall](screenshots/capacity_recall_curve.png)

The chart above compares the model's risk-ranked selection against a random-selection baseline at each capacity level — the model consistently beats random selection, with the largest relative lift (about 3.7x) at the smallest capacity levels, tapering off as capacity increases.

---

# 📊 Evaluation Results

## ROC Curve

![ROC Curve](screenshots/roc_curve.png)

## Precision-Recall Curve

![Precision-Recall Curve](screenshots/precision_recall_curve.png)

More informative than the ROC curve for this problem — the positive rate is only 4.6%, so PR-AUC reflects performance on the rare class more honestly than ROC-AUC does.

## Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

(At the F2-optimal threshold of 0.46, on the test set.)

---

# 🔍 Model Explainability

Feature importance is extracted from the CatBoost model.

The training process generates:

```
models/feature_importance.csv
```

and visualization:

![Feature Importance](screenshots/feature_importance.png)

Top contributing features include `admission_type_id`, `discharge_disposition_id`, `total_visits`, `discharge_high_risk`, and `number_diagnoses` — consistent with existing clinical literature on readmission risk, where prior utilization (inpatient/ER visit history) and discharge circumstances are well-established predictors.

Future improvement:

- SHAP global and per-patient explanation
- Local prediction explanation ("why is this patient high-risk?")

---

# 🖥️ Streamlit Application

The trained model is deployed using Streamlit.

Application features:

✅ Patient information input
✅ Readmission risk score prediction
✅ Risk assessment visualization
✅ Model explainability support

> **Note:** the input schema changed along with the model. The app's input handling should be checked against the current feature list (`models/catboost_readmission.pkl` → `features`) before relying on it — the new pipeline uses CatBoost's native categorical handling (column indices, not one-hot vectors) and adds domain features (`total_visits`, `discharge_high_risk`, `previous_admission_intensity`, etc.) that the earlier Random Forest version did not have.

## Demo Screenshot

![Streamlit Demo](screenshots/demo1.png)
![Streamlit Demo](screenshots/demo2.png)
![Streamlit Demo](screenshots/demo3.png)
![Streamlit Demo](screenshots/demo4.png)
![Streamlit Demo](screenshots/demo5.png)

Demo:

https://hospital-readmission-prediction-5uhxnegmwyy2i9a6xlsz2s.streamlit.app/

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- CatBoost
- Scikit-Learn (splits, metrics)
- Joblib

## Visualization

- Matplotlib
- Seaborn

## Deployment

- Streamlit

---

# 🚀 Installation & Usage

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train Model

```bash
python train.py
```

Generated files:

```
models/catboost_readmission.pkl
models/feature_importance.csv
screenshots/*.png
```

## 3. Run Streamlit

```bash
streamlit run app.py
```

---

# 📚 Model Limitations

- Dataset contains historical hospital records; performance may not transfer to other hospitals or more recent practice patterns.
- Model output is a **risk score for ranking patients**, not a calibrated probability and not a medical diagnosis.
- Feature importance indicates the model's decision influence, not causation.
- Precision at any reasonable recall level is low (this is expected given a ~4.6% positive rate) — this model is best used to prioritize follow-up attention, not as a standalone diagnostic signal.

---

# 📌 Future Improvements

- SHAP explainability (individual prediction reasoning)
- Model calibration (`CalibratedClassifierCV`, Platt scaling / isotonic regression) with Brier score and calibration curve evaluation
- Inference wrapper function (`predict_readmission()`) returning risk score + risk tier
- Cost-based threshold selection using real intervention cost vs. missed-readmission cost, if those figures become available
- Compare against LightGBM
- Monitoring for feature/label drift if deployed beyond a portfolio context

---

# 👨‍💻 Author

Developed as a Machine Learning portfolio project.

Built with: Python + CatBoost + Scikit-Learn + Streamlit
