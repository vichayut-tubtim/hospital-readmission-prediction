# 🏥 Hospital Readmission Prediction

Machine Learning project for predicting whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge.

## 🎯 Problem Statement

Hospital readmissions increase healthcare costs and may indicate gaps in patient care. Early identification of high-risk patients can help healthcare providers plan follow-up treatments and interventions more effectively.

## 📊 Dataset

**UCI Diabetes 130-US Hospitals Dataset**

* 101,766 patient encounters
* 50 original features
* Data collected from 130 US hospitals
* Binary classification task:

  * `1` = Readmitted within 30 days (`<30`)
  * `0` = Not readmitted within 30 days

## 🔧 Data Preprocessing

The following preprocessing steps were applied:

* Removed columns with excessive missing values:

  * `weight`
  * `medical_specialty`
  * `payer_code`
* Replaced missing race values (`?`) with `Unknown`
* Removed records with missing diagnosis codes
* Removed identifier columns:

  * `encounter_id`
  * `patient_nbr`
* Converted target variable into binary classification

## 🤖 Model Pipeline

The project uses a Scikit-Learn Pipeline consisting of:

1. Data Cleaning
2. Missing Value Imputation
3. One-Hot Encoding for categorical features
4. Random Forest Classification

### Preprocessing

* Numerical Features:

  * Median Imputation

* Categorical Features:

  * Most Frequent Imputation
  * One-Hot Encoding (`handle_unknown="ignore"`)

### Model

RandomForestClassifier

* n_estimators = 300
* max_depth = 15
* min_samples_split = 20
* class_weight = "balanced"
* random_state = 42

## 📈 Model Performance

Evaluation Metric:

* ROC-AUC Score: **0.655**

ROC-AUC was selected because the dataset is imbalanced, making accuracy alone an unreliable performance metric.

## 🖥️ Streamlit Application

The trained model is deployed using Streamlit and allows users to:

* Input patient information
* Predict readmission probability
* View risk assessment results in real time

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

## 📂 Project Structure

```text
hospital-readmission-prediction/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── diabetic_data.csv
│
└── models/
    └── model_pipeline.pkl
```

## 🚀 Demo

Streamlit App: https://hospital-readmission-prediction-kqdlipdjdbz4sj8yqnfb6p.streamlit.app/

## 📚 Future Improvements

* Hyperparameter Optimization
* Feature Importance Analysis
* Explainable AI (SHAP)
* XGBoost / LightGBM Comparison
* Threshold Optimization
* Cross Validation
