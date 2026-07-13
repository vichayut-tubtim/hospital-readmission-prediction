# 🏥 Hospital Readmission Prediction

Machine Learning project for predicting the probability of diabetic patient readmission within 30 days after hospital discharge.

The project builds an end-to-end ML pipeline including data preprocessing, model training, evaluation, explainability, and deployment with Streamlit.

---

# 🎯 Problem Statement

Hospital readmissions increase healthcare costs and may indicate potential gaps in patient care.

This project aims to identify patients with higher estimated risk of readmission within 30 days, which could support healthcare providers in prioritizing follow-up strategies.

> Note: This project is for educational purposes only and is not intended for clinical decision-making.

---

# 📊 Dataset

**UCI Diabetes 130-US Hospitals Dataset**

Dataset characteristics:

- 101,766 patient encounters
- 50 original features
- Data collected from 130 US hospitals
- Binary classification problem

Target:

| Label | Meaning |
|---|---|
| 1 | Readmitted within 30 days (`<30`) |
| 0 | Not readmitted within 30 days |

The dataset is highly imbalanced, with significantly fewer positive readmission cases.

---

# 🔄 Machine Learning Pipeline

The project uses a Scikit-Learn Pipeline:

```
Raw Dataset
      |
      v
Data Cleaning
      |
      v
Missing Value Handling
      |
      v
Feature Encoding
      |
      v
Random Forest Classifier
      |
      v
Prediction Probability
      |
      v
Streamlit Web Application
```

---

# 🔧 Data Preprocessing

Applied preprocessing steps:

### Removed high missing-value columns

- `weight`
- `medical_specialty`
- `payer_code`

### Missing value handling

- Replaced unknown race values (`?`) with `Unknown`
- Removed records with missing diagnosis codes

### Removed identifier columns

- `encounter_id`
- `patient_nbr`

### Feature transformation

Categorical features:

- Most Frequent Imputation
- One-Hot Encoding
- `handle_unknown="ignore"`

Numerical features:

- Median Imputation

---

# 🤖 Model

## Random Forest Classifier

Configuration:

```
n_estimators = 300
max_depth = 15
min_samples_split = 20
class_weight = balanced
random_state = 42
```

The model outputs:

- Readmission probability
- Risk classification based on probability threshold

---

# 📈 Model Performance

Evaluation was performed on a held-out test set (20%).

## ROC-AUC Score

```
ROC-AUC: 0.652
```

ROC-AUC was selected because the dataset is imbalanced and accuracy alone does not represent model performance well.

---

## Classification Report

The classification report evaluates the model performance for both classes.

| Class | Description | Precision | Recall | F1-score | Support |
|------|-------------|-----------|--------|----------|---------|
| **0** | Not readmitted within 30 days | 0.92 | 0.68 | 0.78 | 17,799 |
| **1** | Readmitted within 30 days | 0.17 | 0.52 | 0.26 | 2,250 |

### Overall Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.66 |
| Macro Average F1-score | 0.52 |
| Weighted Average F1-score | 0.72 |
| ROC-AUC | 0.652 |

### Interpretation

- The model achieves strong performance on the majority class (**Class 0**).
- Due to class imbalance, predicting readmission cases (**Class 1**) is more challenging.
- Using `class_weight="balanced"` improves recall for readmitted patients, allowing the model to identify more potential high-risk cases.
- ROC-AUC is used as the main evaluation metric because accuracy alone can be misleading for imbalanced classification problems.

---

## Confusion Matrix

The confusion matrix shows how well the model distinguishes between patients who were readmitted within 30 days and those who were not.

|                  | Predicted: No Readmission | Predicted: Readmission |
|------------------|---------------------------|------------------------|
| **Actual: No Readmission** | 12,158 (TN) | 5,641 (FP) |
| **Actual: Readmission** | 1,081 (FN) | 1,169 (TP) |



Because this dataset is highly imbalanced, the model focuses on improving recall for the minority class (readmitted patients) rather than maximizing accuracy.

---

# 🔍 Model Explainability

Feature importance is generated during training and saved as:

```
models/feature_importance.csv
```

The Streamlit application displays important features contributing to the Random Forest model.

Future improvements:

- SHAP explanations
- Feature contribution analysis
- Local prediction explanations

---

# 🖥️ Streamlit Application

The deployed application provides:

✅ Patient information input  
✅ Readmission probability prediction  
✅ Risk assessment visualization  
✅ Model explainability dashboard  

Demo:

https://hospital-readmission-prediction-kqdlipdjdbz4sj8yqnfb6p.streamlit.app/

---

# 🛠️ Tech Stack

## Programming

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-Learn
- Random Forest
- Joblib

## Deployment

- Streamlit

---

# 📂 Project Structure

```
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
├── models/
│   ├── model_pipeline.pkl
│   └── feature_importance.csv
│
└── screenshots/
```

---

# 🚀 How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train.py
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

Possible improvements:

- Hyperparameter optimization
- Cross-validation
- Threshold optimization
- Compare with XGBoost / LightGBM
- SHAP-based explainability
- Model calibration
- Better handling of class imbalance
- Feature engineering

---

# 👨‍💻 Author

Machine Learning Portfolio Project

Built with Python, Scikit-Learn, and Streamlit.