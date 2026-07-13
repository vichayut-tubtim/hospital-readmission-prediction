# 🏥 Hospital Readmission Prediction

Machine Learning Web Application สำหรับทำนายความเสี่ยงที่ผู้ป่วยโรคเบาหวานจะกลับมาโรงพยาบาลภายใน 30 วันหลัง discharge

🔗 **Live Demo:**  
https://hospital-readmission-prediction-kqdlipdjdbz4sj8yqnfb6p.streamlit.app/

---

## 🎯 Problem Statement

Early hospital readmission เป็นปัญหาสำคัญที่ส่งผลต่อคุณภาพการรักษาและค่าใช้จ่ายของระบบสาธารณสุข

โปรเจกต์นี้มุ่งสร้าง Machine Learning model เพื่อช่วยระบุผู้ป่วยที่มีความเสี่ยงสูงต่อการ readmit ภายใน 30 วัน เพื่อสนับสนุนการวางแผนดูแลผู้ป่วยเชิงป้องกัน

> Note: This project is for educational purposes only and is not intended for clinical decision-making.

---

## 📊 Dataset

**Dataset:** Diabetes 130-US Hospitals Dataset (UCI Machine Learning Repository)

รายละเอียด:
- 101,766 patient records
- 50 original features
- Target:
  - `<30` → Readmitted within 30 days (High Risk)
  - `NO`, `>30` → Not readmitted within 30 days

---

## 🔄 Data Processing

ขั้นตอน preprocessing:

- Removed features with high missing values:
  - `weight`
  - `medical_specialty`
  - `payer_code`

- Handled missing values:
  - Replaced missing race values with `Unknown`
  - Removed incomplete diagnosis records

- Removed identifiers:
  - `encounter_id`
  - `patient_nbr`

- Encoded categorical features using Label Encoding

- Addressed class imbalance using:
  - `class_weight="balanced"`

---

## 🤖 Machine Learning Model

Model: **Random Forest Classifier**

Configuration:
- n_estimators = 300
- max_depth = 15
- class_weight = balanced

เพิ่มเติม:
- Probability threshold tuning
- Optimized decision threshold: `0.45`

---

## 📈 Model Performance

Evaluation:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.6638 |
| Recall (High-risk patients) | 0.51 |
| F1-score (High-risk patients) | 0.27 |

เนื่องจาก dataset มี class imbalance สูง จึงให้ความสำคัญกับ Recall ของกลุ่มผู้ป่วยเสี่ยงมากกว่า Accuracy

---

## 🔍 Feature Importance

Top features ที่ model ใช้ในการทำนาย:

| Feature | Importance |
|---|---:|
| number_inpatient | 0.133 |
| diag_1 | 0.084 |
| num_lab_procedures | 0.076 |
| diag_2 | 0.076 |
| diag_3 | 0.075 |
| discharge_disposition_id | 0.073 |
| num_medications | 0.068 |

Insight:
- ประวัติการ admit ก่อนหน้าเป็นปัจจัยสำคัญที่สุด
- จำนวนการตรวจ Lab และจำนวนยาสะท้อนความซับซ้อนของอาการผู้ป่วย
- Diagnosis codes มีผลต่อความเสี่ยง readmission

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Processing
- pandas
- numpy

### Machine Learning
- scikit-learn
- Random Forest

### Deployment
- Streamlit Cloud

---

## 📁 Project Structure
```
hospital-readmission-prediction/
├── app.py                  # Streamlit frontend
├── train.py                # Training pipeline
├── models/
│   └── model_pipeline.pkl  # preprocessing + model
├── data/
│   └── diabetic_data.csv
├── notebooks/
│   └── EDA.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```
Train model:
```bash
python train.py
```
Run application:
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements
- Replace Label Encoding with One-Hot Encoding pipeline
- Try advanced boosting models:
    - XGBoost
    - LightGBM
- Add explainability using SHAP
- Improve model calibration
- Add patient risk dashboard

---

## 👨‍💻 Author

Developed as a Machine Learning deployment project using Python and Streamlit.