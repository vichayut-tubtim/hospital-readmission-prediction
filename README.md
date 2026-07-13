# 🏥 Hospital Readmission Prediction

ทำนายความเสี่ยงที่คนไข้โรคเบาหวานจะกลับมา admit ซ้ำใน 30 วัน

## 🎯 Problem Statement
โรงพยาบาลต้องการระบุคนไข้ที่มีความเสี่ยงสูงล่วงหน้า เพื่อวางแผนการดูแลและลดค่าใช้จ่าย

## 📊 Dataset
- UCI Diabetes 130-US Hospitals Dataset
- 100,000+ records, 50 features

## 🔍 Key Findings
- `num_lab_procedures` คือ feature สำคัญที่สุด
- คนไข้ที่ตรวจ lab เยอะและได้ยาเยอะมีความเสี่ยง readmit สูงกว่า
- ROC-AUC Score: 0.6496

## 🛠️ Tech Stack
- Python, pandas, scikit-learn
- RandomForest Classifier
- Streamlit

## 🚀 Demo
[คลิกที่นี่เพื่อทดลองใช้](#)