# 🤖 Universal Employee Burnout & Attrition Risk Predictor

A universal HR intelligence tool that predicts employee burnout and attrition risk using Machine Learning. Works for any industry — corporate, healthcare, education, manufacturing, retail, and more.

## 🔗 Live Demo
[https://universal-risk-predictor.streamlit.app/](https://universal-risk-predictor.streamlit.app/)

## 🛠️ Tech Stack
- Python 3.13
- XGBoost
- SHAP (Explainable AI)
- Streamlit
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Plotly
- OpenPyXL

## ✨ Features
- Works for ANY industry — not just corporate HR
- Single employee risk prediction with universal input form
- Bulk prediction via CSV upload (any employee dataset)
- Risk gauge meter (Low / Medium / High)
- SHAP-based explainability (top factors driving risk)
- Downloadable color-coded Excel report

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.13
- pip

### Steps

1. Clone the repository
git clone https://github.com/TanishkaPagar/universal-risk-predictor.git
cd universal-risk-predictor

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

4. Open in browser
http://localhost:8501

## 📊 Model Performance
- Algorithm: XGBoost with SMOTE for class imbalance
- Accuracy: ~81%
- ROC-AUC Score: 0.71
- Training Data: IBM HR Analytics Dataset (1,470 employees)

## 📂 Project Structure
employee_burnout_predictor/
├── app.py                  # Main Streamlit app
├── model.py                # ML model training
├── predict.py              # Prediction & SHAP logic
├── utils.py                # Excel report generator
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── model/
    └── universal_model.pkl # Trained model

## 👩‍💻 Developed By
Tanishka Pagar — LABTECH Internship 2026