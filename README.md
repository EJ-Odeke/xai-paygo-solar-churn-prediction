# Explainable AI for Customer Churn Prediction in Uganda's PAYGo Solar Sector

## Master's Capstone Project

An end-to-end machine learning project that predicts customer churn in Uganda's PAYGo solar sector using Explainable Artificial Intelligence (XAI). The project combines data preprocessing, feature engineering, predictive modeling, SHAP explainability, and an interactive Streamlit dashboard to support data-driven customer retention strategies.

---

## Project Overview

Customer churn remains one of the biggest challenges facing PAYGo solar providers. Losing customers reduces recurring revenue, increases customer acquisition costs, and limits sustainable growth.

This project develops an explainable machine learning framework capable of identifying customers at risk of churn while providing transparent explanations for every prediction. The framework enables business stakeholders to understand not only **who is likely to churn**, but also **why**.

---

## Objectives

- Predict customer churn using machine learning models.
- Engineer meaningful behavioural and financial features.
- Compare multiple classification algorithms.
- Improve performance through hyperparameter tuning.
- Explain model predictions using SHAP (SHapley Additive Explanations).
- Deploy an interactive Streamlit dashboard for business users.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Plotly
- Streamlit
- Joblib

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── paygo-solar-ai-capstone.ipynb   
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
│
├── dashboard/
│   └── app.py
│
├── images/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Feature Selection
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Model Explainability (SHAP)
10. Dashboard Deployment

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- Tuned XGBoost (Best Performing Model)

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Explainable AI

To improve model transparency, SHAP was used to explain individual and global model predictions.

The explainability module enables users to:

- Identify the most influential features
- Understand individual customer predictions
- Visualize feature contributions
- Support business decision-making

---

## Dashboard Features

The Streamlit dashboard includes:

- Customer churn prediction
- Customer risk scoring
- SHAP explanation plots
- Model performance metrics
- Interactive visualizations
- Customer behaviour insights

---

## Dataset

The project uses historical PAYGo solar customer records containing information such as:

- Customer tenure
- Payment behaviour
- Repayment rate
- Outstanding balance
- Missed payments
- Customer engagement
- Support interactions

**Note:** The original dataset is not included in this repository due to confidentiality and data privacy restrictions.

---

## Key Features Engineered

Examples include:

- Payment Momentum
- Payment Reliability
- Financial Stress Index
- Outstanding Balance Ratio
- Missed Payment Rate
- Customer Value
- Contact Intensity
- Complaint Pressure
- Support Escalation Score
- Payment Stability Score

---

## Results

The developed framework successfully predicts customer churn while providing interpretable explanations for every prediction.

Key findings include:

- Lower repayment rates increase churn risk.
- Missed payments strongly predict churn.
- Higher outstanding balances are associated with increased churn.
- Customer support interactions provide valuable behavioural insights.
- SHAP explanations improve transparency and trust in machine learning predictions.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/xai-paygo-solar-churn-prediction.git
```

Navigate into the project

```bash
cd xai-paygo-solar-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

- Real-time churn prediction
- Automated model retraining
- Customer retention recommendation engine
- Cloud deployment
- Integration with CRM systems
- Drift monitoring and model performance tracking

---

## Author

**Odeke Elijah Faith**

Master of Artificial Intelligence and Data Science (MAIDS)

Universal Technology and Management University

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgements

Special thanks to my supervisors, lecturers, and colleagues for their guidance and support throughout this research.

The customer data used in this study was provided solely for academic research and has been anonymized to protect confidentiality.
