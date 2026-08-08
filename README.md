# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on their demographic details, services, contract information, and billing details.

## Project Overview

Customer churn is an important problem for businesses because losing existing customers can affect revenue and growth.

In this project, customer data is analyzed and used to build a machine learning model that predicts customer churn.

The project covers the complete machine learning workflow:

- Data understanding
- Exploratory Data Analysis (EDA)
- Data cleaning
- Data preprocessing
- Feature encoding
- Model training
- Model evaluation
- Model comparison
- Feature importance analysis
- Final model selection
- Model saving and validation
- Streamlit web application

## Dataset

The project uses a Telco Customer Churn dataset containing customer demographic, service, contract, and billing information.

The dataset contains:

- Customer information
- Location details
- Demographic information
- Phone and internet services
- Contract details
- Payment information
- Monthly and total charges
- Customer lifetime value
- Churn information

The dataset contains 7,043 customer records.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│
├── data/
│
├── main/
│   └── main.py
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── preprocessor.pkl
│   ├── X_train_processed.pkl
│   ├── X_test_processed.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_data_preprocessing.ipynb
│   └── 04_model_building.ipynb
│
├── reports/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore