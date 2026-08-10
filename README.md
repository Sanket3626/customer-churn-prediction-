# Live Demo

[Try the Customer Churn Prediction App](https://sanket-customer-churn-3626.streamlit.app)

# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on their demographic details, services, contract information, and billing details.

## Project Overview

Customer churn is an important problem for businesses because losing existing customers can directly affect revenue and growth.

The main goal of this project is to use customer information and machine learning to identify customers who may be at a higher risk of leaving a service.

I built this project as an end-to-end machine learning workflow, starting from understanding the dataset and performing exploratory analysis to preprocessing, model training, evaluation, model selection, and finally creating a Streamlit web application for prediction.

The complete project was developed using **VS Code**.

---

## Why This Project?

Customer retention is generally more cost-effective than continuously acquiring new customers.

If a business can identify customers who are more likely to leave, it can take suitable actions such as:

- Offering better plans
- Providing additional support
- Understanding customer issues
- Providing suitable discounts or offers
- Improving the overall customer experience

This project demonstrates how machine learning can be used for this type of prediction problem.

---

## Problem Statement

The objective is to predict the churn status of a customer using information such as:

- Customer demographics
- Location
- Tenure
- Phone services
- Internet services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer lifetime value

The target is a binary classification problem:

- `1` → Customer is likely to churn
- `0` → Customer is unlikely to churn

---

## Project Objectives

The main objectives of this project are:

1. Understand the customer churn dataset.
2. Perform exploratory data analysis.
3. Clean and prepare the data.
4. Handle categorical and numerical features.
5. Build multiple classification models.
6. Compare model performance.
7. Analyze important features.
8. Select a suitable final model.
9. Save the trained model and preprocessing pipeline.
10. Build a Streamlit application for customer churn prediction.

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer information related to:

- Customer details
- Location information
- Demographics
- Phone services
- Internet services
- Contract information
- Payment information
- Monthly and total charges
- Customer lifetime value
- Churn information

The dataset contains **7,043 customer records**.

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Data Preprocessing
        ↓
Feature Encoding
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Feature Importance Analysis
        ↓
Final Model Selection
        ↓
Model Saving
        ↓
Final Validation
        ↓
Streamlit Application
```

---

## 1. Data Understanding

The dataset was first examined to understand:

- Number of rows and columns
- Feature names
- Data types
- Missing values
- Numerical and categorical features
- Target variable

This helped in deciding the preprocessing steps required before model training.

---

## 2. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand patterns in customer data.

The analysis included:

- Churn distribution
- Numerical feature distributions
- Categorical feature analysis
- Customer tenure
- Contract types
- Internet services
- Payment methods
- Monthly charges
- Total charges
- Relationships between different features and churn

The purpose of EDA was to understand which customer characteristics may be related to churn.

---

## 3. Data Cleaning

The data was checked for issues such as:

- Missing values
- Incorrect data types
- Duplicate records
- Inconsistent values

The data was then prepared for the machine learning pipeline.

---

## 4. Data Preprocessing

The preprocessing stage handled the numerical and categorical features separately.

The preprocessing pipeline was saved using `joblib` so that the same transformations can be applied when making predictions through the Streamlit application.

The saved preprocessing file is:

```text
models/preprocessor.pkl
```

---

## 5. Feature Encoding

Categorical variables were converted into a format that machine learning models can use.

Features such as:

- Gender
- Contract
- Internet Service
- Payment Method
- Paperless Billing
- Online Security
- Tech Support

were processed as part of the preprocessing pipeline.

Numerical features were also processed where required.

---

## 6. Machine Learning Models

Three classification models were evaluated during the project:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were compared using classification metrics, with particular attention given to the **F1 Score** because churn prediction requires a reasonable balance between precision and recall.

---

## 7. Model Evaluation

The following metrics were used for evaluating the models:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many of the customers predicted as churn customers actually churned.

### Recall

Measures how many of the actual churn customers were correctly identified.

### F1 Score

The F1 Score combines precision and recall into a single metric.

For this project, F1 Score was given importance during model selection because missing potential churn customers can be costly for a business.

---

## 8. Final Model Selection

Among the evaluated models, the **Decision Tree Classifier** achieved the highest F1 Score and was selected as the final model.

### Final Decision Tree Performance

| Metric | Score |
|---|---:|
| Accuracy | 70.28% |
| Precision | 59.14% |
| Recall | 58.82% |
| F1 Score | 58.98% |
| ROC-AUC | 82.84% |

The Decision Tree achieved an F1 Score of **0.5898**, which was the best among the evaluated models.

The trained model was saved using `joblib`:

```text
models/decision_tree_model.pkl
```

---

## 9. Feature Importance

Feature importance analysis was performed to understand which variables had more influence on the Decision Tree predictions.

Some of the more influential features included:

- Month-to-month contracts
- Fiber optic internet service
- Customer tenure
- Dependents
- Billing-related features

These features can provide useful information about the characteristics of customers who may have a higher risk of churn.

---

## 10. ROC-AUC Analysis

The final Decision Tree achieved a ROC-AUC score of:

```text
0.8284
```

ROC-AUC was used to understand how well the model can distinguish between customers who are likely to churn and customers who are unlikely to churn.

---

## 11. Model Saving and Validation

After selecting the final model, the trained Decision Tree and preprocessing pipeline were saved as `.pkl` files.

The saved model was then loaded again and tested on the unseen test dataset.

This was done to make sure that the saved model could still generate predictions correctly after being reloaded.

---

## Streamlit Web Application

A Streamlit application was created to make the model easier to use.

The application allows a user to enter customer information such as:

### Customer Information

- City
- Zip Code
- Latitude
- Longitude
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Phone Service
- Multiple Lines

### Internet Services

- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Contract and Billing

- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- CLTV

After entering the details, the application processes the input using the saved preprocessing pipeline and sends it to the trained Decision Tree model.

The application then displays the estimated churn risk.

For example:

```text
Churn Risk Score: 100.00%

High Churn Risk

This customer is likely to churn based on the provided details.
```

or:

```text
Churn Risk Score: 20.00%

Low Churn Risk

This customer is unlikely to churn.
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sanket3626/customer-churn-prediction-.git
```

### 2. Open the project folder

```bash
cd customer-churn-prediction-
```

### 3. Create a virtual environment

For Windows:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

```bash
.venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run main/main.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── raw/
│       └── Telco_customer_churn.xlsx
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
│
├── src/
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **VS Code**
- **Git & GitHub**

---

## Key Learnings

While working on this project, I worked through the complete machine learning pipeline instead of only training a model.

Some of the main things covered were:

- Understanding a real-world dataset
- Performing EDA
- Cleaning data
- Preprocessing numerical and categorical features
- Building classification models
- Comparing model performance
- Understanding precision, recall and F1 Score
- Using ROC-AUC for model evaluation
- Analyzing feature importance
- Saving and loading trained models
- Connecting a trained ML model with a Streamlit application
- Managing the project using Git and GitHub

---

## Limitations

The current model is based on the available dataset and therefore its predictions depend on the quality and characteristics of that data.

Some possible limitations include:

- Model performance can vary on new datasets.
- The Decision Tree may not capture every complex relationship in customer behavior.
- Customer churn can also depend on factors that are not present in the dataset.
- The current application provides a prediction but does not include a complete customer retention strategy.

---

## Future Improvements

Some improvements that can be added in the future are:

- Hyperparameter tuning for better model performance
- Trying additional machine learning algorithms
- Handling class imbalance more carefully
- Adding probability-based risk categories
- Adding more visualizations to the Streamlit application
- Adding customer retention recommendations
- Deploying the Streamlit application online
- Adding model monitoring after deployment

---

## Conclusion

This project demonstrates an end-to-end approach to customer churn prediction using machine learning.

Three classification models were evaluated, and the Decision Tree was selected as the final model based on its F1 Score.

The trained model was saved and integrated into a Streamlit application where users can enter customer information and receive a churn risk prediction.

The project also helped in understanding the complete workflow from raw customer data to a usable machine learning application.

---

## Author

**Sanket Dhokchaule**

Computer Science Engineering - Data Science

Built as a machine learning project to understand and apply the complete customer churn prediction workflow.