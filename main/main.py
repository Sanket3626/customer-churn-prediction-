import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("models/decision_tree_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model()


st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn based on "
    "their demographics, services, contract and billing details."
)

st.divider()


st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.text_input("City", "Los Angeles")

    zip_code = st.number_input(
        "Zip Code",
        min_value=0,
        value=90003
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.2
    )


with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col3:
    tenure_months = st.number_input(
        "Tenure Months",
        min_value=0,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


st.subheader("Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.subheader("Contract and Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=4000
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("Predict Customer Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "City": [city],
        "Zip Code": [zip_code],
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure_months],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "CLTV": [cltv]
    })

    try:
        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        st.write(f"Churn Risk Score: {probability:.2%}")

        if probability >= 0.70:
            st.error("⚠️ High Churn Risk")
            st.write(
                "This customer is likely to churn based on the provided details."
            )

        elif probability >= 0.30:
            st.warning("⚠️ Medium Churn Risk")
            st.write(
                "This customer has a moderate likelihood of churning."
            )

        else:
            st.success("✅ Low Churn Risk")
            st.write(
                "This customer is unlikely to churn."
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)