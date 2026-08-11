import os

import joblib
import pandas as pd
import streamlit as st



# Page configuration


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "decision_tree_model.pkl",
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl",
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Telco_customer_churn.xlsx",
)


# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_dataset():
    data = pd.read_excel(DATA_PATH, sheet_name="Telco_Churn")
    return data


model, preprocessor = load_model()
dataset = load_dataset()


# Sample customers for quick demonstration

low_risk_row = (
    dataset[dataset["Churn Value"] == 0]
    .sort_values(["Churn Score", "Tenure Months"], ascending=[True, False])
    .iloc[0]
)

high_risk_row = (
    dataset[dataset["Churn Value"] == 1]
    .sort_values(["Churn Score", "Tenure Months"], ascending=[False, True])
    .iloc[0]
)

# Helper functions

def set_customer_values(row):
    """Load values from a dataset row into the form."""

    fields = {
        "city": row["City"],
        "zip_code": int(row["Zip Code"]),
        "latitude": float(row["Latitude"]),
        "longitude": float(row["Longitude"]),
        "gender": row["Gender"],
        "senior_citizen": row["Senior Citizen"],
        "partner": row["Partner"],
        "dependents": row["Dependents"],
        "tenure_months": int(row["Tenure Months"]),
        "phone_service": row["Phone Service"],
        "multiple_lines": row["Multiple Lines"],
        "internet_service": row["Internet Service"],
        "online_security": row["Online Security"],
        "online_backup": row["Online Backup"],
        "device_protection": row["Device Protection"],
        "tech_support": row["Tech Support"],
        "streaming_tv": row["Streaming TV"],
        "streaming_movies": row["Streaming Movies"],
        "contract": row["Contract"],
        "paperless_billing": row["Paperless Billing"],
        "payment_method": row["Payment Method"],
        "monthly_charges": float(row["Monthly Charges"]),
        "total_charges": float(row["Total Charges"]),
        "cltv": int(row["CLTV"]),
    }

    for key, value in fields.items():
        st.session_state[key] = value


def load_selected_demo():
    """Load a predefined real customer from the dataset."""

    selected = st.session_state["demo_customer"]

    if selected == "Low Churn Example":
        set_customer_values(low_risk_row)

    elif selected == "High Churn Example":
        set_customer_values(high_risk_row)


# Session state defaults

defaults = {
    "demo_customer": "Custom Input",
    "city": "Los Angeles",
    "zip_code": 90003,
    "latitude": 33.964131,
    "longitude": -118.272783,
    "gender": "Male",
    "senior_citizen": "Yes",
    "partner": "Yes",
    "dependents": "Yes",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "Yes",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "cltv": 4000,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Application header

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn based on "
    "their demographics, services, contract and billing details."
)

st.divider()


# Quick demo

st.subheader("Quick Demo")

st.selectbox(
    "Try a sample customer",
    [
        "Custom Input",
        "Low Churn Example",
        "High Churn Example",
    ],
    key="demo_customer",
    on_change=load_selected_demo,
    help=(
        "Choose a sample customer to quickly test the application. "
        "The sample values are taken from the project dataset."
    ),
)

if st.session_state["demo_customer"] != "Custom Input":
    st.caption(
        "Sample customer loaded from the Telco Customer Churn dataset. "
        "You can still change the inputs before predicting."
    )


# Customer information

st.subheader("Customer Information")

# City and ZIP data from the actual dataset
city_options = sorted(dataset["City"].dropna().unique().tolist())

if st.session_state["city"] not in city_options:
    st.session_state["city"] = city_options[0]

city = st.selectbox(
    "City",
    city_options,
    key="city",
    help="Select a city available in the project dataset.",
)

zip_options = sorted(
    dataset.loc[dataset["City"] == city, "Zip Code"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if st.session_state["zip_code"] not in zip_options:
    st.session_state["zip_code"] = zip_options[0]

zip_code = st.selectbox(
    "ZIP Code",
    zip_options,
    key="zip_code",
    help="Select a ZIP code available for the selected city.",
)

# Automatically determine geographic values
location_match = dataset[
    (dataset["City"] == city)
    & (dataset["Zip Code"] == zip_code)
]

if not location_match.empty:
    location_row = location_match.iloc[0]

    st.session_state["latitude"] = float(location_row["Latitude"])
    st.session_state["longitude"] = float(location_row["Longitude"])

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Latitude",
        format="%.6f",
        key="latitude",
        disabled=True,
        help="Automatically populated from the selected city and ZIP code.",
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        key="gender",
    )

with col3:
    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=0,
        key="tenure_months",
        help="Number of months the customer has been with the company.",
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Longitude",
        format="%.6f",
        key="longitude",
        disabled=True,
        help="Automatically populated from the selected city and ZIP code.",
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"],
        key="senior_citizen",
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
        key="phone_service",
    )

col1, col2, col3 = st.columns(3)

with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
        key="partner",
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
        key="dependents",
    )

with col3:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"],
        key="multiple_lines",
    )

st.caption(
    "ZIP code, latitude and longitude are linked to the selected location "
    "from the project dataset."
)


# Internet services

st.subheader("Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        key="internet_service",
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
        key="online_security",
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
        key="online_backup",
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
        key="device_protection",
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
        key="tech_support",
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
        key="streaming_tv",
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
        key="streaming_movies",
    )


# Contract and billing

st.subheader("Contract and Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        key="contract",
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
        key="paperless_billing",
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        key="payment_method",
    )

with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        key="monthly_charges",
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        key="total_charges",
    )

    cltv = st.number_input(
        "Customer Lifetime Value (CLTV)",
        min_value=0,
        key="cltv",
        help="Estimated customer lifetime value.",
    )


# Prediction

st.divider()

if st.button(
    "Predict Customer Churn",
    use_container_width=True,
):

    input_data = pd.DataFrame(
        {
            "City": [city],
            "Zip Code": [zip_code],
            "Latitude": [st.session_state["latitude"]],
            "Longitude": [st.session_state["longitude"]],
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
            "CLTV": [cltv],
        }
    )

    try:
        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        st.write(f"### Churn Risk Score: {probability:.2%}")

        if probability >= 0.70:
            st.error(
                " High Churn Risk",
                icon="⚠️",
            )
            st.write(
                "This customer is likely to churn based on the provided details."
            )

        elif probability >= 0.30:
            st.warning(
                " Medium Churn Risk",
                icon="⚠️",
            )
            st.write(
                "This customer has a moderate likelihood of churning."
            )

        else:
            st.success(
                " Low Churn Risk",
                icon="✅",
            )
            st.write(
                "This customer is unlikely to churn."
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)