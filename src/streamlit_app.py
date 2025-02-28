import streamlit as st
import boto3
import os

# ✅ Load AWS credentials from Streamlit Secrets
AWS_REGION = st.secrets["AWS_REGION"]
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
SAGEMAKER_ENDPOINT = st.secrets["SAGEMAKER_ENDPOINT"]

# ✅ Initialize AWS SageMaker Runtime Client with secrets
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# ✅ Streamlit UI
st.title("🛒 Demand Forecasting App")
st.write("Enter product details to predict demand.")

# User input fields
store_id = st.number_input("Store ID", min_value=1, value=1)
sku_id = st.number_input("SKU ID", min_value=1, value=100)
total_price = st.number_input("Total Price ($)", min_value=1.0, value=10.0)
base_price = st.number_input("Base Price ($)", min_value=1.0, value=8.0)
is_featured_sku = st.selectbox("Is Featured SKU?", [0, 1])
is_display_sku = st.selectbox("Is Display SKU?", [0, 1])

# Button to predict
if st.button("Predict Demand"):
    # Prepare input as CSV format
    test_data = f"{store_id},{sku_id},{total_price},{base_price},{is_featured_sku},{is_display_sku}\n"

    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="text/csv",
        Body=test_data
    )

    # Get prediction result
    result = response["Body"].read().decode("utf-8")
    st.success(f"📈 Predicted Demand: {result}")
