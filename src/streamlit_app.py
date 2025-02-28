import streamlit as st
import boto3

# Define SageMaker endpoint
ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

# Initialize AWS SageMaker Runtime Client
runtime = boto3.client("sagemaker-runtime")

# Streamlit UI
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
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=test_data
    )

    # Get prediction result
    result = response["Body"].read().decode("utf-8")
    st.success(f"📈 Predicted Demand: {result}")
