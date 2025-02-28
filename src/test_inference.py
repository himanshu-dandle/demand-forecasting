# -*- coding: utf-8 -*-
"""
test_inference.py

Sends test data to the deployed SageMaker XGBoost model and gets predictions.
"""

import boto3
import os
from dotenv import load_dotenv

# ✅ Step 1: Load Environment Variables
load_dotenv()
ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

# ✅ Step 2: Initialize SageMaker Runtime Client
runtime = boto3.client("sagemaker-runtime")

# ✅ Step 3: Define Test Input in CSV Format (No Headers, Comma-Separated)
test_data = "5.2,3.1,1.4,0.2\n"  # Modify based on your dataset

print(f"🚀 Sending test input to '{ENDPOINT_NAME}'...")
print(f"📄 Payload: {test_data}")

# ✅ Step 4: Send Request to SageMaker Endpoint
try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",  # ✅ Corrected to text/csv
        Body=test_data
    )

    # ✅ Step 5: Parse Response
    result = response["Body"].read().decode("utf-8")
    print("✅ Model Inference Output:", result)

except Exception as e:
    print(f"❌ Inference failed: {str(e)}")
