# -*- coding: utf-8 -*-
"""
test_inference.py

This script sends test data to the deployed SageMaker model and gets predictions.
"""

import boto3
import json
import os
from dotenv import load_dotenv

# ✅ Load Environment Variables
load_dotenv()
ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

# ✅ Initialize SageMaker Runtime Client
runtime = boto3.client("sagemaker-runtime")

# ✅ Define Test Input (Modify as needed)
test_input = json.dumps({"instances": [[5.2, 3.1, 1.4, 0.2]]})

# ✅ Send Request to SageMaker Endpoint
print(f"🚀 Sending test input to '{ENDPOINT_NAME}'...")

try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=test_input
    )

    # ✅ Parse Response
    result = json.loads(response["Body"].read().decode())
    print("✅ Model Inference Output:", result)

except Exception as e:
    print(f"❌ Inference failed: {str(e)}")
