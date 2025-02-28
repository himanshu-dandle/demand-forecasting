# -*- coding: utf-8 -*-
"""
test_inference.py

Sends batch test data to the deployed SageMaker XGBoost model and stores predictions.
"""

import boto3
import os
import csv

# ✅ Step 1: Define SageMaker Endpoint
ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

# ✅ Step 2: Initialize SageMaker Runtime Client
runtime = boto3.client("sagemaker-runtime")

# ✅ Step 3: Read Test Input from CSV File
INPUT_FILE = "data/test_input.csv"  # Test input file
OUTPUT_FILE = "data/predictions.csv"  # File to store predictions

# Read all test rows from CSV
with open(INPUT_FILE, "r") as f:
    test_data = f.read().strip()  # Read all content & remove extra spaces

print(f"🚀 Sending batch test input to '{ENDPOINT_NAME}'...")
print(f"📄 Payload:\n{test_data}")

# ✅ Step 4: Send Request to SageMaker Endpoint
try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",  # ✅ Correct format
        Body=test_data
    )

    # ✅ Step 5: Parse & Save Response
    predictions = response["Body"].read().decode("utf-8").strip().split("\n")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Data", "Prediction"])  # Header
        for row, pred in zip(test_data.split("\n"), predictions):
            writer.writerow([row, pred])

    print(f"✅ Predictions saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"❌ Inference failed: {str(e)}")
