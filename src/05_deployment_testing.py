# -*- coding: utf-8 -*-
"""
05_deployment_testing.py

Deploys the trained XGBoost model to AWS SageMaker.
"""

import boto3
import os
import tarfile
import joblib
import json
import time
import xgboost as xgb
from dotenv import load_dotenv
from sagemaker import image_uris

# ✅ Step 1: Load AWS Credentials from GitHub Secrets
load_dotenv()

BUCKET_NAME = os.getenv("SAGEMAKER_BUCKET")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE")

if not BUCKET_NAME or not ROLE_ARN:
    raise ValueError("❌ Missing AWS credentials in environment variables!")

# ✅ Step 2: Define Model & Endpoint Names
MODEL_NAME = "xgboost-demand-forecasting"
ENDPOINT_CONFIG_NAME = MODEL_NAME + "-config"
ENDPOINT_NAME = MODEL_NAME + "-endpoint"
MODEL_FILE = "xgboost_model.tar.gz"
MODEL_URI = f"s3://{BUCKET_NAME}/{MODEL_FILE}"

# ✅ Step 3: Get the Correct XGBoost Image URI
region = boto3.Session().region_name
xgboost_image_uri = image_uris.retrieve("xgboost", region, version="1.5-1")

# ✅ Step 4: Initialize AWS Clients
sagemaker_client = boto3.client("sagemaker")
s3 = boto3.client("s3")

# ✅ Step 5: Delete Existing Model, Endpoint & Config
def delete_if_exists():
    resources = {
        "Endpoint": (ENDPOINT_NAME, sagemaker_client.delete_endpoint, sagemaker_client.describe_endpoint),
        "EndpointConfig": (ENDPOINT_CONFIG_NAME, sagemaker_client.delete_endpoint_config, sagemaker_client.describe_endpoint_config),
        "Model": (MODEL_NAME, sagemaker_client.delete_model, sagemaker_client.describe_model),
    }

    for resource, (name, delete_func, describe_func) in resources.items():
        try:
            if resource == "Endpoint":
                describe_func(EndpointName=name)
                delete_func(EndpointName=name)
            elif resource == "EndpointConfig":
                describe_func(EndpointConfigName=name)
                delete_func(EndpointConfigName=name)
            elif resource == "Model":
                describe_func(ModelName=name)
                delete_func(ModelName=name)

            print(f"✅ Deleted existing {resource}: {name}")
        except sagemaker_client.exceptions.ClientError:
            print(f"✅ No existing {resource} found.")

delete_if_exists()

# ✅ Step 6: Convert .pkl Model → XGBoost Native `.model` & Package
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

pkl_model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
if not os.path.exists(pkl_model_path):
    raise FileNotFoundError(f"❌ Model not found at {pkl_model_path}. Did you run `04_model_training.py`?")

print(f"✅ Found trained model: {pkl_model_path}")

# Load model & save in XGBoost's native format
xgb_model = joblib.load(pkl_model_path)
xgb_model_path = os.path.join(MODEL_DIR, "xgboost_model.model")
xgb_model.save_model(xgb_model_path)

# Set correct file permissions & create .tar.gz package
os.chmod(xgb_model_path, 0o644)
tar_path = os.path.join(MODEL_DIR, MODEL_FILE)

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(xgb_model_path, arcname="xgboost_model.model")

print(f"✅ Model packaged at {tar_path}")

# ✅ Step 7: Upload Model to S3
s3.upload_file(tar_path, BUCKET_NAME, MODEL_FILE)
print(f"✅ Model uploaded to S3: {MODEL_URI}")

# ✅ Step 8: Create SageMaker Model
print("🚀 Creating new SageMaker model...")
sagemaker_client.create_model(
    ModelName=MODEL_NAME,
    PrimaryContainer={"Image": xgboost_image_uri, "ModelDataUrl": MODEL_URI},
    ExecutionRoleArn=ROLE_ARN
)
print(f"✅ SageMaker model '{MODEL_NAME}' created.")

# ✅ Step 9: Create Endpoint Configuration
print("🚀 Creating new endpoint configuration...")
sagemaker_client.create_endpoint_config(
    EndpointConfigName=ENDPOINT_CONFIG_NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": MODEL_NAME,
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1
    }]
)
print(f"✅ Endpoint config '{ENDPOINT_CONFIG_NAME}' created.")

# ✅ Step 10: Deploy SageMaker Endpoint
print(f"🚀 Deploying SageMaker endpoint '{ENDPOINT_NAME}'... (This may take a few minutes)")
sagemaker_client.create_endpoint(
    EndpointName=ENDPOINT_NAME,
    EndpointConfigName=ENDPOINT_CONFIG_NAME
)

# ✅ Step 11: Wait for Deployment to Complete
while True:
    response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response["EndpointStatus"]
    print(f"🔍 Current Status: {status}")

    if status == "InService":
        print(f"✅ SageMaker Endpoint '{ENDPOINT_NAME}' is ready for inference!")
        break
    elif status in ["Failed", "RollingBack"]:
        failure_reason = response.get("FailureReason", "Unknown error")
        print(f"❌ Deployment failed! Reason: {failure_reason}")
        raise SystemExit(f"❌ Deployment failed: {failure_reason}")

    time.sleep(30)

print("🎯 Deployment process completed successfully!")

# ✅ Step 12: Test Model Inference
print("🚀 Testing inference on deployed model...")
runtime = boto3.client("sagemaker-runtime")

# Create a sample input (Modify based on your data)
test_input = json.dumps({"instances": [[5.2, 3.1, 1.4, 0.2]]})

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=test_input
)

# Parse response
result = json.loads(response["Body"].read().decode())
print("✅ Model Inference Output:", result)
