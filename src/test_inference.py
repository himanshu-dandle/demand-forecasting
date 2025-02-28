import boto3
import os
from dotenv import load_dotenv

# ✅ Load Environment Variables
load_dotenv()
ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

# ✅ Initialize SageMaker Runtime Client
runtime = boto3.client("sagemaker-runtime")

# ✅ Define Test Input in CSV Format (No Headers)
test_data = "5.2,3.1,1.4,0.2\n"  # Modify based on your dataset

print(f"🚀 Sending test input to '{ENDPOINT_NAME}'...")
print(f"📄 Payload: {test_data}")

try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",  # ✅ Use correct content type
        Body=test_data
    )

    # ✅ Parse Response
    result = response["Body"].read().decode("utf-8")
    print("✅ Model Inference Output:", result)

except Exception as e:
    print(f"❌ Inference failed: {str(e)}")
