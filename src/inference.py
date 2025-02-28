import os
import joblib
import json
import numpy as np

# 🔹 Load the model when the container starts
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "random_forest_model.pkl")  # Default SageMaker model path
    fallback_path = "/opt/ml/model/random_forest_model.pkl"  # Alternate path inside SageMaker

    print(f"🔍 Checking model paths: {model_path} | {fallback_path}")

    if os.path.exists(model_path):
        print(f"✅ Loading model from: {model_path}")
        return joblib.load(model_path)
    elif os.path.exists(fallback_path):
        print(f"✅ Loading model from: {fallback_path}")
        return joblib.load(fallback_path)
    else:
        raise FileNotFoundError(f"❌ Model file not found in {model_path} or {fallback_path}")

# 🔹 Process input request
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"])  # Ensure input format matches training
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 🔹 Make predictions
def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions.tolist()
