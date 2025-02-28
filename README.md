# Demand Forecasting Using Machine Learning  

Accurately predicting future demand to optimize inventory and supply chain management.** 
Live Deployed Model AWS SageMaker Endpoint: xgboost-demand-forecasting-endpoint (Private)
GitHub Repository: [Demand Forecasting Project](https://github.com/himanshu-dandle/demand-forecasting)  
 
---

##  Project Overview  

This project builds a demand forecasting system using machine learning models such as XGBoost, Random Forest, and Gradient Boosting. The model predicts product demand based on historical sales data, price fluctuations, and store performance.

✔ End-to-End Pipeline from EDA → Model Training → Deployment
✔ Deployed on AWS SageMaker for scalable real-time inference
✔ Automated CI/CD with GitHub Actions


### Why Demand Forecasting?  
 **Reduces inventory costs** by minimizing overstock and shortages  
 **Improves decision-making** for procurement and logistics  
 **Enhances customer satisfaction** with better product availability
 

## Dataset Details
The dataset used in this project comes from **Kaggle** and contains historical sales data for demand forecasting.

📌 **Source:** [Kaggle - Demand Forecasting Dataset](https://www.kaggle.com/) (Replace with actual link)

✔ **You need to manually download the dataset from Kaggle** before running the project.  
✔ **Place the dataset in the `data/` directory** before training the model.

The dataset contains the following features:

1. **Column	Description
2. **record_ID	Unique record identifier
3. **week	Sales week (timestamp)
4. **store_id	Store identifier
5. **sku_id	Stock Keeping Unit (Product ID)
6. **total_price	Total revenue generated
7. ** base_price	Product base price
8. **is_featured_sku	Whether the product was promoted
9. **is_display_sku	Whether the product was displayed prominently
10. **units_sold	Target Variable - Number of units sold
11. **year, month, week_num, quarter, day_of_week	Extracted time features




### 🔹 Key Features  
 **Data Cleaning & Feature Engineering** (date-based & categorical features)  
 **ML Model Training & Evaluation** (XGBoost, Random Forest, Gradient Boosting)  
 **AWS SageMaker Deployment** (for scalable predictions)  
 **Automated CI/CD Pipeline** (GitHub Actions for continuous deployment)  

---

## 📂 Project Structure  
📂 Demand_Forecasting_Project/
├── 📁 data/                 # Raw & processed datasets (not committed to GitHub)  
├── 📁 models/               # Trained models & artifacts  
├── 📁 notebooks/            # Jupyter Notebooks for EDA & training  
├── 📁 src/                  # Python scripts for model training & deployment  
│   ├── 01_data_loading.py  
│   ├── 02_eda.py  
│   ├── 03_feature_engineering.py  
│   ├── 04_model_training.py  
│   ├── 05_deployment_testing.py  
│   ├── test_inference.py  # Inference testing script  
├── 📁 deployment/           # Deployment-related configuration files  
├── 📁 .github/workflows/    # GitHub Actions CI/CD pipeline  
│   ├── deploy.yml  
├── 📄 requirements.txt      # Python dependencies  
├── 📄 .gitignore            # Ignored files (data, logs, secrets)  
├── 📄 README.md             # Project documentation  


---

##  Model Performance Evaluation  

| Model               	 	| MAE       	| MSE        | R²        |
|----------------------|--------------------|------------|-----------|
| **XGBoost**         		| 0.3738 	 	| 0.2518     | **0.7462**|
| **Random Forest**   		| 0.3755        | 0.2721     | 0.7258    |
| **Gradient Boosting** 	| 0.5735        | 0.5554     | 0.4403    |

 **XGBoost performed the best** and was selected for deployment.  

---

##  Technologies Used  
- **Programming:** Python (pandas, numpy, scikit-learn, XGBoost)  
- **Notebook Environment:** Jupyter Notebook  
- **Cloud Platform:** AWS SageMaker (for deployment)  
- **Storage:** Amazon S3 (for model storage)  
- **CI/CD Pipeline:** GitHub Actions (for automatic deployment)  

---

##  Step-by-Step Guide to Running the Project  

### 1️ Clone the Repository  
```
git clone https://github.com/himanshu-dandle/demand-forecasting.git
cd Demand_Forecasting_Project


2️ Set Up Virtual Environment & Install Dependencies

python -m venv forecasting_env
source forecasting_env/bin/activate  # Mac/Linux
forecasting_env\Scripts\activate     # Windows
pip install -r requirements.txt

3️ configure AWS Credentials
Create a .env file in the project root (DO NOT COMMIT THIS FILE).
	SAGEMAKER_BUCKET=demand-forecasting-bucket-us-east-1
	SAGEMAKER_ROLE=arn:aws:iam::060795905003:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX

4️ Train the Model
	jupyter notebook notebooks/04_model_training.ipynb
	
5️ Deploy the Model to AWS SageMaker
	jupyter notebook notebooks/05_deployment_testing.ipynb

 Making Predictions with the Deployed Model
Once the model is deployed, send real-time inference requests:

import boto3
import json

runtime = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"

test_input = json.dumps({"instances": [[5.2, 3.1, 1.4, 0.2]]})  # Modify as needed

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=test_input
)

result = json.loads(response["Body"].read().decode())
print("Prediction:", result)

 CI/CD Workflow (GitHub Actions)
This project includes automated model deployment via GitHub Actions.

 Workflow File: .github/workflows/deploy.yml
 Triggers: Runs on every push to main branch

🔄 What Happens in CI/CD?
Checks out the latest code from GitHub
Installs dependencies inside a virtual environment
Deploys the trained model to AWS SageMaker
Runs inference tests to verify deployment


Environment Variables & Secrets
This project uses GitHub Secrets for secure access to AWS credentials:

Secret Name	Description
AWS_ACCESS_KEY_ID		AWS Access Key for authentication
AWS_SECRET_ACCESS_KEY	AWS Secret Key for authentication
SAGEMAKER_BUCKET		S3 bucket name for storing model files
SAGEMAKER_ROLE			AWS IAM Role for SageMaker execution
📌 To set up GitHub Secrets:

Go to your GitHub repo → "Settings" → "Secrets and variables" → "Actions"
Add the required secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)


Future Enhancements
Hyperparameter Tuning: Optimize model parameters for better 
Feature Engineering Improvements (Lags, Moving Averages)
Deep Learning Models: Implement LSTMs or Transformers for time series forecasting
External Data Sources: Include macroeconomic indicators for better predictions
Deploy as API: Use FastAPI or Flask for real-time predictions
Scale Deployment with AWS Lambda

Troubleshooting Guide
Issue: Deployment fails with ModelError (415) - application/json is not an accepted ContentType
Fix: Update test_inference.py to use "text/csv" instead of "application/json".

Issue: FileNotFoundError: models/xgboost_model.pkl not found in CI/CD
Fix: Either commit the .pkl file or upload it to S3 before deployment.

Issue: Model is InService, but inference fails
Fix: Check AWS CloudWatch Logs for error details.



 Contact & Contributions
 Author: Himanshu Dandle
 Email: HIMANSHU.DANDLE#GMAIL.COM
  GitHub: himanshu-dandle