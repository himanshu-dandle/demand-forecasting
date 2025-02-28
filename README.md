## ***🛒 Demand Forecasting using AWS SageMaker & XGBoost***
🔗 GitHub Repository: demand-forecasting ( [Demand Forecasting Project](https://github.com/himanshu-dandle/demand-forecasting)  )

## 📈 Project Overview:
This project builds a demand forecasting model using XGBoost, deployed on AWS SageMaker with a CI/CD pipeline using GitHub Actions.

✅ Key Features:
**Data Preprocessing & Feature Engineering 
✔ **Model Training with XGBoost 📊
✔ **Automated Deployment to AWS SageMaker 🚀
✔ **Real-time Inference via API 🔮
✔ **CI/CD Pipeline for Model Deployment 🔄


### Why Demand Forecasting?  
 **Reduces inventory costs** by minimizing overstock and shortages  
 **Improves decision-making** for procurement and logistics  
 **Enhances customer satisfaction** with better product availability
 

## Dataset Details
***The dataset used in this project comes from **Kaggle** and contains historical sales data for demand forecasting.

***📌 Source: Kaggle - Demand Forecasting Dataset

The dataset includes:

***record_ID – Unique record identifier
***week – Weekly sales data
***store_id – Store identifier
***sku_id – Stock Keeping Unit (SKU) identifier
***total_price – Total revenue generated
***units_sold – Target variable (demand forecast)
✅ Data Preprocessing Includes:
✔ ***Handling Missing Values & Outliers
✔ ***Feature Engineering (Year, Month, Week Number, etc.)
✔ ***Scaling & Encoding for Model Training



### Key Features  
 **Data Cleaning & Feature Engineering** (date-based & categorical features)  
 **ML Model Training & Evaluation** (XGBoost, Random Forest, Gradient Boosting)  
 **AWS SageMaker Deployment** (for scalable predictions)  
 **Automated CI/CD Pipeline** (GitHub Actions for continuous deployment)  

---

## 📂 Project Structure  
*** demand-forecasting/
***│── data/                   # Raw & processed datasets  
***│── models/                 # Trained models & artifacts  
***│── notebooks/              # Jupyter Notebooks for EDA & training  
***│── plots/                  # Visualizations  
***│── src/                    # Python scripts for processing & training  
***│   ├── 04_model_training.py       # Train & save the XGBoost model  
***│   ├── 05_deployment_testing.py   # Deploy & test inference on SageMaker  
***│   ├── test_inference.py          # Run batch inference  
***│── deployment/             # Deployment configurations  
***│── .github/workflows/      # GitHub Actions CI/CD  
***│── .env                    # AWS credentials (ignored in GitHub)  
│***── README.md               # Project documentation  


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

##  🎯 How to Run This Project  

### 1️ Clone the Repository  
```
git clone https://github.com/himanshu-dandle/demand-forecasting.git
cd Demand_Forecasting_Project
```

### 2️ Set Up Virtual Environment & Install Dependencies

python -m venv forecasting_env
source forecasting_env/bin/activate  # Mac/Linux
forecasting_env\Scripts\activate     # Windows
pip install -r requirements.txt

### 3️ configure AWS Credentials
Create a .env file in the project root (DO NOT COMMIT THIS FILE).
	SAGEMAKER_BUCKET=demand-forecasting-bucket-us-east-1
	SAGEMAKER_ROLE=arn:aws:iam::060795905003:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX

###4️ Train the Model
	
	python src/04_model_training.py
	
###5️ Deploy the Model to AWS SageMaker
	python src/05_deployment_testing.py


###6. Making Predictions with the Deployed Model
   python src/test_inference.py

##🚀 Deployment Pipeline (AWS SageMaker & CI/CD)

**This project includes automated model deployment via GitHub Actions.

 Workflow File: .github/workflows/deploy.yml
 Triggers: Runs on every push to main branch

**🔄 What Happens in CI/CD?
***Checks out the latest code from GitHub
***Installs dependencies inside a virtual environment
***Deploys the trained model to AWS SageMaker
***Runs inference tests to verify deployment


**Environment Variables & Secrets
This project uses GitHub Secrets for secure access to AWS credentials:

**Secret Name	Description
**AWS_ACCESS_KEY_ID		AWS Access Key for authentication
**AWS_SECRET_ACCESS_KEY	AWS Secret Key for authentication
**SAGEMAKER_BUCKET		S3 bucket name for storing model files
**SAGEMAKER_ROLE			AWS IAM Role for SageMaker execution

**📌 To set up GitHub Secrets:

***Go to your GitHub repo → "Settings" → "Secrets and variables" → "Actions"
***Add the required secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)


***🔧 Deployment Steps
**Model Training (04_model_training.py)
		Trains XGBoost model & saves it as xgboost_model.pkl
***Model Packaging (05_deployment_testing.py)
	**Converts .pkl to SageMaker-compatible .model
	**Packages the model into xgboost_model.tar.gz
	**Uploads to AWS S3
	**AWS SageMaker Deployment
	**Creates a SageMaker Model
	**Deploys an Inference Endpoint
	**Waits for deployment to be InService
	**GitHub Actions CI/CD Pipeline
	**Triggers deployment on every push to main branch
	**Deletes old models, endpoints & redeploys the latest version

***GitHub Actions Workflow:
```
name: SageMaker Deployment Pipeline

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: 🚀 Checkout Repository
        uses: actions/checkout@v3

      - name: 🐍 Set Up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'

      - name: 📦 Install Dependencies
        run: |
          python -m venv forecasting_env
          source forecasting_env/bin/activate
          pip install --no-cache-dir -r requirements.txt
          pip install --upgrade boto3 sagemaker

      - name: 🔑 Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: 🏗️ Deploy Model to SageMaker
        run: |
          source forecasting_env/bin/activate
          python src/05_deployment_testing.py

      - name: ✅ Deployment Completed
        run: echo "Deployment to AWS SageMaker is successful!"
```

##🔬 Model Inference (Real-Time Predictions)
Once deployed, predictions can be made using the SageMaker Inference API

***1️. Invoke Endpoint Using AWS CLI
```
	aws sagemaker-runtime invoke-endpoint \
    --endpoint-name xgboost-demand-forecasting-endpoint \
    --content-type "text/csv" \
    --body "5.2,3.1,1.4,0.2" \
    output.json --region us-east-1
```
***2.Invoke Endpoint Using Python
```
import boto3

ENDPOINT_NAME = "xgboost-demand-forecasting-endpoint"
runtime = boto3.client("sagemaker-runtime")

test_data = "5.2,3.1,1.4,0.2\n"  # CSV format

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="text/csv",
    Body=test_data
)

result = response["Body"].read().decode("utf-8")
print("✅ Model Inference Output:", result)
```


##Future Enhancements
***Hyperparameter Tuning: Optimize model parameters for better 
***Feature Engineering Improvements (Lags, Moving Averages)
***Deep Learning Models: Implement LSTMs or Transformers for time series forecasting
***External Data Sources: Include macroeconomic indicators for better predictions
***Deploy as API: Use FastAPI or Flask for real-time predictions
***Scale Deployment with AWS Lambda

##Troubleshooting Guide
***Issue: Deployment fails with ModelError (415) - application/json is not an accepted ContentType
***Fix: Update test_inference.py to use "text/csv" instead of "application/json".

***Issue: FileNotFoundError: models/xgboost_model.pkl not found in CI/CD
***Fix: Either commit the .pkl file or upload it to S3 before deployment.

***Issue: Model is InService, but inference fails
***Fix: Check AWS CloudWatch Logs for error details.



***Contact & Contributions
***Author: Himanshu Dandle
***Email: HIMANSHU.DANDLE#GMAIL.COM
***GitHub: himanshu-dandle