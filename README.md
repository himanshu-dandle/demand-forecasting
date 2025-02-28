Demand Forecasting Using Machine Learning
 A Machine Learning-based Demand Forecasting system trained and deployed on AWS SageMaker
 
 
 📌 Overview
Demand forecasting is a crucial problem in supply chain management and retail. This project uses machine learning models like XGBoost, Random Forest, and Gradient Boosting to predict future product demand based on historical sales data.

🔹 Key Steps Covered:
✔ Data Preprocessing (handling missing values, outliers, feature engineering)
✔ Model Training (XGBoost, Random Forest, Gradient Boosting, Prophet)
✔ Model Evaluation (comparing MAE, MSE, R²)
✔ AWS SageMaker Deployment (automatic model deployment pipeline)
✔ CI/CD Pipeline (for automated deployment updates)

📂 Project Structure


Demand_Forecasting_Project/
│── data/                    # Raw & processed datasets  
│── models/                  # Trained models & artifacts  
│── notebooks/               # Jupyter Notebooks for each step  
│── plots/                   # Visualizations & saved plots  
│── src/                     # Python scripts for preprocessing & modeling  
│── forecasting_env/         # Virtual environment for dependencies  
│── model-config.json        # SageMaker model configuration  
│── bucket-policy.json       # S3 bucket permissions  
│── .env                     # AWS credentials (DO NOT COMMIT)  
│── .gitignore               # Files to ignore in Git  
│── README.md                # Project documentation  
│── 01_data_loading.ipynb    # Load & verify raw dataset  
│── 02_eda.ipynb             # Perform Exploratory Data Analysis (EDA)  
│── 03_feature_engineering.ipynb  # Feature selection & transformation  
│── 04_model_training.ipynb  # Train multiple models & select the best one  
│── 05_deployment_testing.ipynb  # Deploy to AWS SageMaker & test inference


📊 Model Performance Comparison
Model				MAE			MSE				R²
XGBoost				0.373834	0.251892	0.7462
Random Forest		0.375562	0.272121	0.7258
Gradient Boosting	0.573512	0.555499	0.4403
✅ XGBoost performed the best and was selected for deployment.

🛠 Technologies Used
Python (pandas, numpy, scikit-learn, XGBoost)
Jupyter Notebook (for EDA & training)
AWS SageMaker (for model deployment)
Amazon S3 (for storing datasets & models)
GitHub Actions (for CI/CD automation)
🚀 Deployment Process
1️⃣ Train the Model (04_model_training.ipynb)
After feature engineering, models are trained and the best-performing model is saved as .pkl.

2️⃣ Convert & Package Model (05_deployment_testing.ipynb)
Convert .pkl → .model (XGBoost native format)
Package as .tar.gz (required for SageMaker)
Upload to Amazon S3
3️⃣ Deploy Model to AWS SageMaker
Automatically delete previous endpoint if exists
Deploy new model & wait until InService
Test inference on deployed model
✅ The entire process is automated in 05_deployment_testing.ipynb.

📌 How to Run This Project
1️⃣ Clone the Repository
git clone https://github.com/himanu-dandle/demand-forecasting.git
cd demand-forecasting

2️⃣ Set Up Virtual Environment & Install Dependencies
python -m venv forecasting_env
source forecasting_env/bin/activate  # For Mac/Linux
forecasting_env\Scripts\activate     # For Windows
pip install -r requirements.txt

3️⃣ Configure AWS Credentials
Create a .env file in the project root (DO NOT COMMIT THIS FILE).
SAGEMAKER_BUCKET=demand-forecasting-bucket-us-east-1
SAGEMAKER_ROLE=arn:aws:iam::060795905003:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX

4️⃣ Train the Model
jupyter notebook 04_model_training.ipynb

5️⃣ Deploy the Model to AWS SageMaker
jupyter notebook 05_deployment_testing.ipynb

📌 Inference: Making Predictions with Deployed Model
Once the model is deployed, you can send real-time inference requests:

python


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
print("✅ Prediction:", result)

🚀 CI/CD Pipeline (Auto-Deploy with GitHub Actions)
Every time new code is pued to GitHub, the CI/CD pipeline:
✅ Runs tests
✅ Re-trains the model (if needed)
✅ Deploys the new model to AWS SageMaker
⚡ Want to enable GitHub Actions for automatic deployment? Let me know!

📌 Future Improvements
🔹 Add Deep Learning models (LSTMs, Transformers for time series forecasting)
🔹 Enable Hyperparameter tuning for better performance
🔹 Set up API Gateway for model inference requests

📩 Contact & Contributions
🙋‍♂️ Author: Himanshu Dandle
📧 Email: himanshu.dandle@gmail.com
🔗 GitHub: himanshu-dandle



