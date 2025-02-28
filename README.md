# Demand Forecasting Using Machine Learning  

**Accurately predicting future demand to optimize inventory and supply chain management.**  

 **GitHub Repository:** [Demand Forecasting Project](https://github.com/himanshu-dandle/demand-forecasting)  

---

##  Project Overview  

Demand forecasting is essential for **retail and supply chain optimization**. This project uses **XGBoost, Random Forest, and Gradient Boosting** to analyze **historical sales data** and predict **future demand**.  

### 🔹 Why Demand Forecasting?  
 **Reduces inventory costs** by minimizing overstock and shortages  
 **Improves decision-making** for procurement and logistics  
 **Enhances customer satisfaction** with better product availability  

### 🔹 Key Features  
 **Data Cleaning & Feature Engineering** (date-based & categorical features)  
 **ML Model Training & Evaluation** (XGBoost, Random Forest, Gradient Boosting)  
 **AWS SageMaker Deployment** (for scalable predictions)  
 **Automated CI/CD Pipeline** (GitHub Actions for continuous deployment)  

---

## 📂 Project Structure  

📦 Demand_Forecasting_Project
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

CI/CD Pipeline (Auto-Deploy with GitHub Actions)
Every push to GitHub triggers a workflow:
 Runs tests
 Re-trains the model (if needed)
 Deploys the new model to AWS SageMaker
⚡ Want to enable GitHub Actions for automatic deployment? Let me know!

Future Enhancements
Hyperparameter Tuning: Optimize model parameters for better accuracy
Deep Learning Models: Implement LSTMs or Transformers for time series forecasting
External Data Sources: Include macroeconomic indicators for better predictions
Deploy as API: Use FastAPI or Flask for real-time predictions

 Contact & Contributions
 Author: Himanshu Dandle
 Email: HIMANSHU.DANDLE#GMAIL.COM
  GitHub: himanshu-dandle