# Telco Customer Churn Prediction  

## Project Overview  
This project builds a complete machine learning system to identify customers who are likely to discontinue the service, enabling businesses to take proactive measures for customer retention. 
This project includes:
  * **Full ML Workflow** - EDA, feature engineering, model selection
  * **Production-ready model** *(LightGBM)*
  * **FastAPI REST API** for programmatic predictions
  * **Gradio Web App** for interactive predictions
  * **Cloud deployment on Railway**
 

---

## Dataset  
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** (7043, 21)  
- **Target:** `Churn` (Yes/No)  
- **Features:** Contract, Internet Service, Phone Service, Tenure, TechSupport, SeniorCitizen, MonthlyCharges, OnlineSecurity, PaperlessBilling, OnlineBackup, MultipleLines, PaymentMethod, TotalCharges, DeviceProtection.

---
## File Structure 
```
├── Customer_Churn_Classification.ipynb   # EDA + ML training notebook
├── app2.py                               # FastAPI + Gradio unified backend
├── best_churn_model.pkl                  # Final Trained LightGBM model
├── feature_names.pkl                     # List of selected feature names (14) used for prediction
├── label_encoding.pkl                    # Label encoder mappings for categorical variables
├── yes_col_names.pkl                     # Encoded columns corresponding to 'Yes/No' categorical values
├── Dockerfile                            # Production Docker Container
├── requirements.txt                      # List of dependencies for the project
├── README.md                             # Project documentation (this file)

```


---

## Data Preprocessing  
- Removed **empty strings** and handled **null values** in `totalcharges` column and converted into float datatype.   
- Encoded categorical columns using **Label Encoding**.
- Mapped binary yes/no columns to 1/0
- Applied **SMOTE** (Synthetic Minority Oversampling) in the training dataset to handle class imbalance. Length of dataset before **SMOTE**: 5282 ; After **SMOTE**: 7760.
---

## Exploratory Data Analysis (EDA)  
- Visualized churn distribution using seaborn, matplotlib  
- Analyzed customer behavior with respect to churn:  
  - Contract type  
  - Payment method  
  - Internet service usage  
  - Tenure and monthly charges  

---

## Feature Selection  
1. Trained a **base XGBoost classifier** to extract feature importances and converted into DataFrame containing column names and their importances.  
   - With all 19 features:  
     - **Accuracy:** 0.778  
     - **ROC-AUC:** 0.818  

2. Selected features contributing **90% cumulative importance** → **Top features finalized**(14). 

3. **Final training dataset size:** (7760, 14 features).  

---

## Model Building & Evaluation  
Trained multiple tree-based classification models , since they perform better on tabular datasets.

- Decision Tree (DT)
- Random Forest Classifier (RFC)
- Gradient Boosting Classifier (GBC)
- AdaBoost Classifier (ABC)
- Bagging Classifier (BC)
- XGBoost Classifier (XGBC)
- LightGBM Classifier (LGBMC) 

### Model Comparison  

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **LGBMC (LightGBM Classifier)** | **0.77797** | **0.82875** |
| RFC (Random Forest Classifier)              | 0.77115     | 0.82001     |
| GBC (Gradient Boosting Classifier)     | 0.77058     | 0.83570     |
| BC (Bagging Classifier)         | 0.76661     | 0.79104     |
| XGBC (XGBoost Classifier)                     | 0.76604     | 0.81832     |
| ABC (AdaBoost Classifier)          | 0.75639     | 0.83326     |
| DT (Decision Tree Classifier)             | 0.73311     | 0.68427     |


**LightGBM Classifier (LGBMC)** performed the best across both **Accuracy** and **AUC**, making it the final chosen model.  

---

## Final Model Performance

**Before Threshold tuning (default 0.5)**
**Classification Report:**  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Churn) | 0.87 | 0.82 | 0.84 | 1294 |
| 1 (Churn)    | 0.57 | 0.67 | 0.61 | 467 |

- **Accuracy:** 0.78  
- **ROC-AUC:** 0.829
  
 **After Threshold tuning (selected threshold 0.4 on probability)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Churn) | 0.89 | 0.77 | 0.82 | 1294 |
| 1 (Churn)    | 0.53 | 0.73 | 0.62 | 467 |

- **Accuracy:** 0.76  
- **ROC-AUC:** 0.829
 
---

## Insights  
- Feature selection reduced dimensionality from **21 → 14 features** while preserving performance.  
- Model achieved a strong **ROC-AUC (0.829)**, indicating good discrimination between churn and non-churn customers.  
- Threshold tuning increased recall for **churn class (1)** by **8.9%**, making the model more effective for identifying at-risk customers.
- **LightGBMClassifier** provided best balance of predictive performance and training efficiency.

---
## Deployment Architechure
- Saved artificats: selected feature list, label-encoding mappings, yes/no column list and final trained model (PKL files).

This project includes 2 deployment interfaces:
### 1. Interactive Gradio App (Frontend UI)
- Build a Gradio interface for interactive churn prediction with:
  -  Dropdown for categorical and binary yes/no features
  -  Numeric Inputs for continuous variables.
  -  Displays prediction: **Churn / No Churn** with probability
### 2. FastAPI (Backend)
- Endpoint `/predict`
- API documentation available at `/docs`
**Example JSON input** \
  `{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PhoneService": "Yes",
  "tenure": 12,
  "TechSupport": "No",
  "SeniorCitizen": "No",
  "MonthlyCharges": 65.4,
  "OnlineSecurity": "No",
  "PaperlessBilling": "Yes",
  "OnlineBackup": "No",
  "MultipleLines": "No",
  "PaymentMethod": "Electronic check",
  "TotalCharges": 785.25,
  "DeviceProtection": "No"
}`

## Docker Containerization
- `docker build -t churn_api .` # building the image
- `docker run -p 8000:8000 churn_api` # running locally
- `http://localhost:8000/` # access UI
- Docker image available at [abhinay1289/customer_api](https://hub.docker.com/repository/docker/abhinay1289/customer_api/)
## Cloud Deployment (Railway)
- Project has been deployed using
  - Docker-based deployment with Gradio UI as root path
  - FastAPI served alongside UI at /docs
 
**URLS (accessible online)** 
- `https://customerapi-production-8ca6.up.railway.app/` # For gradio
- `https://customerapi-production-8ca6.up.railway.app/docs` # for FastAPI
---
* Libraries & Tools Used: Pandas, Numpy, scikit-learn, joblib, matplotlib, seaborn, gradio, xgboost,lightgbm, imblearn, FastAPI, Uvicorn, Docker
  Load pkl files using `var=joblib.load("file.pkl")`
---
## Author  
**Kalavakuri Abhinay**

