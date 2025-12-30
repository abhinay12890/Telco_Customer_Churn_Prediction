# Telco Customer Churn Prediction (End-to-End ML System)

## Project Overview
This project implements a **production-ready, end-to-end machine learning system** to predict customer churn in a telecom environment.  
The focus is on **imbalanced classification**, **robust evaluation using PR-AUC**, **threshold-aware decision-making**, and **cloud-ready deployment**.

The project goes beyond model training and covers:
- Correct metric selection for imbalanced data
- Cross-validated feature selection
- Multi-model benchmarking
- Threshold tuning aligned with business objectives
- API, UI, and Docker-based deployment

---

## Dataset  
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** (7043, 21)  
- **Target:** `Churn` (Yes/No)  

---
## Project Structure
```
├── Customer_Churn_Classification.ipynb
├── app2.py                         # Gradio UI + FastAPI backend
├── best_churn_model.pkl            # Final trained model
├── feature_names.pkl               # Selected feature list
├── feature_encoding.pkl            # Category encoding maps
├── Dockerfile                      # DockerFile
├── requirements.txt                # Requirments for this project
├── README.md
```
---

## Data Preprocessing  
- Removed **empty strings** and handled **null values** in `totalcharges` column and converted into float datatype.   
- Encoded categorical columns using **Label Encoding**.
- Created category-to-integer encoding dictionary
- Ensured strict feature order alignment across training, API interface and Gradio UI
- Avoided data leakage by performing all transformations inside training folds
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
1. Trained a **base LightGBM classifier**  using 5-fold Stratified Cross-Validation
2. Evaluated folds using
   i. **PR-AUC (primary metrics)**
   ii. **ROC-AUC (secondary metric)**
3. Aggergated feature importances (gain-based)
4. Applied quantile-based feature selection (top 70%) and reduced features to 13.
---

## Model Benchmarking
Trained multiple tree-based classification models , since they perform better on tabular datasets with consistent pre-processing and evaluated using PR-RUC, which is appropriate for imbalanced dataset.



- Decision Tree (DT)
- Random Forest Classifier (RF)
- Gradient Boosting Classifier (GBC)
- AdaBoost Classifier (ABC)
- XGBoost Classifier (XGBC)
- LightGBM Classifier (LGBMC) 

### Model Comparison  

| Model                                  | PR-AUC    | ROC-AUC   |
| -------------------------------------- | --------- | --------- |
| **Gradient Boosting Classifier (GBC)** | **0.657** | **0.840** |
| AdaBoost Classifier (ABC)              | 0.650     | 0.840     |
| Random Forest (RF)                     | 0.646     | 0.839     |
| XGBoost (XGBC)                         | 0.635     | 0.827     |
| LightGBM (LGBM)                        | 0.610     | 0.817     |
| Decision Tree (DT)                     | 0.604     | 0.818     |



**Gradient Boosting Classifier (GBC)** performed the best across both **PR-AUC** and **ROC-AUC**, making it the final chosen model.  

---

## Final Model Performance

**Before Threshold tuning (default 0.5)**
**Classification Report:**  

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 (No Churn) | 0.84      | 0.91   | 0.87     | 1552    |
| 1 (Churn)    | 0.67      | 0.51   | 0.58     | 561     |

This provides high-confidence predictions but misses a portion of churners.
Since default probability threshold is incorrect for imbalanced problems, threshold turning is performed by analyzing
 - Precision
 - Recall
 - F1 Score
 - F2 Score
 
 **Aggressive Threshold (~0.10, F2-Optimized)**
| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 (No Churn) | 0.95      | 0.51   | 0.66     | 1552    |
| 1 (Churn)    | 0.40      | 0.92   | 0.56     | 561     |

Captures 92% of churners suitable for early-warning.

**Balanced Threshold (Selected for Deployment)**
| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 (No Churn) | 0.87      | 0.86   | 0.86     | 1552    |
| 1 (Churn)    | 0.62      | 0.65   | 0.63     | 561     |

- **Accuracy:** 0.80  
- **PR-AUC:** 0.657
- **ROC-AUC:** 0.840
 
---

## Insights  
- Feature selection reduced dimensionality from **21 → 13 features** while preserving performance.  
- Model achieved a strong **ROC-AUC (0.840)**, indicating good discrimination between churn and non-churn customers.  
- Threshold tuning increased recall for **churn class (1)** by **27%** from baseline model, making the model more effective for identifying at-risk customers.
- **Gradient Boosting Classifier (GBC)** provided best balance of predictive performance and training efficiency.

---
## Deployment Architechure
- Saved artificats: selected feature list, label-encoding mappings and final trained model (PKL files).

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
## Cloud Deployment (Render)
- Project has been deployed using
  - Docker-based deployment with Gradio UI as root path
  - FastAPI served alongside UI at /docs
 
**URLS (accessible online)** 
-[`https://telco-customer-churn-render.onrender.com/`](https://telco-customer-churn-render.onrender.com/) # For gradio
- [`https://telco-customer-churn-render.onrender.com/docs`](https://telco-customer-churn-render.onrender.com/docs) # for FastAPI
---
* Libraries & Tools Used: Pandas, Numpy, scikit-learn, joblib, matplotlib, seaborn, gradio, xgboost,lightgbm, imblearn, FastAPI, Uvicorn, Docker
  - Load pkl files using `var=joblib.load("file.pkl")`
---
## Author  
**Kalavakuri Abhinay**

