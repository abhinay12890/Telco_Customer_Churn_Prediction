# Telco Customer Churn Prediction  

## Project Overview  
This project focuses on predicting **customer churn** using machine learning classification techniques. The goal is to identify customers who are likely to discontinue the service, enabling businesses to take proactive measures for customer retention.  

The dataset comes from the **Telco Customer Churn dataset** with customer demographics, account information, and service usage details.  

---

## Dataset  
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** (7043, 21)  
- **Target:** `Churn` (Yes/No)  
- **Features:** Contract, Internet Service, Phone Service, Tenure, TechSupport, SeniorCitizen, MonthlyCharges, OnlineSecurity, PaperlessBilling, OnlineBackup, MultipleLines, PaymentMethod, TotalCharges, DeviceProtection.

---
## File Structure 
├── Customer_Churn_Classification.ipynb   # Jupyter notebook with EDA, model training & evaluation
├── app.py                                # Gradio web app for deployment
├── best_churn_model.pkl                  # Trained machine learning model
├── feature_names.pkl                     # List of selected feature names used for prediction
├── label_encoding.pkl                    # Label encoder mappings for categorical variables
├── yes_col_names.pkl                     # Encoded column names corresponding to 'Yes/No' categorical values
├── requirements.txt                      # List of dependencies for the project (deployment side)
├── README.md                             # Project documentation (this file)


---

## Data Preprocessing  
- Removed **empty strings** and handled **null values** in `totalcharges` column and converted into float datatype.   
- Encoded categorical columns using **Label Encoding**.
- Mapped binary yes/no columns to 1/0
- Applied **SMOTE** (Synthetic Minority Oversampling) in the training dataset to handle class imbalance. Length of dataset before **SMOTE**: 5282 ; After **SMOTE**: 7760.

---

## Exploratory Data Analysis (EDA)  
- Visualized churn distribution.  
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

**Before Threshold tuning**
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
## Deployment
- Saved artificats: selected feature list, label-encoding mappings, yes/no column list and final trained model (PKL files).
- Build a Gradio interface for interactive churn prediction with:
  -  Dropdown for categorical and binary yes/no features
  -  Numeric Inputs for continuous variables.
- Deployed the Gradio app on **Hugging Face Spaces** for live predicitons.
---
* Libraries Used: Pandas, Numpy, scikit-learn, joblib, matplotlib, seaborn, gradio, xgboost,lightgbm, imblearn.
  Load pkl files using `var=joblib.load("file.pkl")`
---
## Author  
**Kalavakuri Abhinay**

