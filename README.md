# Telco Customer Churn Prediction  

## Project Overview  
This project focuses on predicting **customer churn** using machine learning classification techniques. The goal is to identify customers who are likely to discontinue the service, enabling businesses to take proactive measures for customer retention.  

The dataset comes from the **Telco Customer Churn dataset** with customer demographics, account information, and service usage details.  

---

## Dataset  
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** (7043, 21)  
- **Target:** `Churn` (Yes/No)  
- **Features:** gender, tenure, contract, payment method, internet service, monthly charges, total charges, etc.  

---

## Data Preprocessing  
- Removed **empty strings** and handled **null values**.  
- Scaled numerical columns using **StandardScaler**.  
- Encoded categorical columns using **Label Encoding**.  
- Applied **SMOTE** (Synthetic Minority Oversampling) to handle class imbalance.  

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
1. Trained a **base XGBoost classifier** to extract feature importances.  
   - With all 21 features:  
     - **Accuracy:** 0.780  
     - **ROC-AUC:** 0.817  

2. Selected features with importance **above the median** → **Top 9 features fixed**.  

3. Performed **iterative feature selection**:  
   - Added remaining features one by one to the fixed 9.  
   - Re-trained model each time and checked **ROC-AUC**.  
   - Finalized best-performing features.  

4. **Final dataset size:** (7760, 9 features).  

---

## Model Building & Evaluation  
Trained multiple classification models:  
- Gradient Boosting Classifier (GBC)  
- AdaBoost Classifier (ABC)  
- Bagging Classifier (BC)  
- Random Forest Classifier (RFC)  
- Support Vector Classifier (SVC)  
- Decision Tree (DT)  
- K-Nearest Classifier (KNC)  
- Stochastic Gradient Descent Classifier (SGDC)  
- Gaussian Naive Bayes (GNB)  

### Model Comparison  

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **GBC (Gradient Boosting)** | **0.7831** | **0.8383** |
| ABC (AdaBoost)              | 0.7564     | 0.8323     |
| BC (Bagging Classifier)     | 0.7553     | 0.7769     |
| RFC (Random Forest)         | 0.7530     | 0.7939     |
| SVC                         | 0.7445     | 0.8100     |
| DT (Decision Tree)          | 0.7433     | 0.7038     |
| KNC (K-Nearest)             | 0.7325     | 0.7623     |
| SGDC                        | 0.7314     | 0.8066     |
| GNB (Naive Bayes)           | 0.7104     | 0.8160     |

**Gradient Boosting Classifier (GBC)** performed the best across both **Accuracy** and **AUC**, making it the final chosen model.  

---

## Final Model Performance  

**Classification Report:**  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Churn) | 0.88 | 0.81 | 0.85 | 1294 |
| 1 (Churn)    | 0.58 | 0.70 | 0.63 | 467 |

- **Accuracy:** 0.7831  
- **ROC-AUC:** 0.8383  
- **Macro Avg F1:** 0.74  
- **Weighted Avg F1:** 0.79  
 
---

## Insights  
- Feature selection reduced dimensionality from **21 → 9 features** while preserving performance.  
- Model achieved a strong **ROC-AUC (0.8383)**, indicating good discrimination between churn and non-churn customers.  
- Recall for **churn class (1)** improved, making the model more effective for identifying at-risk customers.  

---
## Author  
**Abhinay Kalavakuri**

