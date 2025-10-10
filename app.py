import joblib
import pandas as pd
import gradio as gr


yes_col=joblib.load("yes_col_names.pkl")
encode=joblib.load("label_encoding.pkl")
yes_map={"Yes":1,"No":0}
best_model=joblib.load("best_churn_model.pkl")
rev=joblib.load("feature_names.pkl")

yes_col.append("SeniorCitizen")
# Function to Predict Churn for Deployment
def predict_churn(Contract, InternetService, PhoneService, tenure, TechSupport, SeniorCitizen, MonthlyCharges, OnlineSecurity,
 PaperlessBilling,OnlineBackup,MultipleLines,PaymentMethod,TotalCharges,DeviceProtection):
    inputs = {
        'Contract':Contract,
 'InternetService':InternetService,
 'PhoneService':PhoneService,
 'tenure':tenure,
 'TechSupport':TechSupport,
 'SeniorCitizen':SeniorCitizen,
 'MonthlyCharges':MonthlyCharges,
 'OnlineSecurity':OnlineSecurity,
 'PaperlessBilling':PaperlessBilling,
 'OnlineBackup':OnlineBackup,
 'MultipleLines':MultipleLines,
 'PaymentMethod':PaymentMethod,
 'TotalCharges':TotalCharges,
 'DeviceProtection':DeviceProtection
    }
    kl=pd.DataFrame([inputs])
    for x in kl.columns:
        if x in yes_col:
            kl[x]=kl[x].map(yes_map)
        elif x in list(encode.keys()): 
            kl[x]=kl[x].map(encode[x])
    kl=kl[rev]
    proba=best_model.predict_proba(kl)[0][1]
    return "Churn" if proba>=0.4 else "No Churn"

inputs=[]

for x in rev:
    if x in encode.keys():
        inputs.append(gr.Dropdown(list(encode[x].keys()),label=x))
    elif x in yes_col:
        inputs.append(gr.Dropdown(["Yes","No"],label=x))
    else:
        inputs.append(gr.Number(label=x))

app=gr.Interface(fn=predict_churn,inputs=inputs,outputs="text",title="Customer Churn Prediction")
app.launch()
