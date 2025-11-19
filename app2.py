import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

yes_col=joblib.load("yes_col_names.pkl")
encode=joblib.load("label_encoding.pkl")
yes_map={"Yes":1,"No":0}
best_model=joblib.load("best_churn_model.pkl")
rev=joblib.load("feature_names.pkl")

binary_col=yes_col+["SeniorCitizen"]

def preprocess_input(raw):
    df=pd.DataFrame([raw])

    for col in binary_col:
        df[col]=df[col].map(yes_map)

    
    for col,mapping in encode.items():
        if col in df:
            df[col]=df[col].map(mapping)
            df[col]=df[col].fillna(-1)

    for col in df:
        df[col]=pd.to_numeric(df[col],errors="coerce")

    df=df.reindex(columns=rev)
    return df

app=FastAPI(title="Telco Churn API")


class ChurnRequest(BaseModel):
    Contract: str
    InternetService: str
    PhoneService: str
    tenure: float
    TechSupport: str
    SeniorCitizen: str
    MonthlyCharges: float
    OnlineSecurity: str
    PaperlessBilling: str
    OnlineBackup: str
    MultipleLines: str
    PaymentMethod: str
    TotalCharges: float
    DeviceProtection: str

@app.post("/predict")
def predict_api(request:ChurnRequest):
    raw=request.model_dump()
    df=preprocess_input(raw)
    proba=best_model.predict_proba(df)[0][1]
    pred=int(proba>=0.4)

    return {"prediction": "Churn" if pred ==1 else "No Churn","probability":float(proba)}

def gradio_predict(*args):
    # Convert positional inputs into {column_name: value}
    raw = {col: val for col, val in zip(rev, args)}

    df = preprocess_input(raw)
    proba = best_model.predict_proba(df)[0][1]
    pred = "Churn" if proba >= 0.40 else "No Churn"

    return f"{pred} (Probability: {proba:.3f})"




inputs=[]

for col in rev:
    if col in encode:
        inputs.append(gr.Dropdown(list(encode[col].keys()),label=col))
    elif col in binary_col:
        inputs.append(gr.Dropdown(["Yes","No"],label=col))
    else:
        inputs.append(gr.Number(label=col))

demo=gr.Interface(inputs=inputs,outputs="text",fn=gradio_predict,title="Telco Customer Churn Prediction",description="Enter customer details to predict churn")

app=gr.mount_gradio_app(app,demo,path="/")


