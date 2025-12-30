import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

features=joblib.load("feature_names.pkl")
encoding=joblib.load("feature_encoding.pkl")
best_model=joblib.load("best_churn_model.pkl")

threshold=0.40

def preprocess_input(raw:dict):
    df=pd.DataFrame([raw])

    for col, mapping in encoding.items():
        if col in df.columns:
            df[col]=df[col].map(mapping)
    
    numeric_cols=["MonthlyCharges", "TotalCharges", "tenure"]
    for col in numeric_cols:
        df[col]=pd.to_numeric(df[col],errors='raise')

    df=df[features]
    return df

class ChurnRequest(BaseModel):
    MonthlyCharges: float
    TotalCharges: float
    tenure: int
    Contract: str
    gender: str
    PaperlessBilling: str
    OnlineSecurity: str
    OnlineBackup: str
    TechSupport: str
    Partner: str
    Dependents: str
    SeniorCitizen: str
    PaymentMethod: str

app=FastAPI(title="Telco Churn API")


@app.post("/predict")
def predict_api(request:ChurnRequest):
    raw=request.model_dump()
    df=preprocess_input(raw)
    proba=best_model.predict_proba(df)[0][1]
    pred=int(proba>=threshold)

    return {"prediction": "Churn" if pred else "No Churn","probability":float(proba)}

def gradio_predict(*args):
    # Convert positional inputs into {column_name: value}
    raw =dict(zip(features,args))
    df = preprocess_input(raw)
    proba = best_model.predict_proba(df)[0][1]
    pred = "Churn" if proba >= threshold else "No Churn"

    return f"{pred} (Probability: {proba:.3f})"




inputs=[]

for col in features:
    if col in encoding.keys():
        inputs.append(gr.Dropdown(list(encoding[col].keys()),label=col))
    else:
        inputs.append(gr.Number(label=col))

demo=gr.Interface(inputs=inputs,outputs="text",fn=gradio_predict,title="Telco Customer Churn Prediction",description="Enter customer details to predict churn")

app=gr.mount_gradio_app(app,demo,path="/")


