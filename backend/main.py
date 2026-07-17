from fastapi import FastAPI

from schemas import PatientData

from model import predict_readmission



app = FastAPI(
    title="Hospital Readmission API"
)



@app.get("/")
def home():

    return {
        "message":
        "Hospital Readmission API running"
    }



@app.post("/predict")
def predict(
    patient: PatientData
):

    result = predict_readmission(
        patient.dict()
    )


    return result