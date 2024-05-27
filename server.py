import uvicorn
import joblib
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Suppress the specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names")

app = FastAPI()

origins = ["https://dyslexia-backend.onrender.com"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def read_root():
    return {"data": "Welcome to the Dyslexia Predictor API"}

class PredictionRequest(BaseModel):
    data: list = Field(..., description="List of features for prediction")

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if len(request.data) != 26:  # Check if the number of features is correct
            raise HTTPException(status_code=400, detail="Input data must have 26 features.")
        
        data = np.asarray(request.data).astype('float32')
        print(f"Received data: {data}")
        prediction = model.predict([data])[0]
        print(f"Prediction: {prediction}")

        prediction = int(prediction)
        message = "Less likely to have dyslexia" if prediction == 0 else "More likely to have dyslexia"
        print(f"Prediction: {prediction}, Message: {message}")

        return {"prediction": prediction, "message": message}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
