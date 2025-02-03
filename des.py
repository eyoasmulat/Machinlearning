import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Load trained models and label encoder from the 'models' directory
rf_model = joblib.load('models/model_rf.joblib')
knn_model = joblib.load('models/model_knn.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class InputData(BaseModel):
    Age: int
    Training_Hours_per_Week: int
    Previous_Injuries: int
    BMI: float
    Gender: str  # "Male" or "Female"
    Sport: str  # "Basketball", "Rugby", "Soccer", "Tennis"

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Standardize categorical features
        input_df['Gender'] = input_df['Gender'].str.lower().str.strip()
        input_df['Sport'] = input_df['Sport'].str.lower().str.strip()

        # Encode categorical features
        input_df_encoded = pd.get_dummies(input_df, columns=['Gender', 'Sport'], drop_first=True)

        # Ensure compatibility with training features
        expected_features = rf_model.feature_names_in_
        input_df_encoded = input_df_encoded.reindex(columns=expected_features, fill_value=0)

        # Predictions
        rf_prediction = rf_model.predict(input_df_encoded)[0]
        knn_prediction = knn_model.predict(input_df_encoded)[0]

        # Decode predictions
        rf_risk_label = label_encoder.inverse_transform([rf_prediction])[0]
        knn_risk_label = label_encoder.inverse_transform([knn_prediction])[0]

        return JSONResponse(content={
            "Random Forest Prediction": rf_risk_label.capitalize(),
            "KNN Prediction": knn_risk_label.capitalize()
        })
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error to the console
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
