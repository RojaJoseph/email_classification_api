from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from utils import mask_pii

app = FastAPI()

# Define input format
class EmailInput(BaseModel):
    message: str
    model_type: str = "svm"  # default model

@app.post("/classify/")
def classify_email(input_data: EmailInput):
    try:
        masked_message = mask_pii(input_data.message)
        model_path = f"email_classifier_{input_data.model_type}.pkl"

        # Load model
        model = joblib.load(model_path)

        # Predict category
        prediction = model.predict([masked_message])[0]
        return {
            "masked_email": masked_message,
            "predicted_category": prediction
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{input_data.model_type}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
