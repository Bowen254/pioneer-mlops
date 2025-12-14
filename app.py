from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
MODEL_PATH = "model_artifacts/model.pkl"
SCALER_PATH = "model_artifacts/scaler.pkl"
ENCODER_PATH = "model_artifacts/encoders.pkl"
FEATURES_PATH = "model_artifacts/feature_names.pkl"

feature_names = joblib.load(FEATURES_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODER_PATH)

categorical_cols = ["neighbourhood_group", "neighbourhood", "room_type"]
numerical_cols = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365", "days_since_last_review"
]

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="NYC Airbnb Price Prediction API")

# -------------------------------------------------
# Input schema
# -------------------------------------------------
class AirbnbInput(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    room_type: str
    latitude: float
    longitude: float
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float | None = None
    calculated_host_listings_count: int
    availability_365: int
    last_review: str | None = None  # YYYY-MM-DD


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: AirbnbInput):

    df = pd.DataFrame([data.dict()])
    """
    # ----------------------------------
    # Handle missing values (EXACT MATCH)
    # ----------------------------------
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    df["last_review"] = pd.to_datetime(df["last_review"])
    min_last_review=df["last_review"].min()
    df["last_review"] = df["last_review"].fillna(min_last_review)

    # ----------------------------------
    # Date feature engineering
    # ----------------------------------
    df["last_review_year"] = df["last_review"].dt.year
    df["last_review_month"] = df["last_review"].dt.month
    df["last_review_day"] = df["last_review"].dt.day

    df["days_since_last_review"] = (
        pd.Timestamp.today() - df["last_review"]
    ).dt.days

    df.drop(columns=["last_review"], inplace=True)

    # ----------------------------------
    # Encode categorical features
    # ----------------------------------
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    # ----------------------------------
    # Scale numerical features (ONLY num_cols)
    # ----------------------------------
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # -------------------------------------------------
    # Enforce feature order EXACTLY as training
    # -------------------------------------------------
    """
    df = df.reindex(columns=feature_names)
    
    prediction = model.predict(df)

    return {
        "predicted_price": float(prediction[0])
    }
