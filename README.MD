NYC Airbnb Price Prediction
Overview

This project implements an end-to-end ML workflow to predict Airbnb nightly prices in New York City using Metaflow. The workflow includes:

Data engineering and preprocessing

Exploratory data analysis (EDA)

Model training, validation, and drift monitoring

Model packaging and serving via BentoML + FastAPI

Simulation testing using a validation set

All workflow steps include Metaflow inspections and cards for transparency.

Dataset

Source: NYC Airbnb Open Data

Year: 2019

Features include:

neighbourhood_group, neighbourhood, room_type

latitude, longitude

minimum_nights, number_of_reviews, reviews_per_month

calculated_host_listings_count, availability_365

last_review, price

Requirements
pip install -r requirements.txt


Key packages:

metaflow

scikit-learn

pandas, numpy

bentoml

fastapi, uvicorn

evidently

Workflow

The ML workflow consists of the following steps:

Business Problem & Data Loading – Load dataset and define prediction goal

Data Engineering – Clean missing values, encode categorical variables, scale numeric features, create date-based features

EDA & Experimentation – Generate dataset summaries and correlation analyses, logged to cards

Model Development – Train Random Forest Regressor with train-test-validation split

Model Validation – Evaluate using MAE, RMSE, and R2 score

Monitoring & Drift Detection – Detect feature drift with Evidently and generate HTML reports

Model Packaging – Save model with BentoML including preprocessing artifacts

Continuous Retraining Trigger – Placeholder for future retraining logic

Flow Completion – Confirms successful workflow execution

Running the Workflow
python nyc_airbnb_flow.py run


Inspect workflow and cards:

python nyc_airbnb_flow.py inspect
python nyc_airbnb_flow.py card <run-id>

Serving the Model

Start FastAPI service:

uvicorn serve_airbnb_model:app --reload

Prediction Endpoint
POST http://127.0.0.1:8000/predict


Payload Example:

{
  "neighbourhood_group": "Brooklyn",
  "neighbourhood": "Williamsburg",
  "room_type": "Entire home/apt",
  "latitude": 40.708,
  "longitude": -73.957,
  "minimum_nights": 3,
  "number_of_reviews": 120,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "last_review": "2019-06-15"
}


Response Example:

{
  "predicted_price": 250.75
}

Validation Simulation

Uses model_artifacts/validation_set.csv

Calls FastAPI endpoint for each validation sample

Computes deviations: mean, max, min, median (absolute and percentage)

Run the simulation:

python simulate_validation.py

Drift Monitoring

Generated HTML drift report is saved in:

reports/drift_report.html

Deliverables

nyc_airbnb_flow.py – Metaflow workflow

serve_airbnb_model.py – FastAPI inference endpoint

simulate_validation.py – Validation simulation script

model_artifacts/validation_set.csv – Saved validation data

reports/drift_report.html – Drift monitoring report

notebooks/ – Jupyter notebooks with workflow inspections and cards

Business Objective

Predict Airbnb nightly prices within 20% deviation from actual prices while ensuring reproducibility, monitoring, and MLOps best practices.