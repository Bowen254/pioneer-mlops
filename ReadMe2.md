# NYC Airbnb Price Prediction
## Overview

This project demonstrates an end-to-end MLOps workflow for predicting Airbnb nightly prices in New York City using Metaflow.

The pipeline covers the full machine learning lifecycle—from data ingestion and exploration to model deployment and monitoring—while emphasizing reproducibility, transparency, and best practices.

### What’s included?

- Data engineering & preprocessing

- Exploratory data analysis (EDA)

- Model training, validation & evaluation

- Drift detection & monitoring

- Model packaging & API serving with BentoML + FastAPI

- Validation simulation using a held-out dataset

Each step of the workflow is fully traceable using Metaflow inspections and cards.

### Dataset

Source: NYC Airbnb Open Data

Year: 2019

### Key Features

Location: `neighbourhood_group, neighbourhood, latitude, longitude `

Listing details: ` room_type, minimum_nights, availability_365 `

Reviews: `number_of_reviews, reviews_per_month, last_review `

Host info: `calculated_host_listings_count`

Target variable: `price`

### Requirements

Install all dependencies using:

```pip install -r requirements.txt```

### Key Libraries

- metaflow

- scikit-learn

- pandas, numpy

- bentoml

- fastapi, uvicorn

- evidently

## ML Workflow

The workflow consists of the following steps:

- Business Problem & Data Loading
- Load the dataset and define the prediction objective.

- Data Engineering

- Handle missing values

- Encode categorical features

- Scale numerical variables

- Create date-based features

- EDA & Experimentation
- Generate dataset summaries and correlation analyses, logged via Metaflow cards.

#### Model Development
Train a Random Forest Regressor using train, validation, and test splits.

#### Model Validation
- Evaluate performance using:

- MAE

- RMSE

- R² score

#### Monitoring & Drift Detection
Detect feature drift using Evidently and generate HTML reports.

#### Model Packaging
Package the trained model and preprocessing artifacts using BentoML.

#### Continuous Retraining Trigger
Placeholder for future retraining automation.

#### Flow Completion
Confirms successful execution of the entire pipeline.

### Running the Workflow

Run the Metaflow pipeline:

``` python
python nyc_airbnb_flow.py run

# Inspect Runs & Cards
python nyc_airbnb_flow.py inspect
python nyc_airbnb_flow.py card <run-id>

```

### Serving the Model

- Start the FastAPI service:

``` python
uvicorn serve_airbnb_model:app --reload

```

Prediction Endpoint
POST `http://127.0.0.1:8000/predict`

Example Request Payload
``` json
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
```

Example Response
``` json
{
  "predicted_price": 250.75
}
```

### Validation Simulation

The validation simulation:

- Uses model_artifacts/validation_set.csv

- Sends each sample to the FastAPI endpoint

- Computes deviation metrics:

  - Mean

  - Max / Min

  - Median

  - Absolute and percentage differences

Run the simulation:

``` python
python simulate_validation.py

```

### Drift Monitoring

Feature drift reports are automatically generated and saved to:

`reports/drift_report.html`

## Deliverables

- nyc_airbnb_flow.py – Metaflow workflow definition

- serve_airbnb_model.py – FastAPI inference service

- simulate_validation.py – Validation simulation script

- model_artifacts/validation_set.csv – Validation dataset

- reports/drift_report.html – Drift monitoring report

- notebooks/ – Jupyter notebooks with workflow inspections and cards

## Business Objective

Predict Airbnb nightly prices within a 20% deviation from actual values while ensuring:

- Reproducibility

- Observability

- Robust monitoring

- Production-ready MLOps practices