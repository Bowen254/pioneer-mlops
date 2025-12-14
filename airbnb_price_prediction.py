#!/usr/bin/python
# -*- coding: utf-8 -*-

from metaflow import FlowSpec, step, card, current
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import bentoml
import subprocess


class NYCAirbnbFlow(FlowSpec):

    # =============================================================
    # 1. BUSINESS PROBLEM UNDERSTANDING + DATA LOADING
    # =============================================================
    @card
    @step
    def start(self):
        print("Business Problem: Predict Airbnb Price in NYC")
        print("Loading dataset...")

        self.data = pd.read_csv("AB_NYC_2019.csv")
        print(f"Dataset shape: {self.data.shape}")

        # ASCII-only card
        current.card.append(f"""
# Business Problem & Assignment Overview

MLOPS - Take Home Assignment
1. Title: Predict Airbnb Nightly Prices in New York City with Metaflow
2. Data Source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
3. Deliverables:
    1. nyc_airbnb_flow.py - complete Metaflow flow
    2. Jupyter notebook showing run inspection and cards
    3. drift_report.html plus FastAPI inference endpoint
    4. README with screenshots and explanations of all stages

Business Requirements:
- Predict nightly prices with 20 percent deviation from original

Dataset Info:
- Shape: {self.data.shape}
- Columns: {list(self.data.columns)}
- Sample rows:
{self.data.head(5).to_html()}
""")

        self.next(self.data_engineering)

    # =============================================================
    # 2. DATA ENGINEERING PIPELINE (CLEANING + FEATURE EXTRACTION)
    # =============================================================
    @card
    @step
    def data_engineering(self):

        df = self.data.copy()

        # Drop irrelevant columns first
        df = df.drop(columns=['id', 'name', 'host_id', 'host_name'], errors='ignore')

        # Handle missing values
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
        df["last_review"] = pd.to_datetime(df["last_review"])
        df["last_review"] = df["last_review"].fillna(df["last_review"].min())

        # Date Features
        df["last_review_year"] = df["last_review"].dt.year
        df["last_review_month"] = df["last_review"].dt.month
        df["last_review_day"] = df["last_review"].dt.day
        df["days_since_last_review"] = (pd.Timestamp.today() - df["last_review"]).dt.days
        df = df.drop(columns=["last_review"])

        # Label Encoding
        self.categorical_cols = ["neighbourhood_group", "neighbourhood", "room_type"]
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Numerical Scaling
        self.num_cols = [
            "latitude", "longitude", "minimum_nights", "number_of_reviews",
            "reviews_per_month", "calculated_host_listings_count",
            "availability_365", "days_since_last_review"
        ]
        self.scaler = StandardScaler()
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])

        self.data = df

        # Save processed features
        os.makedirs("model_artifacts", exist_ok=True)
        feature_path = "model_artifacts/processed_features.csv"
        df.to_csv(feature_path, index=False)
        print(f"Processed features saved at {feature_path}")

        # Track with DVC
        try:
            subprocess.run(["dvc", "add", feature_path], check=True)
            print("Processed features tracked with DVC")
        except subprocess.CalledProcessError as e:
            print("Failed to track features with DVC:", e)

        # Card info
        missing_vals = df.isna().sum().to_frame("missing_count").to_html()
        summary_stats = df.describe().to_html()

        current.card.append(f"""
# Data Engineering Summary

Dataset Shape: {df.shape}

Categorical Columns: {self.categorical_cols}

Numerical Columns: {self.num_cols}

Missing Values Per Column:
{missing_vals}

Summary Statistics:
{summary_stats}
""")

        self.next(self.eda_experimentation)

    # =============================================================
    # 3. EDA & EXPERIMENTATION
    # =============================================================
    @card
    @step
    def eda_experimentation(self):

        summary_html = self.data.describe().to_html()
        corr_series = self.data.corr(numeric_only=True)["price"].sort_values(ascending=False)
        corr_html = corr_series.to_frame("correlation_with_price").to_html()

        current.card.append(f"""
# Exploratory Data Analysis

Dataset Summary:
{summary_html}

Correlation With Price:
{corr_html}
""")

        print("EDA completed successfully.")
        self.next(self.model_development)

    # =============================================================
    # 4. MODEL DEVELOPMENT & TRAINING
    # =============================================================
    @card
    @step
    def model_development(self):

        # Features and target
        self.X = self.data.drop(columns=["price"])
        self.y = self.data["price"]

        # Store feature names
        self.feature_names = list(self.X.columns)
        print("Training features (order locked):")
        print(self.feature_names)

        # Split: Train + Temp, then Test + Validation
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        self.X_train, self.X_test, self.X_val = X_train, X_test, X_val
        self.y_train, self.y_test, self.y_val = y_train, y_test, y_val

        # Save validation set locally
        os.makedirs("model_artifacts", exist_ok=True)
        val_df = X_val.copy()
        val_df["price"] = y_val
        val_df.to_csv("model_artifacts/validation_set.csv", index=False)
        print("Validation set saved at model_artifacts/validation_set.csv")

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        print("Training model...")
        self.model.fit(self.X_train, self.y_train)

        current.card.append(f"""
# Model Training

Training samples: {len(self.X_train)}
Testing samples: {len(self.X_test)}
Validation samples: {len(self.X_val)}
Model type: RandomForestRegressor
Number of estimators: 200
""")

        self.next(self.model_validation)

    # =============================================================
    # 5. MODEL VALIDATION
    # =============================================================
    @card
    @step
    def model_validation(self):

        y_pred = self.model.predict(self.X_test)

        self.mae = mean_absolute_error(self.y_test, y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        self.r2 = r2_score(self.y_test, y_pred)

        current.card.append(f"""
# Model Validation Metrics

MAE: {self.mae:.4f}
RMSE: {self.rmse:.4f}
R2 Score: {self.r2:.4f}
""")

        print("Model validation completed.")
        self.next(self.monitoring_drift)

    # =============================================================
    # 6. MONITORING & DRIFT DETECTION
    # =============================================================
    @card
    @step
    def monitoring_drift(self):

        report = Report(metrics=[DataDriftPreset()])

        reference_df = self.X_train.reindex(columns=self.feature_names)
        reference_df["price"] = self.y_train.values

        current_df = self.X_test.reindex(columns=self.feature_names)
        current_df["price"] = self.y_test.values

        self.drift_result = report.run(
            reference_data=reference_df,
            current_data=current_df
        )

        os.makedirs("reports", exist_ok=True)
        self.drift_report_path = "reports/drift_report.html"
        self.drift_result.save_html(self.drift_report_path)

        current.card.append(f"""
# Data Drift Report

Drift report saved at: {self.drift_report_path}
""")

        print("Drift report generated.")
        self.next(self.model_packaging)

    # =============================================================
    # 7. MODEL PACKAGING
    # =============================================================
    @card
    @step
    def model_packaging(self):

        print("Saving model using BentoML...")

        bentoml_model = bentoml.sklearn.save_model(
            name="nyc_airbnb_price_model",
            model=self.model,
            signatures={
                "predict": {"batchable": True, "batch_dim": 0}
            },
            metadata={
                "framework": "sklearn",
                "problem_type": "regression",
                "dataset": "NYC Airbnb 2019"
            },
            custom_objects={
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "categorical_cols": self.categorical_cols,
                "numerical_cols": self.num_cols,
                "feature_names": self.feature_names
            }
        )

        # Save locally for serving
        os.makedirs("model_artifacts", exist_ok=True)
        joblib.dump(self.feature_names, "model_artifacts/feature_names.pkl")

        current.card.append(f"""
# Model Packaging

Model saved with BentoML tag: {bentoml_model.tag}
Feature names saved at: model_artifacts/feature_names.pkl
""")

        self.bentoml_model_tag = str(bentoml_model.tag)
        self.next(self.retraining_trigger)

    # =============================================================
    # 8. CONTINUOUS RETRAINING TRIGGER
    # =============================================================
    @card
    @step
    def retraining_trigger(self):
        print("Retraining trigger step (placeholder)")
        current.card.append("Retraining trigger step executed")
        self.next(self.end)

    # =============================================================
    # 9. END
    # =============================================================
    @card
    @step
    def end(self):
        print("Flow execution completed successfully!")
        current.card.append("Flow execution completed successfully!")

if __name__ == "__main__":
    NYCAirbnbFlow()
