import pandas as pd
import requests
import json
import numpy as np

# -------------------------------------------------
# FastAPI endpoint
# -------------------------------------------------
URL = "http://127.0.0.1:8000/predict"

# -------------------------------------------------
# Load validation CSV
# -------------------------------------------------
val_df = pd.read_csv("model_artifacts/validation_set.csv")
val_df=val_df.iloc[0:100,:]

# Fill missing columns if needed
required_columns = [
    "neighbourhood_group", "neighbourhood", "room_type",
    "latitude", "longitude", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "last_review", "price"
]

for col in required_columns:
    if col not in val_df.columns:
        val_df[col] = np.nan

# -------------------------------------------------
# Loop through each row and call endpoint
# -------------------------------------------------
deviations = []

for idx, row in val_df.iterrows():
    payload = {
        "neighbourhood_group": str(row["neighbourhood_group"]) if not pd.isna(row["neighbourhood_group"]) else "Brooklyn",
        "neighbourhood": str(row["neighbourhood"]) if not pd.isna(row["neighbourhood"]) else "Williamsburg",
        "room_type": str(row["room_type"]) if not pd.isna(row["room_type"]) else "Entire home/apt",
        "latitude": float(row["latitude"]) if not pd.isna(row["latitude"]) else 0.0,
        "longitude": float(row["longitude"]) if not pd.isna(row["longitude"]) else 0.0,
        "minimum_nights": int(row["minimum_nights"]) if not pd.isna(row["minimum_nights"]) else 1,
        "number_of_reviews": int(row["number_of_reviews"]) if not pd.isna(row["number_of_reviews"]) else 0,
        "reviews_per_month": float(row["reviews_per_month"]) if not pd.isna(row["reviews_per_month"]) else None,
        "calculated_host_listings_count": int(row["calculated_host_listings_count"]) if not pd.isna(row["calculated_host_listings_count"]) else 1,
        "availability_365": int(row["availability_365"]) if not pd.isna(row["availability_365"]) else 0,
        "last_review": str(row["last_review"]) if not pd.isna(row["last_review"]) else None
    }

    response = requests.post(URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

    if response.status_code == 200:
        pred_price = response.json()["predicted_price"]
        actual_price = row["price"] if not pd.isna(row["price"]) else np.nan
        deviation = pred_price - actual_price if not pd.isna(actual_price) else np.nan
        deviations.append(deviation)
        print(f"Sample {idx}: Actual={actual_price}, Predicted={pred_price:.2f}, Deviation={deviation}")
    else:
        print(f"Sample {idx}: Request failed with status {response.status_code}")

# -------------------------------------------------
# Overall deviation statistics
# -------------------------------------------------
deviations = np.array([d for d in deviations if not np.isnan(d)])
if len(deviations) > 0:
    mean_dev = np.mean(deviations)
    max_dev = np.max(deviations)
    min_dev = np.min(deviations)
    median_dev = np.median(deviations)
    
    # Percentage deviation
    pct_devs = deviations / val_df['price'].values.tolist() * 100
    mean_pct = np.mean(pct_devs)
    max_pct = np.max(pct_devs)
    min_pct = np.min(pct_devs)
    median_pct = np.median(pct_devs)
    
    print("\n=== Overall Deviation Statistics ===")
    print(f"Mean Deviation        : {mean_dev:.2f} (${mean_pct:.2f} %)")
    print(f"Max Deviation         : {max_dev:.2f} (${max_pct:.2f} %)")
    print(f"Min Deviation         : {min_dev:.2f} (${min_pct:.2f} %)")
    print(f"Median Deviation      : {median_dev:.2f} (${median_pct:.2f} %)")
else:
    print("No valid deviations to compute statistics.")