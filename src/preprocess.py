"""
preprocess.py — Telco Churn Data Pipeline
Cleans raw CSV and outputs train/val/test parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = Path("data/processed")


def run(raw_path: Path = RAW_PATH, processed_dir: Path = PROCESSED_DIR) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("📥 Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"   Shape: {df.shape}")

    # --- 1. Fix TotalCharges (blanks → NaN → numeric) ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    null_before = df["TotalCharges"].isna().sum()
    # Impute with tenure * MonthlyCharges (sensible for new customers)
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    print(f"   TotalCharges: fixed {null_before} nulls")

    # --- 2. Drop customerID (not a feature) ---
    df = df.drop(columns=["customerID"])

    # --- 3. Binary encode Yes/No columns ---
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # --- 4. Encode multi-category columns ---
    multi_cat_cols = [
        "gender", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    le = LabelEncoder()
    encoding_map = {}
    for col in multi_cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        encoding_map[col] = {int(i): c for i, c in enumerate(le.classes_)}

    # Save encoding map for inference
    with open(processed_dir / "encoding_map.json", "w") as f:
        json.dump(encoding_map, f, indent=2)

    # --- 5. Feature Engineering (5+ new features) ---
    # F1: Avg monthly spend proxy
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # F2: Tenure band (bucketed lifecycle stage)
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3], right=True, include_lowest=True
    ).astype(int)

    # F3: Charge delta (monthly vs average — positive = recent price hike)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # F4: Streaming services count
    df["streaming_count"] = df["StreamingTV"] + df["StreamingMovies"]

    # F5: Support services count
    df["support_count"] = df["OnlineSecurity"] + df["TechSupport"] + df["OnlineBackup"] + df["DeviceProtection"]

    # F6: Is month-to-month contract (already encoded, just alias for clarity)
    df["is_mtm_contract"] = (df["Contract"] == 0).astype(int)  # 0 = month-to-month after label encode

    print(f"   Engineered 6 new features → final shape: {df.shape}")

    # --- 6. Split ---
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Save feature names
    feature_names = list(X_train.columns)
    with open(processed_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # --- 7. Save parquet ---
    for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        out = Xs.copy()
        out[target] = ys.values
        out.to_parquet(processed_dir / f"{split_name}.parquet", index=False)
        print(f"   ✅ {split_name}.parquet → {out.shape} | churn rate: {ys.mean():.3f}")

    print("\n✅ Preprocessing complete.")
    return {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "features": feature_names,
        "n_features": len(feature_names),
    }


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    info = run()
    print("\nFeatures:", info["features"])