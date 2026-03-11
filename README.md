# Telco Customer Churn Prediction System

A machine learning system that predicts customer churn for telecom companies using XGBoost with SHAP explainability, served via FastAPI.

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned train/val/test splits
├── src/
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train.py                # Model training with Optuna tuning
│   └── app.py                  # FastAPI prediction service
├── artifacts/                  # Trained model and SHAP explainer (generated)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Python Files Functionality

### 1. `preprocess.py` - Data Pipeline

**Purpose:** Transforms raw telco customer data into clean, ML-ready datasets.

**Functionality:**

| Step | Description |
|------|-------------|
| Data Loading | Reads raw CSV from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Missing Values | Fixes `TotalCharges` blanks by imputing with `tenure × MonthlyCharges` |
| Drop Columns | Removes `customerID` (not predictive) |
| Binary Encoding | Converts Yes/No columns to 0/1 (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) |
| Label Encoding | Encodes multi-category columns (`gender`, `Contract`, `PaymentMethod`, etc.) |
| Feature Engineering | Creates 6 new predictive features (see below) |
| Data Split | Stratified 70/15/15 train/val/test split |

**Engineered Features:**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `avg_monthly_spend` | `TotalCharges / tenure` | Spending pattern indicator |
| `tenure_band` | Buckets: 0-12, 12-24, 24-48, 48-72 months | Customer lifecycle stage |
| `charge_delta` | `MonthlyCharges - avg_monthly_spend` | Recent price hike indicator |
| `streaming_count` | `StreamingTV + StreamingMovies` | Entertainment usage |
| `support_count` | Sum of security, backup, protection, tech support | Support services adoption |
| `is_mtm_contract` | 1 if month-to-month contract | High churn risk flag |

**Output Files:**
- `data/processed/train.parquet` - Training set
- `data/processed/val.parquet` - Validation set
- `data/processed/test.parquet` - Test set
- `data/processed/feature_names.json` - List of feature names
- `data/processed/encoding_map.json` - Category encoding mappings

**Usage:**
```bash
cd trainning
python src/preprocess.py
```

---

### 2. `train.py` - Model Training Pipeline

**Purpose:** Trains and optimizes an XGBoost classifier with MLflow tracking and SHAP explainability.

**Functionality:**

| Phase | Description |
|-------|-------------|
| Phase 1: Baseline | Trains XGBoost with default hyperparameters |
| Phase 2: Optuna Tuning | Runs 50 trials to optimize 9 hyperparameters |
| Phase 3: SHAP Analysis | Computes feature importance and generates explanation plots |
| Logging | Tracks all experiments in MLflow |

**Hyperparameters Tuned (Optuna):**

| Parameter | Search Range |
|-----------|--------------|
| `n_estimators` | 100 - 1000 |
| `max_depth` | 3 - 10 |
| `learning_rate` | 0.01 - 0.3 (log scale) |
| `subsample` | 0.6 - 1.0 |
| `colsample_bytree` | 0.6 - 1.0 |
| `min_child_weight` | 1 - 10 |
| `gamma` | 0 - 5 |
| `reg_alpha` | 0 - 1 |
| `reg_lambda` | 0.5 - 5 |

**Metrics Computed:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC (primary optimization target, acceptance criteria: >= 0.82)

**Output Files:**
- `artifacts/model.pkl` - Trained XGBoost model
- `artifacts/explainer.pkl` - SHAP TreeExplainer
- `artifacts/features.json` - Feature list for inference
- `artifacts/best_params.json` - Optuna best hyperparameters
- `artifacts/shap_summary.png` - Global feature importance plot
- `artifacts/shap_waterfall_sample.png` - Single prediction explanation
- `artifacts/shap_feature_importance.csv` - Feature importance rankings

**Usage:**
```bash
cd trainning
python src/train.py
```

**Sample Output:**
```
Data loaded → train:4930 | val:1056 | test:1057 | features:25

🔵 Training BASELINE model...
   Val  AUC: 0.8234 | F1: 0.5891
   Test AUC: 0.8198 | F1: 0.5823

🟡 Running Optuna (50 trials)...
   Best val AUC: 0.8456 (baseline: 0.8234, delta: +0.0222)
   Val  AUC: 0.8456 | F1: 0.6234
   Test AUC: 0.8412 | F1: 0.6187

   Top 5 features by SHAP:
     tenure                         0.2341
     Contract                       0.1892
     MonthlyCharges                 0.1456
     TotalCharges                   0.0987
     is_mtm_contract                0.0876
```

---

### 3. `app.py` - FastAPI Prediction Service

**Purpose:** Serves the trained model as a REST API with real-time predictions and SHAP explanations.

**Functionality:**

| Feature | Description |
|---------|-------------|
| Model Loading | Loads model, SHAP explainer, and encodings at startup |
| Input Validation | Pydantic schemas with field constraints |
| Preprocessing | Applies same transformations as training pipeline |
| Prediction | Returns probability, label, and SHAP contributions |
| CORS Support | Allows cross-origin requests for web integration |

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict churn with SHAP explanations |
| `/health` | GET | Health check (status, model loaded, timestamp) |
| `/model-info` | GET | Model metadata (version, features, best params) |
| `/docs` | GET | Swagger UI interactive documentation |
| `/redoc` | GET | ReDoc API documentation |

**Prediction Request Schema:**

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 427.5
}
```

**Prediction Response Schema:**

```json
{
  "churn_probability": 0.7823,
  "prediction": true,
  "churn_label": "CHURN",
  "top_3_features": [
    {"feature": "Contract", "contribution": 0.234, "direction": "increases_churn"},
    {"feature": "tenure", "contribution": -0.189, "direction": "decreases_churn"},
    {"feature": "MonthlyCharges", "contribution": 0.145, "direction": "increases_churn"}
  ],
  "all_feature_contributions": {"...": "..."},
  "latency_ms": 12.34
}
```

**Usage:**
```bash
cd trainning

# Start the server
python src/app.py

# Or with uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.5,"TotalCharges":427.5}'
```

---

## Complete Workflow

```bash
# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Train model
python src/train.py

# Step 3: Start API server
python src/app.py

# Step 4: Open browser to http://localhost:8000/docs for Swagger UI
```

## Technologies Used

- **ML Framework:** XGBoost
- **Hyperparameter Tuning:** Optuna
- **Experiment Tracking:** MLflow
- **Explainability:** SHAP
- **API Framework:** FastAPI
- **Data Processing:** Pandas, NumPy, Scikit-learn

## License

MIT License
