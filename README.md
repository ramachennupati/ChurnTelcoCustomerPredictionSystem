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

## Dataset

### Source Information

| Attribute | Details |
|-----------|---------|
| **Name** | Telco Customer Churn |
| **Provider** | IBM Sample Data Sets |
| **Platform** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Records** | 7,043 customers |
| **Features** | 21 columns |

### How to Download

**Option 1: Kaggle Website**
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click "Download" (requires free Kaggle account)
3. Extract `archive.zip` to get the CSV
4. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`

**Option 2: Kaggle CLI**
```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/
```

**Option 3: Direct from IBM**
```
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
```

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | string | Unique customer identifier |
| `gender` | string | Male / Female |
| `SeniorCitizen` | int | 1 if senior citizen, 0 otherwise |
| `Partner` | string | Yes / No |
| `Dependents` | string | Yes / No |
| `tenure` | int | Months with company (0-72) |
| `PhoneService` | string | Yes / No |
| `MultipleLines` | string | Yes / No / No phone service |
| `InternetService` | string | DSL / Fiber optic / No |
| `OnlineSecurity` | string | Yes / No / No internet service |
| `OnlineBackup` | string | Yes / No / No internet service |
| `DeviceProtection` | string | Yes / No / No internet service |
| `TechSupport` | string | Yes / No / No internet service |
| `StreamingTV` | string | Yes / No / No internet service |
| `StreamingMovies` | string | Yes / No / No internet service |
| `Contract` | string | Month-to-month / One year / Two year |
| `PaperlessBilling` | string | Yes / No |
| `PaymentMethod` | string | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | float | Monthly charge amount |
| `TotalCharges` | string | Total charges to date (has blanks for new customers) |
| `Churn` | string | Yes / No (target variable) |

### How the Data Was Originally Collected

This is a **synthetic dataset** created by IBM for educational purposes. It simulates realistic telecom customer data including:

1. **Customer Demographics** - Age, gender, partner/dependent status
2. **Account Information** - Tenure, contract type, payment method, billing preferences
3. **Services Subscribed** - Phone, internet, streaming, security, tech support
4. **Billing Details** - Monthly charges and total charges to date
5. **Churn Label** - Whether customer left the company (target variable)

### Key Churn Patterns in the Data

The dataset reflects real-world telecom churn patterns:

| Factor | Churn Impact |
|--------|--------------|
| Month-to-month contracts | Higher churn risk |
| Fiber optic internet | Higher churn (often price-sensitive) |
| Longer tenure | Lower churn (loyalty) |
| Electronic check payment | Higher churn |
| Senior citizens | Higher churn |
| No tech support/security | Higher churn |

---

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

---

## Beginner's Guide to Model Training

This section explains how the model training works in simple terms for beginners.

### What is Machine Learning?

Machine Learning (ML) is teaching computers to learn patterns from data and make predictions. Instead of writing explicit rules, we show the computer examples and let it figure out the patterns.

**Analogy:** Imagine teaching a child to recognize cats. Instead of describing every feature of a cat, you show them many pictures of cats and non-cats. Eventually, they learn to recognize cats on their own. That's machine learning!

### What is Churn Prediction?

**Churn** = When a customer leaves/cancels their service

**Churn Prediction** = Predicting which customers are likely to leave before they actually do

**Why it matters:**
- Acquiring new customers costs 5-7x more than retaining existing ones
- If we can predict who might leave, we can offer them incentives to stay
- This saves the company money and improves customer satisfaction

### Understanding the Training Process

#### Step 1: Data Preparation (What `preprocess.py` does)

Before training, we need to prepare the data:

```
Raw Data (CSV) → Clean Data → Split into Train/Val/Test
```

**Why split the data?**

| Dataset | Size | Purpose |
|---------|------|---------|
| Training Set | 70% | Model learns patterns from this data |
| Validation Set | 15% | Used to tune hyperparameters and prevent overfitting |
| Test Set | 15% | Final evaluation - model has never seen this data |

**Think of it like studying for an exam:**
- Training Set = Textbook (you learn from it)
- Validation Set = Practice tests (you adjust your study strategy)
- Test Set = Final exam (true measure of your knowledge)

#### Step 2: What is XGBoost?

**XGBoost** (eXtreme Gradient Boosting) is our chosen algorithm. It's one of the most powerful and popular ML algorithms.

**How it works (simplified):**

1. Start with a simple prediction (e.g., "50% of customers churn")
2. Build a small decision tree to fix the biggest mistakes
3. Build another tree to fix remaining mistakes
4. Repeat 100-1000 times
5. Combine all trees for the final prediction

```
Tree 1: "Month-to-month contract? → Higher churn"
   +
Tree 2: "Low tenure? → Higher churn"
   +
Tree 3: "No tech support? → Higher churn"
   +
... (hundreds more trees)
   =
Final Prediction: 78% likely to churn
```

**Why XGBoost?**
- Handles missing data automatically
- Works well with both numbers and categories
- Fast and efficient
- Consistently wins ML competitions

#### Step 3: What are Hyperparameters?

**Parameters** = Values the model learns from data (internal)
**Hyperparameters** = Settings WE choose before training (external)

**Analogy:** If the model is a car:
- Parameters = How the engine adjusts while driving (automatic)
- Hyperparameters = Settings you choose: sport mode, eco mode, seat position (manual)

**Key XGBoost Hyperparameters Explained:**

| Hyperparameter | What it Controls | Simple Explanation |
|----------------|------------------|-------------------|
| `n_estimators` | Number of trees | More trees = more learning, but slower |
| `max_depth` | Tree depth | Deeper trees = more complex patterns, risk of overfitting |
| `learning_rate` | Step size | Smaller = learns slowly but carefully |
| `subsample` | Data sampling | Uses random subset of data for each tree |
| `colsample_bytree` | Feature sampling | Uses random subset of features for each tree |
| `min_child_weight` | Minimum samples | Prevents trees from learning noise |
| `gamma` | Pruning threshold | Removes unnecessary tree branches |
| `reg_alpha` | L1 regularization | Penalizes complexity (forces simpler model) |
| `reg_lambda` | L2 regularization | Penalizes large weights (forces balanced model) |

#### Step 4: What is Optuna? (Hyperparameter Tuning)

Finding the best hyperparameters manually is tedious. **Optuna** automates this search.

**How Optuna works:**

```
Trial 1: Try n_estimators=100, max_depth=3, learning_rate=0.1 → AUC=0.78
Trial 2: Try n_estimators=500, max_depth=5, learning_rate=0.05 → AUC=0.82
Trial 3: Try n_estimators=300, max_depth=7, learning_rate=0.03 → AUC=0.84
... (50 trials)
Best: Trial 47 with AUC=0.85
```

**Optuna is smart:**
- It doesn't try random combinations
- It learns from previous trials
- It focuses on promising areas of the search space

#### Step 5: Understanding Model Metrics

After training, we measure how good the model is:

| Metric | What it Measures | Formula | Good Value |
|--------|------------------|---------|------------|
| **Accuracy** | Overall correctness | (Correct predictions) / (Total predictions) | > 0.80 |
| **Precision** | When we predict churn, how often are we right? | (True Churns) / (Predicted Churns) | > 0.65 |
| **Recall** | Of all actual churns, how many did we catch? | (True Churns) / (Actual Churns) | > 0.60 |
| **F1 Score** | Balance of precision and recall | 2 × (Precision × Recall) / (Precision + Recall) | > 0.60 |
| **AUC-ROC** | Overall ranking ability | Area under ROC curve | > 0.82 |

**Confusion Matrix Explained:**

```
                    Predicted
                 STAY    CHURN
Actual  STAY    [4500]    [300]   ← True Negatives / False Positives
        CHURN    [200]    [800]   ← False Negatives / True Positives
```

- **True Positive (800):** Predicted churn, actually churned ✓
- **True Negative (4500):** Predicted stay, actually stayed ✓
- **False Positive (300):** Predicted churn, but stayed ✗ (wasted retention offer)
- **False Negative (200):** Predicted stay, but churned ✗ (lost customer!)

#### Step 6: What is SHAP? (Model Explainability)

SHAP (SHapley Additive exPlanations) tells us WHY the model made a prediction.

**The Problem:** ML models are often "black boxes" - they give predictions but don't explain why.

**SHAP Solution:** For each prediction, SHAP shows how much each feature contributed.

**Example Prediction:**

```
Customer: John, tenure=2 months, Contract=Month-to-month, MonthlyCharges=$95

Model Prediction: 82% likely to churn

SHAP Explanation:
  Base rate (average):                    27% churn probability
  + Contract (Month-to-month):           +25% (increases churn risk)
  + tenure (2 months):                   +18% (new customer, higher risk)
  + MonthlyCharges ($95):                +12% (high bill, higher risk)
  + No TechSupport:                       +5% (increases churn risk)
  - Partner (Yes):                        -3% (decreases churn risk)
  - Other features:                       -2%
  ─────────────────────────────────────────
  Final prediction:                       82% churn probability
```

**SHAP Values:**
- **Positive value** → Feature increases churn probability
- **Negative value** → Feature decreases churn probability
- **Larger absolute value** → Feature has more impact

#### Step 7: What is MLflow? (Experiment Tracking)

When training models, you try many different configurations. **MLflow** keeps track of everything.

**What MLflow logs:**
- Hyperparameters used
- Metrics achieved
- Model artifacts (saved model files)
- Plots and visualizations

**Why it's useful:**
- Compare different experiments
- Reproduce results
- Share findings with team
- Deploy best model to production

### Training Pipeline Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw CSV    │───▶│  Preprocess  │───▶│ Train/Val/   │
│   (7,043     │    │  (clean,     │    │ Test Split   │
│   records)   │    │  encode,     │    │ (70/15/15)   │
│              │    │  engineer)   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE 1: BASELINE                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│  │  Default   │───▶│   Train    │───▶│  Evaluate  │──▶ AUC=0.82 │
│  │  Params    │    │  XGBoost   │    │  on Val    │              │
│  └────────────┘    └────────────┘    └────────────┘              │
└──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE 2: OPTUNA TUNING                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│  │  Trial 1   │───▶│   Train    │───▶│  Evaluate  │──▶ AUC=0.78 │
│  │  Trial 2   │───▶│   Train    │───▶│  Evaluate  │──▶ AUC=0.83 │
│  │  ...       │    │   ...      │    │   ...      │              │
│  │  Trial 50  │───▶│   Train    │───▶│  Evaluate  │──▶ AUC=0.85 │
│  └────────────┘    └────────────┘    └────────────┘              │
│                          Best Trial Selected ───────────────────▶│
└──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE 3: FINAL MODEL                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│  │   Best     │───▶│   Train    │───▶│  SHAP      │              │
│  │  Params    │    │  Final     │    │  Analysis  │              │
│  └────────────┘    └────────────┘    └────────────┘              │
│         │                │                  │                     │
│         ▼                ▼                  ▼                     │
│  ┌────────────┐   ┌────────────┐    ┌────────────┐              │
│  │best_params │   │ model.pkl  │    │ shap_plots │              │
│  │   .json    │   │            │    │   .png     │              │
│  └────────────┘   └────────────┘    └────────────┘              │
└──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FINAL EVALUATION                              │
│                                                                   │
│   Test Set Results:                                              │
│   • AUC-ROC:   0.84  ✓ (target: ≥0.82)                          │
│   • Accuracy:  0.81                                              │
│   • Precision: 0.67                                              │
│   • Recall:    0.62                                              │
│   • F1 Score:  0.64                                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### What Happens Inside `train.py` (Line by Line)

```python
# 1. LOAD DATA
X_train, y_train, X_val, y_val, X_test, y_test, features = load_data()
# Loads the preprocessed parquet files
# X = features (inputs), y = churn label (output)

# 2. BASELINE TRAINING
model = xgb.XGBClassifier(
    n_estimators=100,      # 100 trees
    max_depth=5,           # Each tree max 5 levels deep
    learning_rate=0.1,     # Learn at moderate speed
)
model.fit(X_train, y_train)  # Learn patterns from training data

# 3. EVALUATE BASELINE
predictions = model.predict_proba(X_val)[:, 1]  # Get churn probabilities
auc = roc_auc_score(y_val, predictions)          # Measure performance

# 4. OPTUNA TUNING
def objective(trial):
    # Optuna suggests hyperparameters to try
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        # ... more parameters
    }
    model = train_model(params, X_train, y_train, X_val, y_val)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction="maximize")  # We want higher AUC
study.optimize(objective, n_trials=50)              # Try 50 combinations

# 5. TRAIN FINAL MODEL WITH BEST PARAMS
best_params = study.best_params
final_model = train_model(best_params, X_train, y_train, X_val, y_val)

# 6. SHAP EXPLAINABILITY
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train)
# Now we know which features drive predictions!

# 7. SAVE EVERYTHING
pickle.dump(final_model, open("model.pkl", "wb"))
pickle.dump(explainer, open("explainer.pkl", "wb"))
# Model is ready for deployment!
```

### Common Beginner Questions

**Q: Why not use 100% of data for training?**
A: The model would memorize the training data (overfitting) and fail on new data. We need validation/test sets to measure real-world performance.

**Q: What is overfitting?**
A: When a model learns the training data too well, including noise and outliers. It performs great on training data but poorly on new data.

```
Underfitting          Good Fit           Overfitting
(too simple)         (just right)        (too complex)

    ○ ○                  ○ ○                 ○ ○
  ○     ○              ○     ○           ○ ∿   ∿ ○
────────────         ╭───────╮          ╭─╯╰─╮ ╭─╯
                    ╭╯       ╰╮        ╭╯    ╰─╯

Training: 60%       Training: 85%      Training: 99%
Test: 58%           Test: 84%          Test: 65%
```

**Q: Why 50 Optuna trials?**
A: Balance between thoroughness and time. More trials = better results but longer training. 50 trials usually finds a good solution.

**Q: What if my AUC is below 0.82?**
A: Try:
- More feature engineering
- More Optuna trials
- Different algorithms (LightGBM, CatBoost)
- Collect more data
- Check for data quality issues

**Q: How do I know if my model is good enough?**
A: Compare against:
- Business requirements (e.g., "need 80% of churners caught")
- Baseline (random guessing = 0.5 AUC)
- Previous models
- Industry benchmarks

### Glossary for Beginners

| Term | Definition |
|------|------------|
| **Algorithm** | A set of rules/steps to solve a problem |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve - measures ranking quality |
| **Binary Classification** | Predicting one of two outcomes (churn/no churn) |
| **Cross-Validation** | Testing model on multiple data splits for reliable evaluation |
| **Decision Tree** | A flowchart-like model that makes decisions based on feature values |
| **Epoch** | One complete pass through the training data |
| **Feature** | An input variable used for prediction (e.g., tenure, contract type) |
| **Feature Engineering** | Creating new features from existing ones to improve predictions |
| **Gradient Boosting** | Building models sequentially, each fixing previous errors |
| **Hyperparameter** | A setting chosen before training (not learned from data) |
| **Label** | The target variable we're trying to predict (churn: yes/no) |
| **Model** | A mathematical representation of patterns in data |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Prediction** | The model's output for a given input |
| **Regularization** | Techniques to prevent overfitting (L1, L2) |
| **SHAP** | Method to explain individual predictions |
| **Training** | The process of a model learning patterns from data |
| **Underfitting** | Model is too simple to capture patterns |
| **Validation** | Evaluating model performance on held-out data |

---

## Technologies Used

- **ML Framework:** XGBoost
- **Hyperparameter Tuning:** Optuna
- **Experiment Tracking:** MLflow
- **Explainability:** SHAP
- **API Framework:** FastAPI
- **Data Processing:** Pandas, NumPy, Scikit-learn

## License

MIT License
