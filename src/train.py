"""
train.py — XGBoost training with MLflow tracking, Optuna tuning, and SHAP explanations.
Run: python src/train.py
"""

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# MLflow setup
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_URI = MLRUNS_DIR.absolute().as_uri()
EXPERIMENT_NAME = "telco-churn-xgboost"


def load_data():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "feature_names.json") as f:
        features = json.load(f)

    X_train, y_train = train[features], train["Churn"]
    X_val,   y_val   = val[features],   val["Churn"]
    X_test,  y_test  = test[features],  test["Churn"]
    return X_train, y_train, X_val, y_val, X_test, y_test, features


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_true, y_pred_proba), 4),
    }


def train_model(params, X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def save_shap_plots(model, X_train, feature_names, artifacts_dir):
    print("   Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Global summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False, max_display=10)
    plt.tight_layout()
    summary_path = str(artifacts_dir / "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Single-prediction waterfall
    sample_idx = 42
    plt.figure(figsize=(10, 6))
    shap_exp = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_train.iloc[sample_idx].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    waterfall_path = str(artifacts_dir / "shap_waterfall_sample.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(artifacts_dir / "shap_feature_importance.csv", index=False)

    print("   [PASS] SHAP plots saved")
    return shap_values, explainer, importance_df


def explain(customer_features: dict, model, explainer, feature_names: list) -> dict:
    """Return per-feature SHAP contributions for a single prediction."""
    row = pd.DataFrame([customer_features])[feature_names]
    shap_vals = explainer.shap_values(row)[0]
    contributions = {
        feat: round(float(val), 6)
        for feat, val in zip(feature_names, shap_vals)
    }
    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    return sorted_contribs


# ─────────────────────────────────────────────
# PHASE 1: Baseline
# ─────────────────────────────────────────────
def run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, features):
    print("\n[BASELINE] Training BASELINE model...")
    baseline_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline") as run:
        model = train_model(baseline_params, X_train, y_train, X_val, y_val)
        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        val_metrics  = compute_metrics(y_val, val_proba)
        test_metrics = compute_metrics(y_test, test_proba)

        mlflow.log_params({**baseline_params, "dataset_version": "v1", "author": "soham"})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tags({"model_type": "xgboost", "stage": "baseline"})
        mlflow.xgboost.log_model(model, "model")

        # Feature importance plot
        fig, ax = plt.subplots(figsize=(8, 6))
        feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)[:15]
        feat_imp.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("XGBoost Feature Importance (Baseline)")
        ax.invert_yaxis()
        plt.tight_layout()
        imp_path = str(ARTIFACTS_DIR / "feature_importance.png")
        plt.savefig(imp_path, dpi=150)
        plt.close()
        mlflow.log_artifact(imp_path)

        run_id = run.info.run_id

    print(f"   Val  AUC: {val_metrics['auc_roc']} | F1: {val_metrics['f1']}")
    print(f"   Test AUC: {test_metrics['auc_roc']} | F1: {test_metrics['f1']}")
    return model, baseline_params, val_metrics, test_metrics, run_id


# ─────────────────────────────────────────────
# PHASE 2: Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────
def run_optuna(X_train, y_train, X_val, y_val, X_test, y_test, features, baseline_auc):
    print("\n[OPTUNA] Running Optuna (50 trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5),
        }
        model = train_model(params, X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=1)

    best_params = study.best_params
    best_val_auc = study.best_value
    print(f"   Best val AUC: {best_val_auc:.4f} (baseline: {baseline_auc:.4f}, delta: +{best_val_auc - baseline_auc:.4f})")

    # Save best params
    with open(ARTIFACTS_DIR / "best_params.json", "w") as f:
        json.dump({"params": best_params, "val_auc": best_val_auc}, f, indent=2)

    # Train final tuned model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="optuna-tuned") as run:
        model = train_model(best_params, X_train, y_train, X_val, y_val)
        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        val_metrics  = compute_metrics(y_val, val_proba)
        test_metrics = compute_metrics(y_test, test_proba)

        mlflow.log_params({**best_params, "n_optuna_trials": 50, "dataset_version": "v1", "author": "soham"})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tags({"model_type": "xgboost", "stage": "tuned", "tuner": "optuna"})

        # SHAP
        shap_values, explainer, importance_df = save_shap_plots(model, X_train, features, ARTIFACTS_DIR)
        mlflow.log_artifact(str(ARTIFACTS_DIR / "shap_summary.png"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "shap_waterfall_sample.png"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "shap_feature_importance.csv"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "best_params.json"))

        # Save model + explainer locally
        with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(ARTIFACTS_DIR / "explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        with open(ARTIFACTS_DIR / "features.json", "w") as f:
            json.dump(features, f)

        mlflow.log_artifact(str(ARTIFACTS_DIR / "model.pkl"))
        mlflow.xgboost.log_model(model, "xgboost_model")

        run_id = run.info.run_id

    print(f"   Val  AUC: {val_metrics['auc_roc']} | F1: {val_metrics['f1']}")
    print(f"   Test AUC: {test_metrics['auc_roc']} | F1: {test_metrics['f1']}")
    print(f"\n   Top 5 features by SHAP:")
    for _, row in importance_df.head(5).iterrows():
        print(f"     {row['feature']:30s} {row['mean_abs_shap']:.4f}")

    return model, explainer, best_params, val_metrics, test_metrics, run_id


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(BASE_DIR)

    X_train, y_train, X_val, y_val, X_test, y_test, features = load_data()
    print(f"Data loaded -> train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)} | features:{len(features)}")

    baseline_model, baseline_params, baseline_val, baseline_test, baseline_run_id = \
        run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, features)

    tuned_model, explainer, best_params, tuned_val, tuned_test, tuned_run_id = \
        run_optuna(X_train, y_train, X_val, y_val, X_test, y_test, features, baseline_val["auc_roc"])

    improvement = tuned_test["auc_roc"] - baseline_test["auc_roc"]
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Baseline  Test AUC: {baseline_test['auc_roc']}")
    print(f"  Tuned     Test AUC: {tuned_test['auc_roc']}")
    print(f"  Improvement:        +{improvement:.4f}")
    print(f"  Acceptance criteria (>=0.82): {'[PASS] PASS' if tuned_test['auc_roc'] >= 0.82 else '[FAIL] FAIL'}")
    print(f"  Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"  MLflow runs: {MLFLOW_TRACKING_URI}")
    print(f"{'='*55}")