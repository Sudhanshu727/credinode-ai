"""
CrediNode AI - Central Configuration
All paths, hyperparameters, and constants in one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Auto-create directories
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset URLs ──────────────────────────────────────────────────────────────
DATASETS = {
    "german_credit": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "filename": "german_credit.data",
        "description": "German Credit Risk - UCI (1000 samples, 20 features)"
    },
    "give_me_credit_sample": {
        # We use a public mirror / generate equivalent synthetic
        "url": None,  # Requires Kaggle auth — we generate equivalent
        "filename": "give_me_some_credit.csv",
        "description": "Give Me Some Credit - Default prediction"
    },
}

# ─── Synthetic Data Config ─────────────────────────────────────────────────────
SYNTHETIC_CONFIG = {
    "n_merchants": 10_000,       # Total merchants to generate
    "fraud_rate": 0.08,          # 8% fraud/ghost rate
    "default_rate": 0.15,        # 15% default rate
    "seed": 42,
    "n_graph_edges": 35_000,     # Transaction graph edges
    "time_series_days": 90,      # Days of transaction history per merchant
}

# ─── Gate 1: Isolation Forest ─────────────────────────────────────────────────
GATE1_CONFIG = {
    "contamination": 0.08,       # Expected fraud fraction
    "n_estimators": 200,
    "max_samples": "auto",
    "random_state": 42,
    "threshold_percentile": 10,  # Bottom 10% = ghost/fraud
}

# ─── Gate 2A: Business Stability Index ────────────────────────────────────────
BSI_CONFIG = {
    "lookback_days": 30,
    "min_transactions": 5,       # Minimum to compute BSI
    "weights": {
        "revenue_cv": 0.35,      # Coefficient of variation of daily revenue
        "transaction_entropy": 0.25,
        "settlement_regularity": 0.20,
        "active_days_ratio": 0.20,
    }
}

# ─── Gate 2B: GNN Config ──────────────────────────────────────────────────────
GNN_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 512,
    "edge_weight_threshold": 0.1,  # Min transaction volume for edge
}

# ─── Gate 3: Ensemble Config ──────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "xgb_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 5,   # Adjust for class imbalance
        "random_state": 42,
        "eval_metric": "auc",
        "early_stopping_rounds": 30,
    },
    "lgbm_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    },
    "ensemble_weights": {
        "xgb": 0.5,
        "lgbm": 0.5,
    },
    "shap_top_features": 5,      # Top N features to explain
}

# ─── CrediNode Score ───────────────────────────────────────────────────────────
SCORE_CONFIG = {
    "min_score": 300,
    "max_score": 900,
    "bands": {
        "Excellent": (800, 900),
        "Good":      (700, 799),
        "Fair":      (600, 699),
        "Poor":      (500, 599),
        "Very Poor": (300, 499),
    },
    "band_loan_limits": {        # Max loan limit in INR for each band
        "Excellent": 500_000,
        "Good":      200_000,
        "Fair":       75_000,
        "Poor":       25_000,
        "Very Poor":       0,
    }
}

# ─── API Config ───────────────────────────────────────────────────────────────
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "CrediNode AI API",
    "version": "1.0.0",
}

# ─── Feature Names ────────────────────────────────────────────────────────────
# Features fed into Gate 3 ensemble
ENSEMBLE_FEATURES = [
    # BSI outputs
    "bsi_score",
    "revenue_cv",
    "transaction_entropy",
    "settlement_regularity",
    "active_days_ratio",
    "avg_daily_revenue",
    "revenue_trend_slope",
    # GNN outputs
    "gnn_risk_score",
    "neighbor_avg_default_rate",
    "network_centrality",
    "high_risk_neighbor_count",
    # Behavioral DNA (Gate 1 features)
    "anomaly_score",
    "device_session_entropy",
    "location_variance",
    "temporal_pattern_score",
    # Merchant demographics
    "business_age_days",
    "merchant_category_encoded",
    "city_tier",
    "has_soundbox",
    "qr_active",
]

GATE1_FEATURES = [
    "device_session_entropy",
    "location_variance",
    "temporal_pattern_score",
    "login_hour_entropy",
    "transaction_velocity",
    "unique_device_count",
    "ip_change_frequency",
    "weekend_activity_ratio",
]
