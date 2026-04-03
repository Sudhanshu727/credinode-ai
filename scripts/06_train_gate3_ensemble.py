"""
Script 06: Gate 3 — Ensemble Risk Scorer + SHAP Explainability
===============================================================
Final fusion layer: combines BSI, GNN topology, and behavioral
features into a single CrediNode Score (300–900) via XGBoost + LightGBM.

SHAP explanations translate the model's output into human-readable
reasons that are shown to merchants via the Fimi chatbot.

Saves:
  - models/gate3_xgb.joblib
  - models/gate3_lgbm.joblib
  - models/gate3_scaler.joblib
  - models/gate3_shap_explainer.joblib

Run: python scripts/06_train_gate3_ensemble.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    PROCESSED_DIR, MODELS_DIR, ENSEMBLE_CONFIG, ENSEMBLE_FEATURES, SCORE_CONFIG
)


def load_and_prepare(df: pd.DataFrame):
    """Prepare features, encode categoricals, handle missing values."""
    features = ENSEMBLE_FEATURES.copy()

    # Encode merchant category
    le = LabelEncoder()
    df["merchant_category_encoded"] = le.fit_transform(
        df["merchant_category"].fillna(0).astype(str)
    )

    X = df[features].fillna(0)
    y = df["is_default"]
    return X, y, le


def evaluate_model(name, model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    preds = (probs >= 0.5).astype(int)

    print(f"\n  ── {name} Results ─────────────────────────────")
    print(classification_report(y_test, preds,
                                target_names=["No Default", "Default"]))
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"  Avg Precision:  {ap:.4f}")
    print(f"  Brier Score:    {brier:.4f}  (lower = better calibrated)")
    return auc, ap, probs


def probability_to_score(prob: float) -> int:
    """
    Convert default probability to CrediNode Score (300–900).
    Lower probability → higher score.
    """
    min_s = SCORE_CONFIG["min_score"]
    max_s = SCORE_CONFIG["max_score"]
    # Inverse relationship: prob=0 → 900, prob=1 → 300
    score = max_s - (prob * (max_s - min_s))
    return int(np.clip(score, min_s, max_s))


def score_to_band(score: int) -> str:
    for band, (lo, hi) in SCORE_CONFIG["bands"].items():
        if lo <= score <= hi:
            return band
    return "Very Poor"


def generate_shap_explanation(shap_values: np.ndarray, feature_names: list,
                               feature_values: np.ndarray, top_n: int = 5) -> list:
    """Convert SHAP values into human-readable reasons."""
    FEATURE_DESCRIPTIONS = {
        "bsi_score": "Business Stability Index",
        "revenue_cv": "Revenue Consistency",
        "transaction_entropy": "Transaction Pattern Diversity",
        "settlement_regularity": "Settlement Timeliness",
        "active_days_ratio": "Business Activity Rate",
        "avg_daily_revenue": "Average Daily Revenue",
        "revenue_trend_slope": "Revenue Growth Trend",
        "gnn_risk_score": "Network Risk Score",
        "neighbor_avg_default_rate": "Neighbor Default Rate",
        "network_centrality": "Network Centrality",
        "high_risk_neighbor_count": "High-Risk Neighbors",
        "anomaly_score": "Identity Authenticity Score",
        "device_session_entropy": "Device Usage Pattern",
        "location_variance": "Location Consistency",
        "temporal_pattern_score": "Temporal Activity Pattern",
        "business_age_days": "Business Age",
        "merchant_category_encoded": "Business Category",
        "city_tier": "City Location Tier",
        "has_soundbox": "Soundbox Usage",
        "qr_active": "QR Code Activity",
    }

    # Sort features by absolute SHAP impact
    impacts = list(zip(feature_names, shap_values, feature_values))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, shap_val, feat_val in impacts[:top_n]:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        direction = "increased" if shap_val > 0 else "decreased"
        risk_word = "raised" if shap_val > 0 else "lowered"

        if feat == "bsi_score":
            if shap_val < 0:
                msg = f"Your Business Stability (BSI={feat_val:.2f}) {risk_word} your risk score — consistent revenue is rewarded."
            else:
                msg = f"Inconsistent business revenue (BSI={feat_val:.2f}) {risk_word} your risk — stabilize your daily transactions."
        elif feat == "settlement_regularity":
            if shap_val < 0:
                msg = f"Regular settlement history ({feat_val:.0%} on-time) {risk_word} your risk."
            else:
                msg = f"Irregular settlements ({feat_val:.0%} on-time) {risk_word} your risk — ensure timely settlements."
        elif feat == "neighbor_avg_default_rate":
            if shap_val > 0:
                msg = f"Your trading partners have a {feat_val:.1%} average default rate, which {risk_word} your risk."
            else:
                msg = f"Your trading network has low default rates ({feat_val:.1%}), which {risk_word} your risk."
        elif feat == "business_age_days":
            if shap_val < 0:
                msg = f"Business age of {int(feat_val)} days builds credibility and {risk_word} your risk."
            else:
                msg = f"A newer business (age={int(feat_val)} days) {risk_word} your risk — build history over time."
        elif feat == "gnn_risk_score":
            if shap_val > 0:
                msg = f"Network contagion signals from your merchant cluster {risk_word} your risk score."
            else:
                msg = f"Your merchant network shows low contagion risk, which {risk_word} your score."
        elif feat == "avg_daily_revenue":
            msg = f"Your average daily revenue (₹{feat_val:,.0f}) {risk_word} your risk profile."
        elif feat == "has_soundbox":
            if feat_val > 0:
                msg = f"Active Soundbox usage demonstrates operational consistency and {risk_word} your risk."
            else:
                msg = f"No Soundbox usage detected — registering one may {risk_word} your score."
        else:
            msg = f"{desc} (value={feat_val:.3g}) {direction} your default risk."

        reasons.append({"feature": feat, "shap": round(shap_val, 4), "message": msg})

    return reasons


def plot_shap_summary(explainer, X_test, feature_names):
    """Save SHAP summary plot."""
    print("\n  Generating SHAP summary plot...")
    shap_values = explainer.shap_values(X_test[:500])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test[:500],
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("Gate 3: Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    path = MODELS_DIR / "gate3_shap_summary.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  ✓ SHAP plot saved: {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Gate 3: Ensemble Scorer + SHAP")
    print("=" * 60)

    # Load data (non-ghost merchants only)
    df = pd.read_csv(PROCESSED_DIR / "full_features.csv")
    df = df[df["is_ghost"] == 0].reset_index(drop=True)
    print(f"  Loaded {len(df):,} merchants | Default rate: {df['is_default'].mean():.1%}")

    X, y, le = load_and_prepare(df)
    feature_names = ENSEMBLE_FEATURES

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Apply SMOTE to handle class imbalance
    print("\n  Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    # X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    import pandas as pd
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns) # Restores feature names
    print(f"  After SMOTE: {len(X_train_sm):,} samples | Ratio: {y_train_sm.mean():.1%} default")

    # ─── XGBoost ──────────────────────────────────────────────────────────────
    print("\n  [1/2] Training XGBoost...")
    eval_set = [(X_test, y_test)]
    xgb_model = xgb.XGBClassifier(
        **{k: v for k, v in ENSEMBLE_CONFIG["xgb_params"].items()
           if k not in ["early_stopping_rounds", "eval_metric"]},
        eval_metric=ENSEMBLE_CONFIG["xgb_params"]["eval_metric"],
        early_stopping_rounds=ENSEMBLE_CONFIG["xgb_params"]["early_stopping_rounds"],
        use_label_encoder=False,
        verbosity=0,
    )
    xgb_model.fit(X_train_sm, y_train_sm,
                  eval_set=eval_set,
                  verbose=False)
    xgb_auc, xgb_ap, xgb_probs = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    # ─── LightGBM ─────────────────────────────────────────────────────────────
    print("\n  [2/2] Training LightGBM...")
    lgbm_model = lgb.LGBMClassifier(**ENSEMBLE_CONFIG["lgbm_params"])
    lgbm_model.fit(X_train_sm, y_train_sm)
    lgbm_auc, lgbm_ap, lgbm_probs = evaluate_model("LightGBM", lgbm_model, X_test, y_test)

    # ─── Ensemble ─────────────────────────────────────────────────────────────
    w_xgb = ENSEMBLE_CONFIG["ensemble_weights"]["xgb"]
    w_lgbm = ENSEMBLE_CONFIG["ensemble_weights"]["lgbm"]
    ensemble_probs = w_xgb * xgb_probs + w_lgbm * lgbm_probs

    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_ap = average_precision_score(y_test, ensemble_probs)
    print(f"\n  ── Ensemble Results ─────────────────────────────")
    print(f"  ROC-AUC:        {ensemble_auc:.4f}")
    print(f"  Avg Precision:  {ensemble_ap:.4f}")

    # ─── SHAP ─────────────────────────────────────────────────────────────────
    print("\n  Building SHAP explainer (TreeExplainer on XGBoost)...")
    shap_explainer = shap.TreeExplainer(xgb_model)
    plot_shap_summary(shap_explainer, pd.DataFrame(X_test.values, columns=feature_names), feature_names)

    # ─── Score conversion demo ────────────────────────────────────────────────
    print("\n  ── Sample Score Conversions ────────────────────")
    sample_probs = [0.05, 0.15, 0.35, 0.55, 0.75]
    for p in sample_probs:
        score = probability_to_score(p)
        band = score_to_band(score)
        loan_limit = SCORE_CONFIG["band_loan_limits"][band]
        print(f"    Default Prob={p:.0%} → Score={score} ({band}) → Loan Limit=₹{loan_limit:,}")

    # ─── Save all artifacts ───────────────────────────────────────────────────
    joblib.dump(xgb_model, MODELS_DIR / "gate3_xgb.joblib")
    joblib.dump(lgbm_model, MODELS_DIR / "gate3_lgbm.joblib")
    joblib.dump(shap_explainer, MODELS_DIR / "gate3_shap_explainer.joblib")
    joblib.dump(le, MODELS_DIR / "gate3_label_encoder.joblib")

    meta = {
        "feature_names": feature_names,
        "ensemble_weights": ENSEMBLE_CONFIG["ensemble_weights"],
        "xgb_auc": xgb_auc,
        "lgbm_auc": lgbm_auc,
        "ensemble_auc": ensemble_auc,
        "score_config": SCORE_CONFIG,
    }
    joblib.dump(meta, MODELS_DIR / "gate3_meta.joblib")

    print(f"\n  ✓ All models saved to: {MODELS_DIR}")
    print(f"\n  ── Final Model Summary ──────────────────────────")
    print(f"  Gate 3 XGBoost AUC:   {xgb_auc:.4f}")
    print(f"  Gate 3 LightGBM AUC:  {lgbm_auc:.4f}")
    print(f"  Ensemble AUC:         {ensemble_auc:.4f}")
    print("\n✅ Gate 3 training complete!")
    print("Next: uvicorn api.main:app --reload --port 8000")
