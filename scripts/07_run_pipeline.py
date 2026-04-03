"""
Script 07: Full Inference Pipeline Demo
========================================
Demonstrates the complete CrediNode AI pipeline on sample merchants.
Shows Gate 1 → Gate 2A → Gate 2B → Gate 3 → Score + SHAP.

This is how a real scoring request would flow through the system.

Run: python scripts/07_run_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    PROCESSED_DIR, MODELS_DIR, GATE1_FEATURES, ENSEMBLE_FEATURES,
    SCORE_CONFIG, BSI_CONFIG
)


def load_models():
    """Load all trained model artifacts."""
    models = {}

    gate1_path = MODELS_DIR / "gate1_isolation_forest.joblib"
    if gate1_path.exists():
        models["gate1"] = joblib.load(gate1_path)
        print("  ✓ Gate 1: Isolation Forest loaded")
    else:
        print("  ⚠ Gate 1 model not found — run script 03 first")
        models["gate1"] = None

    gate3_xgb_path = MODELS_DIR / "gate3_xgb.joblib"
    gate3_lgbm_path = MODELS_DIR / "gate3_lgbm.joblib"
    gate3_meta_path = MODELS_DIR / "gate3_meta.joblib"
    gate3_shap_path = MODELS_DIR / "gate3_shap_explainer.joblib"

    if gate3_xgb_path.exists():
        models["xgb"] = joblib.load(gate3_xgb_path)
        models["lgbm"] = joblib.load(gate3_lgbm_path)
        models["meta"] = joblib.load(gate3_meta_path)
        models["shap"] = joblib.load(gate3_shap_path)
        print("  ✓ Gate 3: Ensemble + SHAP loaded")
    else:
        print("  ⚠ Gate 3 models not found — run script 06 first")
        models["xgb"] = None

    return models


def probability_to_score(prob: float) -> int:
    min_s = SCORE_CONFIG["min_score"]
    max_s = SCORE_CONFIG["max_score"]
    return int(np.clip(max_s - (prob * (max_s - min_s)), min_s, max_s))


def score_to_band(score: int) -> str:
    for band, (lo, hi) in SCORE_CONFIG["bands"].items():
        if lo <= score <= hi:
            return band
    return "Very Poor"


def run_gate1(merchant: dict, gate1_artifact) -> dict:
    """
    Gate 1: Ghost / Synthetic Identity Check.
    Returns: {"passed": bool, "anomaly_score": float}
    """
    if gate1_artifact is None:
        return {"passed": True, "anomaly_score": 0.5, "note": "model not loaded"}

    model = gate1_artifact["model"]
    scaler = gate1_artifact["scaler"]
    threshold = gate1_artifact["threshold"]
    features = gate1_artifact["features"]
    raw_min = gate1_artifact.get("raw_min", -0.7)
    raw_max = gate1_artifact.get("raw_max",  0.7)

    X_df = pd.DataFrame([[merchant.get(f, 0) for f in features]], columns=features)
    X_scaled = scaler.transform(X_df)
    raw_score = model.score_samples(X_scaled)[0]
    anomaly_score = float(np.clip((raw_score - raw_min) / (raw_max - raw_min + 1e-9), 0, 1))

    passed = anomaly_score >= threshold
    return {
        "passed": passed,
        "anomaly_score": round(anomaly_score, 4),
        "threshold": round(threshold, 4),
        "verdict": "LEGITIMATE" if passed else "GHOST/FRAUD - REJECTED",
    }


def run_gate2a(merchant: dict) -> dict:
    """Gate 2A: Business Stability Index."""
    weights = BSI_CONFIG["weights"]
    revenue_cv = merchant.get("revenue_cv", 0.5)
    txn_entropy = merchant.get("transaction_entropy", 3.0)
    settlement_reg = merchant.get("settlement_regularity", 0.8)
    active_ratio = merchant.get("active_days_ratio", 0.85)

    bsi_score = (
        weights["revenue_cv"] * (1 - min(revenue_cv, 1)) +
        weights["transaction_entropy"] * min(txn_entropy / 5.0, 1) +
        weights["settlement_regularity"] * settlement_reg +
        weights["active_days_ratio"] * active_ratio
    )
    return {
        "bsi_score": round(bsi_score, 4),
        "components": {
            "revenue_consistency": round(1 - min(revenue_cv, 1), 4),
            "transaction_diversity": round(min(txn_entropy / 5.0, 1), 4),
            "settlement_regularity": round(settlement_reg, 4),
            "active_days_ratio": round(active_ratio, 4),
        }
    }


def run_gate3(merchant: dict, models: dict) -> dict:
    """Gate 3: Ensemble risk scoring + SHAP."""
    if models.get("xgb") is None:
        # Fallback: simple heuristic scoring
        bsi = merchant.get("bsi_score", 0.5)
        gnn = merchant.get("gnn_risk_score", 0.2)
        prob = (1 - bsi) * 0.5 + gnn * 0.5
        score = probability_to_score(prob)
        band = score_to_band(score)
        return {
            "default_probability": round(prob, 4),
            "credinode_score": score,
            "band": band,
            "loan_limit": SCORE_CONFIG["band_loan_limits"][band],
            "shap_reasons": [{"message": "Score computed via heuristic (models not trained yet)"}],
            "note": "Train models first for full scoring"
        }

    features = models["meta"]["feature_names"]
    X = np.array([[merchant.get(f, 0) for f in features]])
    X_df = pd.DataFrame(X, columns=features)

    w_xgb = models["meta"]["ensemble_weights"]["xgb"]
    w_lgbm = models["meta"]["ensemble_weights"]["lgbm"]

    p_xgb = models["xgb"].predict_proba(X_df)[0, 1]
    p_lgbm = models["lgbm"].predict_proba(X_df)[0, 1]
    prob = w_xgb * p_xgb + w_lgbm * p_lgbm

    score = probability_to_score(prob)
    band = score_to_band(score)
    loan_limit = SCORE_CONFIG["band_loan_limits"][band]

    # SHAP explanation (inline)
    shap_vals = models["shap"].shap_values(X_df)[0]
    FEATURE_LABELS = {
        "bsi_score": "Business Stability", "settlement_regularity": "Settlement Timeliness",
        "neighbor_avg_default_rate": "Neighbor Default Rate", "business_age_days": "Business Age",
        "gnn_risk_score": "Network Contagion Risk", "avg_daily_revenue": "Daily Revenue",
        "revenue_cv": "Revenue Consistency", "has_soundbox": "Soundbox Usage",
    }
    impacts = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for idx, sv in impacts[:5]:
        fname = features[idx]
        label = FEATURE_LABELS.get(fname, fname.replace("_"," ").title())
        direction = "raised" if sv > 0 else "lowered"
        reasons.append({"feature": fname, "shap": round(float(sv), 4),
                        "message": f"{label} {direction} your risk score."})

    return {
        "default_probability": round(float(prob), 4),
        "xgb_probability": round(float(p_xgb), 4),
        "lgbm_probability": round(float(p_lgbm), 4),
        "credinode_score": score,
        "band": band,
        "loan_limit": loan_limit,
        "shap_reasons": reasons[:5],
    }


def score_merchant(merchant: dict, models: dict) -> dict:
    """Run a merchant through the full CrediNode pipeline."""
    result = {
        "merchant_id": merchant.get("merchant_id", "UNKNOWN"),
        "pipeline": []
    }

    # ── Gate 1 ──────────────────────────────────────────────────────────────
    gate1_result = run_gate1(merchant, models.get("gate1"))
    result["gate1"] = gate1_result
    result["pipeline"].append("GATE1")

    if not gate1_result["passed"]:
        result["status"] = "REJECTED"
        result["rejection_reason"] = "Identity verification failed (Ghost/Fraud detected)"
        result["credinode_score"] = 300
        result["band"] = "Very Poor"
        result["loan_limit"] = 0
        return result

    # ── Gate 2A ─────────────────────────────────────────────────────────────
    gate2a_result = run_gate2a(merchant)
    result["gate2a"] = gate2a_result
    result["pipeline"].append("GATE2A")
    merchant["bsi_score"] = gate2a_result["bsi_score"]

    # ── Gate 3 ──────────────────────────────────────────────────────────────
    gate3_result = run_gate3(merchant, models)
    result["gate3"] = gate3_result
    result["pipeline"].append("GATE3")

    result["status"] = "SCORED"
    result["credinode_score"] = gate3_result["credinode_score"]
    result["band"] = gate3_result["band"]
    result["loan_limit"] = gate3_result["loan_limit"]
    result["default_probability"] = gate3_result["default_probability"]
    result["shap_reasons"] = gate3_result.get("shap_reasons", [])

    return result


# ─── Sample Merchant Profiles ────────────────────────────────────────────────

SAMPLE_MERCHANTS = [
    {
        "merchant_id": "M000001",
        "name": "Sharma Kirana Store",
        "description": "Stable 5-year grocery store, Tier 1 city",
        # BSI features
        "revenue_cv": 0.15,
        "transaction_entropy": 4.2,
        "settlement_regularity": 0.95,
        "active_days_ratio": 0.92,
        "avg_daily_revenue": 8500,
        "revenue_trend_slope": 25.0,
        # Gate 1 features
        "device_session_entropy": 0.78,
        "location_variance": 0.08,
        "temporal_pattern_score": 0.85,
        "login_hour_entropy": 0.45,
        "transaction_velocity": 2.1,
        "unique_device_count": 2,
        "ip_change_frequency": 0.1,
        "weekend_activity_ratio": 0.22,
        # Network features
        "gnn_risk_score": 0.12,
        "neighbor_avg_default_rate": 0.06,
        "network_centrality": 0.15,
        "high_risk_neighbor_count": 1,
        # Demographics
        "business_age_days": 1825,
        "merchant_category_encoded": 0,
        "city_tier": 1,
        "has_soundbox": 1,
        "qr_active": 1,
        "anomaly_score": 0.82,
    },
    {
        "merchant_id": "M000002",
        "name": "Raju Fast Food",
        "description": "New restaurant, volatile revenue, high-risk neighborhood",
        "revenue_cv": 0.65,
        "transaction_entropy": 2.1,
        "settlement_regularity": 0.55,
        "active_days_ratio": 0.60,
        "avg_daily_revenue": 2200,
        "revenue_trend_slope": -15.0,
        "device_session_entropy": 0.62,
        "location_variance": 0.25,
        "temporal_pattern_score": 0.45,
        "login_hour_entropy": 0.65,
        "transaction_velocity": 4.5,
        "unique_device_count": 4,
        "ip_change_frequency": 0.8,
        "weekend_activity_ratio": 0.42,
        "gnn_risk_score": 0.58,
        "neighbor_avg_default_rate": 0.38,
        "network_centrality": 0.05,
        "high_risk_neighbor_count": 6,
        "business_age_days": 180,
        "merchant_category_encoded": 1,
        "city_tier": 2,
        "has_soundbox": 0,
        "qr_active": 1,
        "anomaly_score": 0.55,
    },
    {
        "merchant_id": "M000003",
        "name": "[GHOST] Bot Account XZ9",
        "description": "Synthetic identity — loan farming attempt",
        "revenue_cv": 0.95,
        "transaction_entropy": 0.4,
        "settlement_regularity": 0.10,
        "active_days_ratio": 0.25,
        "avg_daily_revenue": 12000,
        "revenue_trend_slope": -200.0,
        "device_session_entropy": 0.08,   # Bot sessions: very low entropy
        "location_variance": 0.92,        # Many IPs/locations
        "temporal_pattern_score": 0.05,
        "login_hour_entropy": 0.98,
        "transaction_velocity": 35.0,     # Unusually high velocity
        "unique_device_count": 18,
        "ip_change_frequency": 12.5,
        "weekend_activity_ratio": 0.52,
        "gnn_risk_score": 0.88,
        "neighbor_avg_default_rate": 0.75,
        "network_centrality": 0.01,
        "high_risk_neighbor_count": 15,
        "business_age_days": 22,
        "merchant_category_encoded": 4,
        "city_tier": 3,
        "has_soundbox": 0,
        "qr_active": 1,
        "anomaly_score": 0.04,
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Full Pipeline Demo")
    print("=" * 60)

    print("\n  Loading models...")
    models = load_models()

    print("\n" + "=" * 60)
    for merchant_data in SAMPLE_MERCHANTS:
        print(f"\n  🏪 Scoring: {merchant_data['name']}")
        print(f"     {merchant_data['description']}")
        print("  " + "-" * 55)

        result = score_merchant(merchant_data, models)

        print(f"  Pipeline:    {' → '.join(result['pipeline'])}")
        print(f"  Status:      {result['status']}")

        if result["status"] == "REJECTED":
            print(f"  ❌ REJECTED: {result['rejection_reason']}")
        else:
            score = result["credinode_score"]
            band = result["band"]
            prob = result.get("default_probability", 0)
            limit = result.get("loan_limit", 0)

            bar_len = int((score - 300) / 600 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)

            print(f"\n  CrediNode Score: {score} [{bar}]")
            print(f"  Band:            {band}")
            print(f"  Default Prob:    {prob:.1%}")
            print(f"  Max Loan Limit:  ₹{limit:,}")

            if result.get("shap_reasons"):
                print(f"\n  📊 Top Factors:")
                for reason in result["shap_reasons"][:3]:
                    print(f"    → {reason['message']}")

        print()
