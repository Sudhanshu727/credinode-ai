"""
Script 03: Train Gate 1 — Ghost Defaulter Detector (Isolation Forest)
=======================================================================
Gate 1 catches synthetic identities and loan-farming rings using
behavioral DNA fingerprinting (unsupervised anomaly detection).

Algorithm: Isolation Forest
  - Trained ONLY on behavioral features (no explicit fraud labels needed)
  - Anomaly score < threshold → flagged as "Ghost" → pipeline rejected
  - Also evaluates against labeled fraud to measure detection rate

Saves: models/gate1_isolation_forest.joblib

Run: python scripts/03_train_gate1_anomaly.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score
)

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, MODELS_DIR, GATE1_CONFIG, GATE1_FEATURES


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "full_features.csv")
    print(f"  Loaded {len(df):,} merchants")
    print(f"  Ghost/Fraud rate: {df['is_ghost'].mean():.1%}")
    return df


def train_isolation_forest(df: pd.DataFrame):
    X = pd.DataFrame(df[GATE1_FEATURES].fillna(0), columns=GATE1_FEATURES)
    y = df["is_ghost"]

    # Standardize features — keep DataFrame so scaler stores feature names
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Training Isolation Forest...")
    print(f"    n_estimators: {GATE1_CONFIG['n_estimators']}")
    print(f"    contamination: {GATE1_CONFIG['contamination']}")

    iforest = IsolationForest(
        n_estimators=GATE1_CONFIG["n_estimators"],
        contamination=GATE1_CONFIG["contamination"],
        max_samples=GATE1_CONFIG["max_samples"],
        random_state=GATE1_CONFIG["random_state"],
        n_jobs=-1,
    )
    iforest.fit(X_scaled)

    # Raw anomaly scores (more negative = more anomalous)
    raw_scores = iforest.score_samples(X_scaled)

    # Save min/max from training data so inference can reproduce exact normalization
    raw_min = float(raw_scores.min())
    raw_max = float(raw_scores.max())

    # Normalize to [0, 1] where 1 = most legitimate
    anomaly_score_normalized = (raw_scores - raw_min) / (raw_max - raw_min + 1e-9)

    # Threshold: bottom percentile = ghost
    threshold = np.percentile(anomaly_score_normalized,
                              GATE1_CONFIG["threshold_percentile"])
    y_pred = (anomaly_score_normalized < threshold).astype(int)

    return iforest, scaler, anomaly_score_normalized, y_pred, threshold, raw_min, raw_max


def evaluate(y_true, y_pred, anomaly_scores, threshold):
    print("\n  ── Evaluation Results ──────────────────────────")
    print(classification_report(y_true, y_pred,
                                target_names=["Legitimate", "Ghost/Fraud"]))

    auc = roc_auc_score(y_true, 1 - anomaly_scores)
    ap = average_precision_score(y_true, 1 - anomaly_scores)
    cm = confusion_matrix(y_true, y_pred)

    print(f"  ROC-AUC Score:            {auc:.4f}")
    print(f"  Average Precision (AP):   {ap:.4f}")
    print(f"  Anomaly Threshold:        {threshold:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    ghost_caught = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-6)
    legit_blocked = cm[0,1] / (cm[0,0] + cm[0,1] + 1e-6)
    print(f"\n  Ghost Detection Rate:    {ghost_caught:.1%}")
    print(f"  Legitimate Blocked:      {legit_blocked:.1%}  (false alarms)")

    return auc


def plot_results(df, anomaly_scores, threshold):
    """Save diagnostic plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Gate 1: Ghost Defaulter Detector — Isolation Forest", fontsize=14)

    ax = axes[0]
    legit_scores = anomaly_scores[df["is_ghost"] == 0]
    ghost_scores = anomaly_scores[df["is_ghost"] == 1]
    ax.hist(legit_scores, bins=50, alpha=0.6, label="Legitimate", color="#2ecc71")
    ax.hist(ghost_scores, bins=50, alpha=0.6, label="Ghost/Fraud", color="#e74c3c")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Anomaly Score (0=most anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    ax = axes[1]
    feature_means = pd.DataFrame({
        "Feature": GATE1_FEATURES,
        "Ghost Mean": [df[df["is_ghost"]==1][f].mean() for f in GATE1_FEATURES],
        "Legit Mean": [df[df["is_ghost"]==0][f].mean() for f in GATE1_FEATURES],
    })
    feature_means["Separation"] = abs(feature_means["Ghost Mean"] - feature_means["Legit Mean"])
    feature_means = feature_means.sort_values("Separation", ascending=True)
    ax.barh(feature_means["Feature"], feature_means["Separation"], color="#3498db")
    ax.set_xlabel("Mean Separation (Ghost vs Legit)")
    ax.set_title("Feature Discriminability")

    ax = axes[2]
    thresholds = np.linspace(0.01, 0.3, 30)
    detection_rates = []
    false_alarm_rates = []
    for t in thresholds:
        pred = (anomaly_scores < t).astype(int)
        tp = ((pred == 1) & (df["is_ghost"] == 1)).sum()
        fn = ((pred == 0) & (df["is_ghost"] == 1)).sum()
        fp = ((pred == 1) & (df["is_ghost"] == 0)).sum()
        tn = ((pred == 0) & (df["is_ghost"] == 0)).sum()
        detection_rates.append(tp / (tp + fn + 1e-6))
        false_alarm_rates.append(fp / (fp + tn + 1e-6))
    ax.plot(thresholds, detection_rates, label="Detection Rate", color="#e74c3c")
    ax.plot(thresholds, false_alarm_rates, label="False Alarm Rate", color="#f39c12")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Chosen={threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Detection vs False Alarm Tradeoff")
    ax.legend()

    plt.tight_layout()
    plot_path = MODELS_DIR / "gate1_evaluation.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n  ✓ Plot saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Gate 1: Ghost Defaulter Detector")
    print("=" * 60)

    df = load_data()

    iforest, scaler, anomaly_scores, y_pred, threshold, raw_min, raw_max = train_isolation_forest(df)

    auc = evaluate(df["is_ghost"], y_pred, anomaly_scores, threshold)

    plot_results(df, anomaly_scores, threshold)

    # Save model — includes raw_min/max so inference normalization matches training exactly
    artifact = {
        "model": iforest,
        "scaler": scaler,
        "threshold": threshold,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "features": GATE1_FEATURES,
        "auc": auc,
    }
    model_path = MODELS_DIR / "gate1_isolation_forest.joblib"
    joblib.dump(artifact, model_path)
    print(f"\n  ✓ Model saved: {model_path}")
    print("\n✅ Gate 1 training complete!")
    print("Next: python scripts/04_train_gate2a_bsi.py")
