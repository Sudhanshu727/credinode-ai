"""
Script 04: Gate 2A — Business Stability Index (BSI) Scorer
============================================================
The BSI transforms raw time-series transaction patterns into a
normalized [0, 1] stability score.

This script:
  1. Validates BSI scores against actual default outcomes
  2. Calibrates the weights via logistic regression
  3. Saves a lightweight BSI scorer object

The BSI is NOT trained on default labels directly — it's a
domain-engineered feature. We validate it and optionally tune weights.

Saves: models/gate2a_bsi_calibrator.joblib

Run: python scripts/04_train_gate2a_bsi.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, MODELS_DIR, BSI_CONFIG

BSI_SUB_FEATURES = [
    "revenue_cv",
    "transaction_entropy",
    "settlement_regularity",
    "active_days_ratio",
    "revenue_trend_slope",
    "avg_daily_revenue",
]


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "full_features.csv")
    # Only non-ghost merchants for BSI calibration
    df = df[df["is_ghost"] == 0].copy()
    print(f"  Loaded {len(df):,} non-ghost merchants for BSI calibration")
    print(f"  Default rate: {df['is_default'].mean():.1%}")
    return df


def analyze_bsi_scores(df: pd.DataFrame):
    """Show correlation between BSI sub-features and defaults."""
    print("\n  ── BSI Feature Correlation with Default ───────────")
    for feat in BSI_SUB_FEATURES + ["bsi_score"]:
        corr = df[feat].corr(df["is_default"])
        direction = "↑ bad" if corr > 0 else "↓ good"
        print(f"    {feat:<30} corr={corr:+.4f}  {direction}")


def calibrate_bsi(df: pd.DataFrame):
    """
    Calibrate BSI weights with logistic regression.
    This tells us which BSI sub-features matter most for defaults.
    """
    X = df[BSI_SUB_FEATURES].fillna(0)
    y = df["is_default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    
    # Cross-validate
    cv_aucs = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"\n  Logistic Regression Calibration (5-fold CV):")
    print(f"    ROC-AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    lr.fit(X_scaled, y)

    # Feature importances from coefficients
    coef_df = pd.DataFrame({
        "Feature": BSI_SUB_FEATURES,
        "Coefficient": lr.coef_[0],
    }).sort_values("Coefficient")
    print(f"\n  Feature Coefficients (higher = more default risk):")
    for _, row in coef_df.iterrows():
        bar = "█" * int(abs(row["Coefficient"]) * 10)
        sign = "+" if row["Coefficient"] > 0 else "-"
        print(f"    {row['Feature']:<35} {sign}{abs(row['Coefficient']):.4f}  {bar}")

    return lr, scaler, cv_aucs.mean()


def plot_bsi_analysis(df: pd.DataFrame):
    """Save BSI diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Gate 2A: Business Stability Index Analysis", fontsize=14)

    colors = {0: "#2ecc71", 1: "#e74c3c"}
    labels = {0: "No Default", 1: "Default"}

    for i, feat in enumerate(BSI_SUB_FEATURES):
        ax = axes[i // 3][i % 3]
        for target in [0, 1]:
            vals = df[df["is_default"] == target][feat].dropna()
            ax.hist(vals, bins=40, alpha=0.6, label=labels[target],
                    color=colors[target], density=True)
        ax.set_title(feat)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = MODELS_DIR / "gate2a_bsi_analysis.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n  ✓ BSI analysis plot saved: {plot_path}")
    plt.close()

    # BSI score vs default rate
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = pd.cut(df["bsi_score"], bins=10)
    default_rate = df.groupby(bins, observed=False)["is_default"].mean()
    bsi_means = df.groupby(bins, observed=False)["bsi_score"].mean()
    ax.bar(range(len(default_rate)), default_rate.values, color="#3498db", alpha=0.8)
    ax.set_xticks(range(len(bsi_means)))
    ax.set_xticklabels([f"{v:.2f}" for v in bsi_means.values], rotation=45)
    ax.set_xlabel("BSI Score Bin (mean)")
    ax.set_ylabel("Default Rate")
    ax.set_title("BSI Score vs Default Rate (Higher BSI = Lower Default Risk)")
    plt.tight_layout()
    plot_path2 = MODELS_DIR / "gate2a_bsi_vs_default.png"
    plt.savefig(plot_path2, dpi=120, bbox_inches="tight")
    print(f"  ✓ BSI vs default plot saved: {plot_path2}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Gate 2A: Business Stability Index")
    print("=" * 60)

    df = load_data()

    analyze_bsi_scores(df)

    lr_calibrator, bsi_scaler, auc = calibrate_bsi(df)

    plot_bsi_analysis(df)

    artifact = {
        "calibrator": lr_calibrator,
        "scaler": bsi_scaler,
        "features": BSI_SUB_FEATURES,
        "weights": BSI_CONFIG["weights"],
        "auc": auc,
    }
    model_path = MODELS_DIR / "gate2a_bsi_calibrator.joblib"
    joblib.dump(artifact, model_path)
    print(f"\n  ✓ BSI calibrator saved: {model_path}")
    print(f"\n  BSI Feature Summary:")
    bsi_stats = df[BSI_SUB_FEATURES + ["bsi_score"]].describe().round(4)
    print(bsi_stats.to_string())
    print("\n✅ Gate 2A (BSI) training complete!")
    print("Next: python scripts/05_train_gate2b_gnn.py")
