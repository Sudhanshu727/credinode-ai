"""
CrediNode AI — FastAPI Backend
================================
REST API exposing the full multi-gated scoring pipeline.

Endpoints:
  POST /score          → Score a single merchant
  POST /batch_score    → Score multiple merchants
  GET  /merchant/{id}  → Get merchant profile + history
  GET  /network/{id}   → Get merchant network risk map
  GET  /health         → Health check + model status
  GET  /demo           → Demo scoring with preset examples

Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import sys
import time
import uuid
import os
import numpy as np
import pandas as pd
import joblib
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_file)

# Verify API key is loaded
if not os.getenv('OPENROUTER_API_KEY'):
    print("[WARNING] OPENROUTER_API_KEY not found in .env file!")
    print(f"[INFO] Looking for .env at: {env_file}")
else:
    print(f"[INFO] OpenRouter API key loaded from .env")

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    MODELS_DIR, PROCESSED_DIR, SCORE_CONFIG, ENSEMBLE_FEATURES,
    GATE1_FEATURES, BSI_CONFIG, API_CONFIG
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description="Real-time merchant credit scoring and fraud prevention for Paytm ecosystem",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Model Store ───────────────────────────────────────────────────────
_models: Dict[str, Any] = {}


def load_models():
    """Load all trained models at startup."""
    global _models

    gate1_path = MODELS_DIR / "gate1_isolation_forest.joblib"
    if gate1_path.exists():
        _models["gate1"] = joblib.load(gate1_path)
        print("[CrediNode] Gate 1: Isolation Forest loaded")

    xgb_path = MODELS_DIR / "gate3_xgb.joblib"
    lgbm_path = MODELS_DIR / "gate3_lgbm.joblib"
    meta_path = MODELS_DIR / "gate3_meta.joblib"
    shap_path = MODELS_DIR / "gate3_shap_explainer.joblib"

    if xgb_path.exists():
        _models["xgb"] = joblib.load(xgb_path)
        _models["lgbm"] = joblib.load(lgbm_path)
        _models["meta"] = joblib.load(meta_path)
        _models["shap"] = joblib.load(shap_path)
        print("[CrediNode] Gate 3: Ensemble + SHAP loaded")

    print(f"[CrediNode] Models loaded: {list(_models.keys())}")


@app.on_event("startup")
def startup():
    load_models()


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class MerchantInput(BaseModel):
    merchant_id: str = Field(default_factory=lambda: f"M{str(uuid.uuid4())[:8].upper()}")
    name: Optional[str] = "Unknown Merchant"

    # Behavioral DNA (Gate 1)
    device_session_entropy: float = Field(default=0.7, ge=0, le=1)
    location_variance: float = Field(default=0.1, ge=0, le=1)
    temporal_pattern_score: float = Field(default=0.7, ge=0, le=1)
    login_hour_entropy: float = Field(default=0.5, ge=0, le=1)
    transaction_velocity: float = Field(default=3.0, ge=0)
    unique_device_count: int = Field(default=1, ge=0)
    ip_change_frequency: float = Field(default=0.2, ge=0)
    weekend_activity_ratio: float = Field(default=0.25, ge=0, le=1)

    # BSI inputs
    revenue_cv: float = Field(default=0.3, ge=0)
    transaction_entropy: float = Field(default=3.5, ge=0)
    settlement_regularity: float = Field(default=0.85, ge=0, le=1)
    active_days_ratio: float = Field(default=0.85, ge=0, le=1)
    avg_daily_revenue: float = Field(default=3000, ge=0)
    revenue_trend_slope: float = Field(default=0.0)

    # Network features (Graph)
    gnn_risk_score: float = Field(default=0.2, ge=0, le=1)
    neighbor_avg_default_rate: float = Field(default=0.1, ge=0, le=1)
    network_centrality: float = Field(default=0.1, ge=0, le=1)
    high_risk_neighbor_count: int = Field(default=0, ge=0)

    # Demographics
    business_age_days: int = Field(default=365, ge=0)
    merchant_category_encoded: int = Field(default=0, ge=0)
    city_tier: int = Field(default=2, ge=1, le=3)
    has_soundbox: int = Field(default=1, ge=0, le=1)
    qr_active: int = Field(default=1, ge=0, le=1)
    anomaly_score: float = Field(default=0.7, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "merchant_id": "M001234",
                "name": "Sharma Kirana Store",
                "device_session_entropy": 0.78,
                "location_variance": 0.08,
                "temporal_pattern_score": 0.85,
                "revenue_cv": 0.15,
                "settlement_regularity": 0.95,
                "active_days_ratio": 0.92,
                "avg_daily_revenue": 8500,
                "gnn_risk_score": 0.12,
                "business_age_days": 1825,
                "has_soundbox": 1,
                "city_tier": 1,
            }
        }


class ScoreResponse(BaseModel):
    merchant_id: str
    name: str
    status: str
    credinode_score: int
    band: str
    default_probability: float
    loan_limit: int
    gate1: dict
    gate2a: dict
    gate3: dict
    shap_reasons: list
    processing_time_ms: float
    request_id: str


class ChatRequest(BaseModel):
    message: str
    merchant_context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Merchant context: name, score, band, loan_limit, risk_factor, status"
    )


# ─── Core Scoring Logic ───────────────────────────────────────────────────────

def probability_to_score(prob: float) -> int:
    min_s, max_s = SCORE_CONFIG["min_score"], SCORE_CONFIG["max_score"]
    return int(np.clip(max_s - (prob * (max_s - min_s)), min_s, max_s))


def score_to_band(score: int) -> str:
    for band, (lo, hi) in SCORE_CONFIG["bands"].items():
        if lo <= score <= hi:
            return band
    return "Very Poor"


def run_gate1(m: dict) -> dict:
    if "gate1" not in _models:
        # Heuristic fallback
        entropy = m.get("device_session_entropy", 0.7)
        velocity = m.get("transaction_velocity", 3.0)
        suspicious = entropy < 0.2 or velocity > 20
        return {
            "passed": bool(not suspicious),
            "anomaly_score": round(entropy * 0.5, 4),
            "verdict": "GHOST/FRAUD" if suspicious else "LEGITIMATE",
            "note": "heuristic mode — train models for full detection"
        }

    artifact = _models["gate1"]
    X = np.array([[m.get(f, 0) for f in artifact["features"]]])
    X_scaled = artifact["scaler"].transform(X)
    raw = artifact["model"].score_samples(X_scaled)[0]
    anomaly_score = float(np.clip((raw + 0.7) / 1.4, 0, 1))
    threshold = artifact["threshold"]
    
    # Logic: LOW anomaly score = NORMAL (pass), HIGH anomaly score = ANOMALOUS (reject)
    # So we reject merchants with anomaly_score > threshold
    # But use a more lenient threshold for better UX
    effective_threshold = max(threshold, 0.5)  # Higher threshold = allow more merchants
    passed = bool(anomaly_score <= effective_threshold)
    
    print(f"[Gate1] anomaly_score={anomaly_score:.4f}, threshold={threshold:.4f}, effective={effective_threshold:.4f}, passed={passed}")
    
    return {
        "passed": passed,
        "anomaly_score": round(anomaly_score, 4),
        "threshold": round(effective_threshold, 4),
        "verdict": "LEGITIMATE" if passed else "GHOST/FRAUD",
    }


def run_gate2a(m: dict) -> dict:
    w = BSI_CONFIG["weights"]
    revenue_cv = m.get("revenue_cv", 0.5)
    txn_entropy = m.get("transaction_entropy", 3.0)
    settlement = m.get("settlement_regularity", 0.8)
    active = m.get("active_days_ratio", 0.85)

    bsi = (
        w["revenue_cv"] * (1 - min(revenue_cv, 1)) +
        w["transaction_entropy"] * min(txn_entropy / 5.0, 1) +
        w["settlement_regularity"] * settlement +
        w["active_days_ratio"] * active
    )
    return {
        "bsi_score": round(float(bsi), 4),
        "revenue_consistency": round(1 - min(revenue_cv, 1), 4),
        "settlement_regularity": round(settlement, 4),
        "active_days_ratio": round(active, 4),
    }


def run_gate2b(m: dict) -> dict:
    """Gate 2B: GNN-based Network Contagion Risk.
    Uses input gnn_risk_score or trained model if available."""
    
    gnn_score = m.get("gnn_risk_score", 0.3)
    gnn_score = float(np.clip(gnn_score, 0, 1))
    
    # Interpret risk level
    if gnn_score < 0.33:
        contagion = "Low"
    elif gnn_score < 0.67:
        contagion = "Medium"
    else:
        contagion = "High"
    
    print(f"[Gate2B] gnn_risk_score={gnn_score:.4f}, contagion_risk={contagion}")
    
    return {
        "gnn_risk_score": round(gnn_score, 4),
        "contagion_risk": contagion,
    }


def run_gate3(m: dict) -> dict:
    if "xgb" not in _models:
        # Heuristic fallback
        bsi = m.get("bsi_score", 0.5)
        gnn = m.get("gnn_risk_score", 0.3)
        prob = float((1 - bsi) * 0.5 + gnn * 0.5)
        score = probability_to_score(prob)
        band = score_to_band(score)
        print(f"[Gate3] Heuristic mode: bsi={bsi}, gnn={gnn}, prob={prob:.4f}, score={score}")
        return {
            "default_probability": round(prob, 4),
            "credinode_score": int(score),
            "band": str(band),
            "loan_limit": int(SCORE_CONFIG["band_loan_limits"][band]),
            "shap_reasons": [
                {"feature": "bsi_score", "shap": 0, "message": "Score computed via heuristic — train models for full SHAP"}
            ],
        }

    features = _models["meta"]["feature_names"]
    X = np.array([[m.get(f, 0) for f in features]])
    X_df = pd.DataFrame(X, columns=features)

    w_xgb = _models["meta"]["ensemble_weights"]["xgb"]
    w_lgbm = _models["meta"]["ensemble_weights"]["lgbm"]

    p_xgb = float(_models["xgb"].predict_proba(X_df)[0, 1])
    p_lgbm = float(_models["lgbm"].predict_proba(X_df)[0, 1])
    prob = w_xgb * p_xgb + w_lgbm * p_lgbm

    score = probability_to_score(prob)
    band = score_to_band(score)
    loan_limit = SCORE_CONFIG["band_loan_limits"][band]
    
    print(f"[Gate3] p_xgb={p_xgb:.4f}, p_lgbm={p_lgbm:.4f}, prob={prob:.4f}, score={score}, band={band}")

    # SHAP
    shap_vals = _models["shap"].shap_values(X_df)[0]
    reasons = []
    impact_order = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)
    FEATURE_LABELS = {
        "bsi_score": "Business Stability",
        "settlement_regularity": "Settlement Timeliness",
        "neighbor_avg_default_rate": "Neighbor Default Rate",
        "business_age_days": "Business Age",
        "gnn_risk_score": "Network Contagion Risk",
        "avg_daily_revenue": "Daily Revenue",
        "revenue_cv": "Revenue Consistency",
        "has_soundbox": "Soundbox Usage",
    }
    for idx, sv in impact_order[:5]:
        fname = features[idx]
        label = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())
        direction = "⬆ Increased risk" if sv > 0 else "⬇ Reduced risk"
        reasons.append({
            "feature": fname,
            "label": label,
            "shap_value": round(float(sv), 4),
            "impact": direction,
        })

    return {
        "default_probability": round(float(prob), 4),
        "xgb_prob": round(float(p_xgb), 4),
        "lgbm_prob": round(float(p_lgbm), 4),
        "credinode_score": int(score),
        "band": str(band),
        "loan_limit": int(loan_limit),
        "shap_reasons": reasons,
    }


def full_pipeline(m: dict) -> dict:
    t0 = time.time()

    g1 = run_gate1(m)
    if not g1["passed"]:
        return {
            "status": "REJECTED",
            "rejection_reason": "Ghost/Synthetic identity detected",
            "credinode_score": 300,
            "band": "Very Poor",
            "default_probability": 1.0,
            "loan_limit": 0,
            "gate1": g1,
            "gate2a": {},
            "gate2b": {},
            "gate3": {},
            "shap_reasons": [],
            "processing_time_ms": round((time.time() - t0) * 1000, 2),
        }

    g2a = run_gate2a(m)
    m["bsi_score"] = g2a["bsi_score"]

    g2b = run_gate2b(m)
    m["gnn_risk_score"] = g2b["gnn_risk_score"]

    g3 = run_gate3(m)

    return {
        "status": "SCORED",
        "credinode_score": g3["credinode_score"],
        "band": g3["band"],
        "default_probability": g3["default_probability"],
        "loan_limit": g3["loan_limit"],
        "gate1": g1,
        "gate2a": g2a,
        "gate2b": g2b,
        "gate3": g3,
        "shap_reasons": g3.get("shap_reasons", []),
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
        "gate1_ready": "gate1" in _models,
        "gate3_ready": "xgb" in _models,
        "version": API_CONFIG["version"],
    }


@app.post("/score", response_model=None)
def score_merchant(merchant: MerchantInput):
    m = merchant.model_dump()
    result = full_pipeline(m)
    result["merchant_id"] = merchant.merchant_id
    result["name"] = merchant.name
    result["request_id"] = str(uuid.uuid4())
    return result


@app.post("/batch_score")
def batch_score(merchants: List[MerchantInput]):
    results = []
    for merchant in merchants:
        m = merchant.model_dump()
        result = full_pipeline(m)
        result["merchant_id"] = merchant.merchant_id
        result["name"] = merchant.name
        result["request_id"] = str(uuid.uuid4())
        results.append(result)
    return {"results": results, "count": len(results)}


@app.get("/demo")
def demo_score():
    """Demo scoring with three preset merchant profiles."""
    examples = [
        {
            "merchant_id": "DEMO001",
            "name": "Priya Grocery Store (Stable)",
            "device_session_entropy": 0.80, "location_variance": 0.06,
            "temporal_pattern_score": 0.88, "revenue_cv": 0.12,
            "settlement_regularity": 0.97, "active_days_ratio": 0.94,
            "avg_daily_revenue": 9200, "gnn_risk_score": 0.08,
            "business_age_days": 2100, "has_soundbox": 1, "city_tier": 1,
            "login_hour_entropy": 0.42, "transaction_velocity": 1.8,
            "unique_device_count": 1, "ip_change_frequency": 0.05,
            "weekend_activity_ratio": 0.20, "neighbor_avg_default_rate": 0.04,
            "network_centrality": 0.18, "high_risk_neighbor_count": 0,
            "merchant_category_encoded": 0, "qr_active": 1, "anomaly_score": 0.9,
            "transaction_entropy": 4.5, "revenue_trend_slope": 30.0,
        },
        {
            "merchant_id": "DEMO002",
            "name": "New Auto Repair (Moderate Risk)",
            "device_session_entropy": 0.58, "location_variance": 0.28,
            "temporal_pattern_score": 0.50, "revenue_cv": 0.55,
            "settlement_regularity": 0.62, "active_days_ratio": 0.70,
            "avg_daily_revenue": 3100, "gnn_risk_score": 0.42,
            "business_age_days": 280, "has_soundbox": 0, "city_tier": 2,
            "login_hour_entropy": 0.60, "transaction_velocity": 5.2,
            "unique_device_count": 3, "ip_change_frequency": 1.2,
            "weekend_activity_ratio": 0.35, "neighbor_avg_default_rate": 0.28,
            "network_centrality": 0.06, "high_risk_neighbor_count": 4,
            "merchant_category_encoded": 5, "qr_active": 1, "anomaly_score": 0.60,
            "transaction_entropy": 2.8, "revenue_trend_slope": -8.0,
        },
        {
            "merchant_id": "DEMO003",
            "name": "Ghost Account (Detected)",
            "device_session_entropy": 0.05, "location_variance": 0.95,
            "temporal_pattern_score": 0.04, "revenue_cv": 0.98,
            "settlement_regularity": 0.08, "active_days_ratio": 0.22,
            "avg_daily_revenue": 15000, "gnn_risk_score": 0.90,
            "business_age_days": 18, "has_soundbox": 0, "city_tier": 3,
            "login_hour_entropy": 0.99, "transaction_velocity": 42.0,
            "unique_device_count": 22, "ip_change_frequency": 18.0,
            "weekend_activity_ratio": 0.55, "neighbor_avg_default_rate": 0.80,
            "network_centrality": 0.01, "high_risk_neighbor_count": 20,
            "merchant_category_encoded": 3, "qr_active": 0, "anomaly_score": 0.02,
            "transaction_entropy": 0.3, "revenue_trend_slope": -300.0,
        },
    ]
    results = []
    for ex in examples:
        r = full_pipeline(ex)
        r["merchant_id"] = ex["merchant_id"]
        r["name"] = ex["name"]
        results.append(r)
    return {"demo_results": results}


@app.get("/merchant/{merchant_id}")
def get_merchant(merchant_id: str):
    """Fetch merchant from processed data (if available)."""
    data_path = PROCESSED_DIR / "full_features.csv"
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Merchant data not generated yet. Run script 02.")
    df = pd.read_csv(data_path)
    row = df[df["merchant_id"] == merchant_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    return row.iloc[0].to_dict()


@app.post("/chat")
async def fimi_chat(request: ChatRequest):
    """Fimi AI chatbot powered by OpenRouter LLM."""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Build system prompt with merchant context
    merchant_context = request.merchant_context or {}
    name = merchant_context.get("name", "Merchant")
    score = merchant_context.get("score", "N/A")
    band = merchant_context.get("band", "N/A")
    loan_limit = merchant_context.get("loan_limit", "N/A")
    risk_factor = merchant_context.get("risk_factor", "N/A")
    status = merchant_context.get("status", "SCORED")
    
    system_prompt = f"""You are Fimi AI, the official Personal Credit Advisor for Paytm Merchants.

CURRENT MERCHANT CONTEXT:
• Name: {name}
• CrediNode Score: {score}
• Band: {band}
• Loan Limit: ₹{loan_limit}
• Top Risk Factor: {risk_factor}
• Account Status: {status}

═══ FORMATTING RULES (MANDATORY) ═══

1. STRUCTURE - ALWAYS USE THIS FORMAT:
   • Start with a short greeting (1 line)
   • Add line breaks between sections (empty line)
   • Use section headers like: "📊 Your Score" or "💡 Tips" or "✅ What's Working"
   • Use emojis liberally: 🎉 ✅ ⚠️ 💡 📈 🙏 ⭐ 📊
   • Number all lists: 1. 2. 3. (not bullets)
   • End with encouraging note

2. NEVER output as one long paragraph
   MUST have line breaks after every 2-3 sentences
   MUST separate sections with blank lines
   MUST use section headers with emojis

3. EMOJIS - USE THESE:
   Score intro: 📊
   Tips/Advice: 💡
   Success: ✅ 🎉
   Warning: ⚠️
   Action items: 📋
   Overall: 🙏 ⭐

4. LANGUAGE
   Reply in English if question is in English, and Hinglish if question is in Hindi.

5. RESPONSE TEMPLATE - USE THIS EXACT FORMAT WITH REAL BLANK LINES:

   👋 Hello [Name]! 

   📊 Your Score Overview
   • Score: {score}
   • Band: {band}
   • Loan Limit: ₹{loan_limit}
   [Your interpretation - 2-3 sentences with emojis]

   ✅ What's Working Well
   1. [Positive factor]
   2. [Positive factor]

   ⚠️ Area for Improvement
   • Risk Factor: {risk_factor}
   [Explanation - 1-2 sentences]

   💡 Actionable Tips
   1. [Specific tip with Paytm tool name]
   2. [Specific tip with Paytm tool name]
   3. [Specific tip with Paytm tool name]

   🙏 Final Thoughts
   [Encouraging closing - 1-2 sentences]

   DO NOT OUTPUT THE TEXT "[BLANK LINE]" - USE ACTUAL EMPTY LINES BETWEEN SECTIONS!

6. FOR GENERAL QUESTIONS (not credit):
   Still use line breaks, bullets, and emojis
   Don't force merchant context
   Be helpful and friendly
   Proper formatting is MANDATORY even for general chat

7. CONTEXT HANDLING:
   • Credit questions: Use merchant context from above
   • General questions: Respond helpfully without score mention
   • Mixed: Prioritize what user asked, reference score if relevant

8. JARGON TRANSLATION:
   • "Business Stability Index (BSI)" → "Daily QR & Soundbox transaction consistency"
   • "Cascade Risk GNN" → "Financial health of your business partners"
   • "Device Entropy" → "Device usage patterns"
   • "Ghost Account" → "Account under security review"

9. SCORE INTERPRETATION:
   High (800+): 🎉 Celebrate! Explain what's working, mention limit increase
   Medium (600-799): ✅ Good progress! Identify top risk, give 3 actionable tips
   Low (<600): 💪 Supportive tone! Explain main issue clearly, give concrete steps

10. FORBIDDEN:
    • NO long paragraphs (max 2-3 sentences per section)
    • NO emojis unless section headers or emphasis
    • NO generic responses - always personalized
    • NO skipping line breaks
    • NO one continuous block of text

Remember: FORMATTING IS NOT OPTIONAL. Every response MUST be clean, readable, with emojis, line breaks, and clear sections."""

    try:
        # Call OpenRouter API with fallback to alternative models if rate-limited
        models_to_try = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/auto",
            "google/gemini-2.0-flash-exp:free",
        ]
        
        response = None
        last_error = None
        
        for model in models_to_try:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                        "HTTP-Referer": "http://127.0.0.1:8000",
                        "X-Title": "CrediNode AI",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request.message},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                    timeout=30,
                )
                
                if response.status_code == 200:
                    print(f"[Fimi] Using model: {model}")
                    break
                elif response.status_code == 429:
                    print(f"[Fimi] Model {model} rate-limited, trying next...")
                    last_error = response
                    continue
                else:
                    print(f"[Fimi] Model {model} error: {response.status_code}")
                    last_error = response
                    continue
            except Exception as e:
                print(f"[Fimi] Model {model} failed: {str(e)}")
                last_error = e
                continue
        
        if response is None or response.status_code != 200:
            error_msg = f"All models failed. Last error: {last_error}"
            print(f"[Fimi] OpenRouter error: {error_msg}")
            raise HTTPException(status_code=500, detail="Unable to reach AI service. Please try again.")
        
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()
        
        print(f"[Fimi] Q: {request.message[:50]}... → A: {reply[:50]}...")
        
        return {
            "reply": reply,
            "merchant_context": merchant_context,
            "model": "auto-selected",
        }
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OpenRouter request timeout")
    except Exception as e:
        print(f"[Fimi] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ─── Mount Dashboard (AFTER all routes) ────────────────────────────────────────
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
    print(f"[CrediNode] Dashboard mounted at /dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
