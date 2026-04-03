# 🏦 CrediNode AI — Merchant Credit Underwriting Engine
### FIN-O-HACK | Paytm × ASSETS DTU | Track 2: AI for Small Businesses

---

## 🚀 What This Is

CrediNode AI is a real-time, multi-gated merchant credit scoring and fraud prevention system built specifically for India's UPI/QR-code ecosystem. It moves beyond static CIBIL scores by leveraging:

- **Gate 1 — Ghost Defaulter Detector:** Isolation Forest anomaly detection to catch synthetic identities and loan-farming rings
- **Gate 2A — Business Stability Index (BSI):** Time-series volatility scoring from transaction behavior
- **Gate 2B — Cascade Risk GNN:** Graph Convolutional Network that models default contagion across merchant networks
- **Gate 3 — Ensemble Scorer + SHAP:** XGBoost/LightGBM ensemble with human-readable SHAP explanations
- **💡 BONUS — Fimi AI Chatbot:** Natural language credit advisor powered by CrediNode's output

---

## 📁 Project Structure

```
credinode/
├── data/                   # Datasets (auto-downloaded by scripts)
│   ├── raw/
│   └── processed/
├── models/                 # Saved trained models
├── scripts/
│   ├── 01_download_data.py       # Downloads all datasets
│   ├── 02_generate_synthetic.py  # Generates India-specific merchant data
│   ├── 03_train_gate1_anomaly.py # Train Isolation Forest
│   ├── 04_train_gate2a_bsi.py    # Train BSI scorer
│   ├── 05_train_gate2b_gnn.py    # Train Graph Neural Network
│   ├── 06_train_gate3_ensemble.py# Train XGBoost ensemble
│   └── 07_run_pipeline.py        # Run full inference pipeline
├── api/
│   └── main.py                   # FastAPI backend
├── dashboard/
│   └── index.html                # Full dashboard (open in browser)
├── config/
│   └── settings.py               # All configs
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (5 Steps)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & generate data

```bash
python scripts/01_download_data.py
python scripts/02_generate_synthetic.py
```

### 3. Train all models (run in order)

```bash
python scripts/03_train_gate1_anomaly.py
python scripts/04_train_gate2a_bsi.py
python scripts/05_train_gate2b_gnn.py
python scripts/06_train_gate3_ensemble.py
```

### 4. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Open the Dashboard

Open `dashboard/index.html` in your browser. Done! 🎉

---

## 📊 Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| `Give Me Some Credit` | Kaggle | Default prediction base |
| `German Credit Data` | UCI ML Repo | Credit risk features |
| `PaySim` | Kaggle | Fraud transaction patterns |
| `Synthetic Merchant Data` | Generated | India SMB behavioral profiles |

---

## 🧠 Innovation Highlights

1. **Behavioral DNA Fingerprinting** — Device entropy + temporal interaction patterns prevent synthetic account fraud
2. **Cascade Risk GNN** — Graph convolutional network models how default spreads like contagion across merchant networks
3. **SHAP Explainability in Hindi/English** — Every score has a human-readable reason via Fimi chatbot
4. **Soundbox Telemetry Features** — QR scan frequency, Soundbox ping regularity as leading creditworthiness indicators
5. **Cold-Start Handling** — Special scoring path for merchants with <30 days of history using neighbor embeddings

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/score` | Score a single merchant |
| POST | `/batch_score` | Score multiple merchants |
| GET | `/merchant/{id}` | Get merchant details + history |
| GET | `/network/{id}` | Get merchant's risk network |
| GET | `/health` | API health check |

---

## 📬 Fimi Chatbot

The Fimi conversational interface (in the dashboard) interprets CrediNode scores into plain-language explanations:

> *"Aapka CrediNode Score 742 hai. Aapke Soundbox usage ki consistency ne score badhayi. Lekin aapke ek supplier ka network stress zyada hai — yeh risk factor hai."*
