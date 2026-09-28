# 🤖 AI-Powered Predictive Self-Healing CI/CD Pipeline

A college micro-project prototype that uses Machine Learning to predict
CI/CD build failures and automatically performs simulated self-healing actions.

---

## 📌 Project Overview

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| ML Model | Random Forest Classifier (scikit-learn) |
| Data | Pandas / NumPy (simulated) |
| Persistence | CSV + JSONL |

---

## 🗂️ Project Structure

```
AI Predictive/
├── app.py                  ← Streamlit dashboard (entry point)
├── requirements.txt        ← Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py   ← Simulated CI/CD build data
│   ├── predictor.py        ← ML training & prediction
│   └── healing.py          ← Self-healing decision engine
│
├── data/
│   ├── training_data.csv   ← Auto-generated on first run
│   └── build_history.csv   ← Live build run history
│
├── model/
│   └── random_forest.pkl   ← Saved model (auto-generated)
│
└── logs/
    ├── pipeline.log        ← Execution log
    └── feedback.jsonl      ← Feedback for future retraining
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

> On first launch the model is trained automatically — no manual setup needed.

---

## ⚙️ How It Works

### Workflow

```
Developer → CI/CD Pipeline → Data Collection → Feature Engineering
    → ML Prediction → Risk Decision → Self-Healing → Dashboard → Feedback
```

### Features used by the ML model

| Feature | Description |
|---|---|
| `commit_size` | Number of lines changed in the commit |
| `build_duration` | Build time in seconds |
| `test_count` | Total number of tests run |
| `failed_tests` | Number of tests that failed |
| `previous_failures` | Count of recent build failures |
| `dependency_changes` | Number of dependency updates |
| `cpu_usage` | CPU usage percentage during build |
| `memory_usage` | Memory usage percentage during build |

### Risk Classification

| Risk | Failure Probability |
|---|---|
| 🟢 LOW | 0 – 39% |
| 🟡 MEDIUM | 40 – 69% |
| 🔴 HIGH | 70 – 100% |

### Self-Healing Actions

| Failure Type | Healing Action |
|---|---|
| Flaky Test | Retry |
| Deployment Failure | Rollback |
| High CPU/Memory | Resource Scaling |
| Dependency Conflict | Dependency Fix |
| Unknown Failure | Notify Engineer |

---

## 🖥️ Dashboard Features

- **Start Pipeline** — Runs a realistic randomised build simulation
- **Simulate Failure** — Injects a high-risk failure scenario
- **Clear History** — Resets build history from the session and CSV
- **Metrics** — Total / Successful / Failed / Auto-Healed build counts
- **Current Pipeline** — Full details of the latest run
- **Build History** — Scrollable table of all past runs
- **ML Model Info** — Model metadata (collapsible)
- **Feedback log** — Every outcome is saved for future retraining

---

## 📋 Requirements

- Python 3.10 or higher
- No internet connection needed after install
- No Docker, Jenkins, AWS, or paid services required
- Runs on any student laptop

---

## 👨‍💻 College Project Notes

This prototype demonstrates:

1. **Simulated CI/CD data generation** with realistic feature distributions
2. **Supervised ML classification** using Random Forest
3. **Probabilistic risk scoring** (0–100% failure probability)
4. **Rule-based healing engine** mapped to failure categories
5. **Feedback loop** — outcomes saved for continuous improvement
6. **Professional Streamlit UI** suitable for live demo

---

*Built with Python · Streamlit · scikit-learn · Pandas · NumPy*
