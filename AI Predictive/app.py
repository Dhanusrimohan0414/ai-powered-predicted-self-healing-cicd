"""
app.py
------
AI-Powered Predictive Self-Healing CI/CD Pipeline
Streamlit Dashboard – entry point.

Run with:
    streamlit run app.py
"""

import os
import sys
import logging
import json
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Make sure src/ is importable when running from project root ──────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_generator import generate_new_build, generate_failure_build
from src.predictor import train_model, predict_build, get_model_info
from src.healing import simulate_healing, get_healing_summary

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
HISTORY_CSV     = os.path.join(DATA_DIR, "build_history.csv")
FEEDBACK_JSONL  = os.path.join(LOGS_DIR, "feedback.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ────────────────────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    """Load build history from CSV; return empty list if file absent."""
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            return df.to_dict(orient="records")
        except Exception as e:
            log.warning("Could not load history: %s", e)
    return []


def save_to_history(record: dict) -> None:
    """Append one pipeline run record to the CSV history file."""
    new_row = pd.DataFrame([record])
    if os.path.exists(HISTORY_CSV):
        existing = pd.read_csv(HISTORY_CSV)
        updated  = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(HISTORY_CSV, index=False)


def save_feedback(record: dict) -> None:
    """
    Persist actual outcome as a feedback entry in JSONL format.
    This data can be used for future model retraining.
    """
    feedback = {
        "build_id":           record.get("build_id"),
        "timestamp":          record.get("timestamp"),
        "prediction":         record.get("prediction"),
        "failure_probability": record.get("failure_probability"),
        "final_status":       record.get("final_status"),
        "healing_result":     record.get("healing_result"),
        "healing_action":     record.get("healing_action"),
        "failure_type":       record.get("failure_type"),
        "feedback_recorded":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(FEEDBACK_JSONL, "a") as f:
        f.write(json.dumps(feedback) + "\n")


def risk_color(risk: str) -> str:
    """Return a hex colour for the risk badge."""
    return {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}.get(risk, "#95a5a6")


def status_icon(status: str) -> str:
    return "✅" if status == "SUCCESS" else "❌"


def healing_icon(result: str) -> str:
    icons = {"RECOVERED": "🔧", "FAILED": "💥", "NOT REQUIRED": "✔️"}
    return icons.get(result, "❓")


# ────────────────────────────────────────────────────────────────────────────
#  Initialise session state
# ────────────────────────────────────────────────────────────────────────────

def init_session():
    if "model" not in st.session_state:
        with st.spinner("🤖 Loading / training ML model …"):
            st.session_state["model"] = train_model()
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    if "current_run" not in st.session_state:
        st.session_state["current_run"] = None


# ────────────────────────────────────────────────────────────────────────────
#  Core pipeline execution
# ────────────────────────────────────────────────────────────────────────────

def run_pipeline(build_data: dict) -> dict:
    """
    Full pipeline:  features → ML prediction → risk → healing → record.
    Returns the complete run record.
    """
    model = st.session_state["model"]

    # Prediction
    result = predict_build(model, build_data)

    # Healing simulation
    healing = simulate_healing(
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        failure_type=build_data["failure_type"],
        failure_probability=result["failure_probability"],
    )

    # Compose the full record
    record = {
        "build_id":           build_data["build_id"],
        "timestamp":          build_data["timestamp"],
        # features
        "commit_size":        build_data["commit_size"],
        "build_duration":     build_data["build_duration"],
        "test_count":         build_data["test_count"],
        "failed_tests":       build_data["failed_tests"],
        "previous_failures":  build_data["previous_failures"],
        "dependency_changes": build_data["dependency_changes"],
        "cpu_usage":          build_data["cpu_usage"],
        "memory_usage":       build_data["memory_usage"],
        # prediction
        "prediction":         result["prediction"],
        "failure_probability": result["failure_probability"],
        "risk_level":         result["risk_level"],
        # healing
        "failure_type":       build_data["failure_type"],
        "healing_action":     healing["healing_action"],
        "healing_result":     healing["healing_result"],
        "healing_attempted":  healing["healing_attempted"],
        "notes":              healing["notes"],
        "final_status":       healing["final_status"],
    }

    # Persist
    save_to_history(record)
    save_feedback(record)
    st.session_state["history"].append(record)
    st.session_state["current_run"] = record

    log.info(
        "Pipeline run: build_id=%s | prediction=%s | risk=%s | "
        "failure_prob=%.1f%% | healing=%s | final=%s",
        record["build_id"], record["prediction"], record["risk_level"],
        record["failure_probability"], record["healing_action"], record["final_status"],
    )

    return record


# ────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ────────────────────────────────────────────────────────────────────────────

def render_metric_cards(summary: dict):
    """Render the four top-level metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Builds",   summary["total_builds"])
    col2.metric("✅ Successful",      summary["successful"])
    col3.metric("❌ Failed",          summary["failed"])
    col4.metric("🔧 Auto-Healed",     summary["auto_healed"])


def render_current_pipeline(run: dict):
    """Show the most recent pipeline run in a structured card layout."""
    st.markdown("---")
    st.subheader("🚀 Current Pipeline Execution")

    risk        = run["risk_level"]
    prediction  = run["prediction"]
    final       = run["final_status"]
    healing_res = run["healing_result"]

    # ── Colour-coded risk badge ──────────────────────────────────────────────
    badge_color = risk_color(risk)
    st.markdown(
        f"""
        <div style="
            background: {badge_color};
            padding: 8px 18px;
            border-radius: 8px;
            display: inline-block;
            color: white;
            font-weight: bold;
            font-size: 1rem;
            margin-bottom: 12px;
        ">Risk Level: {risk}</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-column detail grid ───────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("**Pipeline Details**")
        st.markdown(f"🆔 **Build ID:** `{run['build_id']}`")
        st.markdown(f"🕐 **Time:** {run['timestamp']}")
        st.markdown(
            f"📊 **Failure Probability:** "
            f"`{run['failure_probability']}%`"
        )
        st.markdown(f"🎯 **Prediction:** {status_icon(prediction)} {prediction}")
        st.markdown(f"⚠️  **Failure Type:** {run['failure_type']}")

    with right:
        st.markdown("**Self-Healing Report**")
        st.markdown(f"🔧 **Healing Action:** {run['healing_action']}")
        st.markdown(
            f"💊 **Healing Result:** "
            f"{healing_icon(healing_res)} {healing_res}"
        )
        st.markdown(f"🏁 **Final Status:** {status_icon(final)} {final}")
        st.markdown(f"📝 **Notes:** _{run['notes']}_")

    # ── Feature values expander ──────────────────────────────────────────────
    with st.expander("📋 Build Feature Values", expanded=False):
        feat_cols = st.columns(4)
        features = [
            ("Commit Size",        run["commit_size"],        "lines"),
            ("Build Duration",     run["build_duration"],     "sec"),
            ("Test Count",         run["test_count"],         ""),
            ("Failed Tests",       run["failed_tests"],       ""),
            ("Previous Failures",  run["previous_failures"],  ""),
            ("Dependency Changes", run["dependency_changes"], ""),
            ("CPU Usage",          run["cpu_usage"],          "%"),
            ("Memory Usage",       run["memory_usage"],       "%"),
        ]
        for i, (name, val, unit) in enumerate(features):
            feat_cols[i % 4].metric(name, f"{val} {unit}".strip())


def render_history_table(history: list[dict]):
    """Render the build history as a formatted DataFrame table."""
    st.markdown("---")
    st.subheader("📜 Build History")

    if not history:
        st.info("No build history yet. Click **Start Pipeline** to run your first build.")
        return

    display_cols = [
        "build_id", "timestamp", "failure_probability", "risk_level",
        "prediction", "failure_type", "healing_action", "final_status",
    ]

    df = pd.DataFrame(history)

    # Keep only columns that exist (guard against legacy data)
    cols_present = [c for c in display_cols if c in df.columns]
    df_display = df[cols_present].copy()

    # Human-friendly column names
    df_display.columns = [
        c.replace("_", " ").title() for c in cols_present
    ]

    # Colour-map the Final Status column via pandas Styler
    def highlight_status(val):
        if val == "SUCCESS":
            return "background-color: #d4efdf; color: #1a5e2d;"
        elif val == "FAILED":
            return "background-color: #fadbd8; color: #7b241c;"
        return ""

    status_col = "Final Status"
    if status_col in df_display.columns:
        styled = df_display.style.map(highlight_status, subset=[status_col])
        styled = df_display.style

    st.dataframe(styled, use_container_width=True, height=300)
    st.caption(f"Showing {len(df_display)} build(s) total.")


def render_model_info():
    """Show model metadata in a collapsible section."""
    with st.expander("🧠 ML Model Information", expanded=False):
        info = get_model_info(st.session_state["model"])
        col1, col2 = st.columns(2)
        col1.markdown(f"**Type:** {info['estimator_type']}")
        col1.markdown(f"**Trees:** {info['n_estimators']}")
        col1.markdown(f"**Max Depth:** {info['max_depth']}")
        col2.markdown(f"**Features:** {info['n_features']}")
        col2.markdown(f"**Model file:** `{os.path.basename(info['model_path'])}`")
        col2.markdown(f"**Training CSV:** `{os.path.basename(info['training_csv'])}`")


# ────────────────────────────────────────────────────────────────────────────
#  Main Streamlit app
# ────────────────────────────────────────────────────────────────────────────

def main():
    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="AI-Powered Predictive Self-Healing CI/CD Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Header banner */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header h1 {
            color: #e2e2e2;
            font-size: 1.8rem;
            margin: 0;
        }
        .main-header p {
            color: #a0aec0;
            margin: 4px 0 0 0;
            font-size: 0.95rem;
        }
        /* Metric cards */
        [data-testid="stMetric"] {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 12px 16px;
            border: 1px solid #e0e0e0;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #1a1a2e;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e2e2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Title banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 AI-Powered Predictive Self-Healing CI/CD Pipeline</h1>
            <p>Machine Learning · Anomaly Detection · Automated Recovery · Real-time Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session initialisation ────────────────────────────────────────────────
    init_session()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/robot-2.png",
            width=80,
        )
        st.markdown("## ⚙️ Pipeline Control")
        st.markdown("---")

        # ── Buttons ───────────────────────────────────────────────────────────
        start_btn   = st.button("▶️  Start Pipeline",    use_container_width=True, type="primary")
        failure_btn = st.button("💥  Simulate Failure",  use_container_width=True)
        clear_btn   = st.button("🗑️  Clear History",     use_container_width=True)

        st.markdown("---")
        st.markdown("### Risk Thresholds")
        st.markdown("🟢 **LOW** → < 40%")
        st.markdown("🟡 **MEDIUM** → 40 – 69%")
        st.markdown("🔴 **HIGH** → ≥ 70%")

        st.markdown("---")
        st.markdown("### Healing Actions")
        st.markdown("🔁 Flaky Test → **Retry**")
        st.markdown("↩️  Deployment → **Rollback**")
        st.markdown("📈 High CPU/Mem → **Scale**")
        st.markdown("📦 Dependency → **Fix**")
        st.markdown("📣 Unknown → **Notify**")

        st.markdown("---")
        st.caption("Built with Streamlit · Scikit-learn · Pandas")

    # ── Button handlers ───────────────────────────────────────────────────────

    if start_btn:
        with st.spinner("⚙️ Running pipeline …"):
            build = generate_new_build()
            run_pipeline(build)
        st.success(f"Pipeline completed: **{st.session_state['current_run']['build_id']}**")

    if failure_btn:
        with st.spinner("💥 Injecting failure scenario …"):
            build = generate_failure_build()
            run_pipeline(build)
        st.warning(
            f"Failure scenario executed: "
            f"**{st.session_state['current_run']['build_id']}** — "
            f"Final status: {st.session_state['current_run']['final_status']}"
        )

    if clear_btn:
        st.session_state["history"] = []
        st.session_state["current_run"] = None
        # Remove the CSV file
        if os.path.exists(HISTORY_CSV):
            os.remove(HISTORY_CSV)
        st.info("Build history cleared.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader("📊 Pipeline Metrics")
    summary = get_healing_summary(st.session_state["history"])
    render_metric_cards(summary)

    # ── Current run ───────────────────────────────────────────────────────────
    if st.session_state["current_run"]:
        render_current_pipeline(st.session_state["current_run"])
    else:
        st.markdown("---")
        st.info("👆 Click **Start Pipeline** in the sidebar to run your first CI/CD build simulation.")

    # ── History table ─────────────────────────────────────────────────────────
    render_history_table(st.session_state["history"])

    # ── Model info ────────────────────────────────────────────────────────────
    render_model_info()

    # ── Feedback status ───────────────────────────────────────────────────────
    if os.path.exists(FEEDBACK_JSONL):
        with open(FEEDBACK_JSONL) as f:
            feedback_count = sum(1 for _ in f)
        st.caption(
            f"💾 Feedback log: **{feedback_count}** build outcome(s) recorded → "
            f"`logs/feedback.jsonl`"
        )


if __name__ == "__main__":
    main()
