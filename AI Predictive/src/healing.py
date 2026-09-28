"""
healing.py
----------
Self-healing decision engine.

Maps the predicted risk level and failure type to a concrete healing
action, then simulates executing that action and reports the outcome.
"""

import random
import logging

log = logging.getLogger(__name__)


# ── Healing action catalogue ─────────────────────────────────────────────────
HEALING_ACTIONS = {
    "Flaky Test":          "Retry",
    "Deployment Failure":  "Rollback",
    "High CPU/Memory":     "Resource Scaling",
    "Dependency Conflict": "Dependency Fix",
    "Unknown Failure":     "Notify Engineer",
}

# Success rates for each action (realistic but optimistic for demo)
ACTION_SUCCESS_RATES = {
    "Retry":            0.85,
    "Rollback":         0.90,
    "Resource Scaling": 0.80,
    "Dependency Fix":   0.75,
    "Notify Engineer":  0.60,   # human intervention – less certain
}

# Descriptions shown on the dashboard
ACTION_DESCRIPTIONS = {
    "Retry":            "Re-running the failed pipeline stage automatically.",
    "Rollback":         "Rolling back to the last known stable deployment.",
    "Resource Scaling": "Scaling up CPU/memory limits for the build agent.",
    "Dependency Fix":   "Running dependency resolution and cache rebuild.",
    "Notify Engineer":  "Sending alert to the on-call engineer for manual review.",
}


def select_healing_action(failure_type: str) -> str:
    """
    Choose the appropriate healing action based on the failure type.

    Falls back to 'Notify Engineer' for any unrecognised failure type.
    """
    action = HEALING_ACTIONS.get(failure_type, "Notify Engineer")
    log.info("Failure type '%s' → selected action '%s'", failure_type, action)
    return action


def simulate_healing(
    prediction: str,
    risk_level: str,
    failure_type: str,
    failure_probability: float,
) -> dict:
    """
    Simulate executing the chosen healing action.

    Logic:
    ------
    • LOW risk   → healing is almost always unnecessary (build passes on its own).
    • MEDIUM risk → healing is attempted; success rate comes from ACTION_SUCCESS_RATES.
    • HIGH risk   → healing is always attempted; slight success-rate penalty for
                    very high probabilities (>= 85 %).

    Returns a dict with:
      - healing_action      : str   name of the action taken
      - action_description  : str   human-readable description
      - healing_result      : str   "RECOVERED" | "FAILED" | "NOT REQUIRED"
      - final_status        : str   "SUCCESS" | "FAILED"
      - healing_attempted   : bool
      - notes               : str   short explanation
    """
    action = select_healing_action(failure_type)
    description = ACTION_DESCRIPTIONS[action]
    base_success_rate = ACTION_SUCCESS_RATES[action]

    # ── Decision tree ────────────────────────────────────────────────────────
    if prediction == "SUCCESS" and risk_level == "LOW":
        # Build is expected to succeed – no intervention needed
        return {
            "healing_action":    action,
            "action_description": description,
            "healing_result":    "NOT REQUIRED",
            "final_status":      "SUCCESS",
            "healing_attempted": False,
            "notes":             "Build predicted as successful. No healing needed.",
        }

    # For MEDIUM/HIGH risk or a predicted FAILURE – attempt healing
    # Adjust success rate based on how extreme the failure probability is
    if failure_probability >= 85:
        adjusted_rate = base_success_rate * 0.75   # harder to heal very risky builds
    elif failure_probability >= 70:
        adjusted_rate = base_success_rate * 0.90
    else:
        adjusted_rate = base_success_rate

    healed = random.random() < adjusted_rate

    if healed:
        healing_result = "RECOVERED"
        final_status   = "SUCCESS"
        notes = (
            f"[{action}] executed successfully. "
            f"Build recovered from {failure_type}."
        )
    else:
        healing_result = "FAILED"
        final_status   = "FAILED"
        notes = (
            f"[{action}] attempted but could not recover from {failure_type}. "
            "Manual intervention may be required."
        )

    log.info(
        "Healing: action=%s | result=%s | final_status=%s",
        action, healing_result, final_status,
    )

    return {
        "healing_action":     action,
        "action_description": description,
        "healing_result":     healing_result,
        "final_status":       final_status,
        "healing_attempted":  True,
        "notes":              notes,
    }


def get_healing_summary(history: list) -> dict:
    """
    Aggregate healing statistics from a list of pipeline run records.

    Each record is expected to be a dict (as stored in build_history.csv).

    Returns a summary dict used by the dashboard metrics section.
    """
    total         = len(history)
    successful    = sum(1 for r in history if r.get("final_status") == "SUCCESS")
    failed        = sum(1 for r in history if r.get("final_status") == "FAILED")
    auto_healed   = sum(
        1 for r in history
        if r.get("healing_result") == "RECOVERED" and r.get("healing_attempted", False)
    )

    return {
        "total_builds":   total,
        "successful":     successful,
        "failed":         failed,
        "auto_healed":    auto_healed,
    }
