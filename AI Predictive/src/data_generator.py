"""
data_generator.py
-----------------
Generates simulated historical CI/CD build data for training the ML model,
and generates realistic feature values for each new pipeline execution.
"""

import numpy as np
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta


# ── Reproducibility seed for training data ──────────────────────────────────
RANDOM_SEED = 42

# ── Failure type catalogue ───────────────────────────────────────────────────
FAILURE_TYPES = [
    "Flaky Test",
    "Deployment Failure",
    "High CPU/Memory",
    "Dependency Conflict",
    "Unknown Failure",
]


def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create a synthetic historical build dataset.

    Each row represents one CI/CD pipeline run with realistic feature
    distributions that reflect real-world build behaviour.

    Returns a pandas DataFrame ready for model training.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Raw features ────────────────────────────────────────────────────────
    commit_size         = rng.integers(1, 500, n_samples)          # lines changed
    build_duration      = rng.integers(30, 600, n_samples)         # seconds
    test_count          = rng.integers(5, 300, n_samples)
    failed_tests        = rng.integers(0, 50, n_samples)
    previous_failures   = rng.integers(0, 10, n_samples)
    dependency_changes  = rng.integers(0, 20, n_samples)
    cpu_usage           = rng.uniform(10, 100, n_samples).round(2)
    memory_usage        = rng.uniform(20, 95, n_samples).round(2)

    # ── Deterministic label with realistic noise ─────────────────────────────
    # A build is more likely to fail when:
    #   • many tests failed
    #   • high previous failure count
    #   • lots of dependency changes
    #   • CPU or memory is pushed high
    failure_score = (
        (failed_tests / (test_count + 1)) * 40
        + previous_failures * 3
        + dependency_changes * 1.5
        + np.where(cpu_usage > 80, 10, 0)
        + np.where(memory_usage > 85, 8, 0)
    )

    # Normalise to [0, 1] probability then threshold at 0.5
    prob = failure_score / (failure_score.max() + 1e-9)
    noise = rng.uniform(-0.05, 0.05, n_samples)
    build_success = (prob + noise < 0.5).astype(int)   # 1 = success, 0 = failure

    df = pd.DataFrame({
        "commit_size":        commit_size,
        "build_duration":     build_duration,
        "test_count":         test_count,
        "failed_tests":       failed_tests,
        "previous_failures":  previous_failures,
        "dependency_changes": dependency_changes,
        "cpu_usage":          cpu_usage,
        "memory_usage":       memory_usage,
        "build_success":      build_success,
    })

    return df


def generate_new_build() -> dict:
    """
    Simulate one fresh pipeline execution.

    Returns a dict with:
      - build_id          : unique identifier
      - timestamp         : ISO datetime string
      - feature values    : all 8 input features
      - failure_type      : randomly assigned failure type (used by healing engine)
    """
    rng = np.random.default_rng()   # non-seeded for live variation

    # Occasionally push values into the risky zone so failures can occur
    risky = random.random() < 0.35   # ~35 % of builds are risky

    commit_size        = int(rng.integers(200, 500)) if risky else int(rng.integers(1, 200))
    build_duration     = int(rng.integers(300, 600)) if risky else int(rng.integers(30, 300))
    test_count         = int(rng.integers(5, 300))
    failed_tests       = int(rng.integers(10, 50))  if risky else int(rng.integers(0, 5))
    previous_failures  = int(rng.integers(4, 10))   if risky else int(rng.integers(0, 3))
    dependency_changes = int(rng.integers(8, 20))   if risky else int(rng.integers(0, 7))
    cpu_usage          = round(float(rng.uniform(75, 99)), 2) if risky else round(float(rng.uniform(10, 74)), 2)
    memory_usage       = round(float(rng.uniform(80, 95)), 2) if risky else round(float(rng.uniform(20, 79)), 2)

    failure_type = random.choice(FAILURE_TYPES)

    return {
        "build_id":           f"BUILD-{uuid.uuid4().hex[:8].upper()}",
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit_size":        commit_size,
        "build_duration":     build_duration,
        "test_count":         test_count,
        "failed_tests":       failed_tests,
        "previous_failures":  previous_failures,
        "dependency_changes": dependency_changes,
        "cpu_usage":          cpu_usage,
        "memory_usage":       memory_usage,
        "failure_type":       failure_type,
    }


def generate_failure_build() -> dict:
    """
    Simulate a pipeline run that is heavily biased toward failure.
    Used by the 'Simulate Failure' button in the dashboard.
    """
    rng = np.random.default_rng()

    return {
        "build_id":           f"BUILD-{uuid.uuid4().hex[:8].upper()}",
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit_size":        int(rng.integers(400, 500)),
        "build_duration":     int(rng.integers(500, 600)),
        "test_count":         int(rng.integers(100, 300)),
        "failed_tests":       int(rng.integers(30, 50)),
        "previous_failures":  int(rng.integers(7, 10)),
        "dependency_changes": int(rng.integers(15, 20)),
        "cpu_usage":          round(float(rng.uniform(88, 99)), 2),
        "memory_usage":       round(float(rng.uniform(88, 95)), 2),
        "failure_type":       random.choice(FAILURE_TYPES),
    }


def get_feature_columns() -> list:
    """Return the ordered list of feature column names used by the ML model."""
    return [
        "commit_size",
        "build_duration",
        "test_count",
        "failed_tests",
        "previous_failures",
        "dependency_changes",
        "cpu_usage",
        "memory_usage",
    ]
