"""
features.py — Feature engineering pipeline

Transforms raw historical / live game data into model-ready feature vectors.

Feature groups:
  1. Season win percentage (home & away)
  2. Rolling goals for / against (last 20 games)
  3. Last-5 form win %
  4. Home vs Away splits
  5. Rest days (fatigue proxy)
  6. Head-to-head record
  7. Differential features (home - away deltas)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Column contracts
# ──────────────────────────────────────────────────────────────────────────────

RAW_FEATURE_COLS = [
    "home_win_pct_season",
    "away_win_pct_season",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "home_win_pct_L5",
    "away_win_pct_L5",
    "home_rest_days",
    "away_rest_days",
    "home_h2h_win_pct",
    "away_h2h_win_pct",
]

ENGINEERED_FEATURE_COLS = [
    # Differentials — often the most predictive
    "win_pct_diff_season",
    "win_pct_diff_L5",
    "goals_for_diff",
    "goals_against_diff",
    "goal_diff_net",          # (home GF - home GA) - (away GF - away GA)
    "rest_diff",
    "h2h_win_pct_diff",
    # Composite strength scores
    "home_strength_score",
    "away_strength_score",
    "strength_diff",
    # Fatigue flags
    "home_back2back",
    "away_back2back",
]

ALL_FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_FEATURE_COLS
TARGET_COL = "home_win"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a DataFrame that contains at minimum the RAW_FEATURE_COLS columns
    (plus TARGET_COL for training) and return it with all engineered features
    appended.

    Safe to call on both historical training data and live prediction frames.
    """
    df = df.copy()

    # ── Differential features ──────────────────────────────────────────────
    df["win_pct_diff_season"] = df["home_win_pct_season"] - df["away_win_pct_season"]
    df["win_pct_diff_L5"] = df["home_win_pct_L5"] - df["away_win_pct_L5"]
    df["goals_for_diff"] = df["home_goals_for_avg"] - df["away_goals_for_avg"]
    df["goals_against_diff"] = df["home_goals_against_avg"] - df["away_goals_against_avg"]

    # Net goal differential: (team's own scoring - defensive weakness) delta
    home_net = df["home_goals_for_avg"] - df["home_goals_against_avg"]
    away_net = df["away_goals_for_avg"] - df["away_goals_against_avg"]
    df["goal_diff_net"] = home_net - away_net

    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["h2h_win_pct_diff"] = df["home_h2h_win_pct"] - df["away_h2h_win_pct"]

    # ── Composite strength scores ──────────────────────────────────────────
    # Weighted blend of season form, recent form, and scoring ability
    df["home_strength_score"] = (
        0.35 * df["home_win_pct_season"]
        + 0.30 * df["home_win_pct_L5"]
        + 0.20 * _normalise(df["home_goals_for_avg"])
        + 0.15 * (1 - _normalise(df["home_goals_against_avg"]))
    )
    df["away_strength_score"] = (
        0.35 * df["away_win_pct_season"]
        + 0.30 * df["away_win_pct_L5"]
        + 0.20 * _normalise(df["away_goals_for_avg"])
        + 0.15 * (1 - _normalise(df["away_goals_against_avg"]))
    )
    df["strength_diff"] = df["home_strength_score"] - df["away_strength_score"]

    # ── Fatigue / back-to-back flags ───────────────────────────────────────
    df["home_back2back"] = (df["home_rest_days"] <= 1).astype(int)
    df["away_back2back"] = (df["away_rest_days"] <= 1).astype(int)

    # ── Validate no NaN in feature columns ────────────────────────────────
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found in features:\n{nan_counts[nan_counts > 0]}")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    logger.debug(f"Feature engineering complete. Shape: {df.shape}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Call engineer_features, then return (X, y).

    y is None when TARGET_COL is absent (live prediction mode).
    """
    df = engineer_features(df)
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y


def build_live_feature_row(
    home_team: str,
    away_team: str,
    team_stats: dict,
) -> pd.DataFrame:
    """
    Construct a single-row feature DataFrame for a live game.

    `team_stats` maps team_name → stats dict with keys matching RAW_FEATURE_COLS
    components. When a stat is unknown, we fall back to league-average defaults.
    """
    defaults = {
        "win_pct_season": 0.5,
        "goals_for_avg": 2.9,
        "goals_against_avg": 2.9,
        "win_pct_L5": 0.5,
        "rest_days": 2,
        "h2h_win_pct": 0.5,
    }

    hs = team_stats.get(home_team, {})
    as_ = team_stats.get(away_team, {})

    row = {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct_season": hs.get("win_pct_season", defaults["win_pct_season"]),
        "away_win_pct_season": as_.get("win_pct_season", defaults["win_pct_season"]),
        "home_goals_for_avg": hs.get("goals_for_avg", defaults["goals_for_avg"]),
        "home_goals_against_avg": hs.get("goals_against_avg", defaults["goals_against_avg"]),
        "away_goals_for_avg": as_.get("goals_for_avg", defaults["goals_for_avg"]),
        "away_goals_against_avg": as_.get("goals_against_avg", defaults["goals_against_avg"]),
        "home_win_pct_L5": hs.get("win_pct_L5", defaults["win_pct_L5"]),
        "away_win_pct_L5": as_.get("win_pct_L5", defaults["win_pct_L5"]),
        "home_rest_days": hs.get("rest_days", defaults["rest_days"]),
        "away_rest_days": as_.get("rest_days", defaults["rest_days"]),
        "home_h2h_win_pct": hs.get("h2h_win_pct", defaults["h2h_win_pct"]),
        "away_h2h_win_pct": as_.get("h2h_win_pct", defaults["h2h_win_pct"]),
    }

    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a series; return 0.5 if constant."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for a quick sanity check."""
    return X.describe().T[["mean", "std", "min", "max"]]
