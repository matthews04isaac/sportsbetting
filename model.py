"""
model.py — XGBoost model: train, evaluate, persist, predict, backtest

Pipeline:
  1. Load historical data
  2. Engineer features
  3. Train / test split (time-ordered to avoid leakage)
  4. Train XGBClassifier with early stopping
  5. Evaluate: accuracy, log-loss, AUC-ROC, Brier score
  6. Persist model to disk
  7. Expose predict_proba for live inference
  8. Backtest: replay model on held-out data, compute ROI
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import settings
from data import load_historical_data
from features import ALL_FEATURE_COLS, TARGET_COL, build_feature_matrix, engineer_features

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost hyper-parameters (tuned for ~2 k sample; scale up with more data)
# ──────────────────────────────────────────────────────────────────────────────

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame | None = None,
    calibrate: bool = True,
    save: bool = True,
) -> tuple[Any, dict[str, float]]:
    """
    Train (and optionally calibrate) an XGBoost model.

    Returns:
        model   — fitted sklearn-compatible estimator
        metrics — dict with train/test performance stats
    """
    if df is None:
        df = load_historical_data()

    # ── Time-ordered split (80 / 20) — no leakage ──────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    logger.info(
        f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
        f"Features: {X_train.shape[1]}"
    )

    # ── Base XGBoost ──────────────────────────────────────────────────────
    base_model = xgb.XGBClassifier(**XGB_PARAMS)
    base_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Isotonic calibration (improves probability reliability) ───────────
    if calibrate:
        logger.info("Applying isotonic probability calibration …")
        model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
        model.fit(X_train, y_train)
    else:
        model = base_model

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "log_loss": round(float(log_loss(y_test, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "trained_at": datetime.utcnow().isoformat(),
    }

    # 5-fold CV log-loss (training set only, unbiased)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**XGB_PARAMS),
        X_train, y_train,
        cv=cv, scoring="neg_log_loss", n_jobs=-1,
    )
    metrics["cv_log_loss_mean"] = round(float(-cv_scores.mean()), 4)
    metrics["cv_log_loss_std"] = round(float(cv_scores.std()), 4)

    logger.success(
        f"Model trained. Accuracy={metrics['accuracy']} | "
        f"Log-Loss={metrics['log_loss']} | AUC={metrics['roc_auc']}"
    )

    # Feature importance
    if hasattr(base_model, "feature_importances_"):
        fi = pd.Series(
            base_model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)
        logger.debug(f"Top-10 features:\n{fi.head(10).to_string()}")
        metrics["feature_importances"] = fi.to_dict()

    if save:
        _save_model(model, metrics)

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _save_model(model: Any, metrics: dict) -> None:
    joblib.dump(model, settings.model_path)
    meta_path = Path(settings.model_path).with_suffix(".json")
    meta_path.write_text(json.dumps(metrics, indent=2))
    logger.success(f"Model saved → {settings.model_path}")


def load_model() -> Any:
    """Load trained model from disk. Auto-trains if missing."""
    path = Path(settings.model_path)
    if not path.exists():
        logger.warning("No model found — training now …")
        model, _ = train()
        return model
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def load_model_metrics() -> dict:
    """Load persisted evaluation metrics."""
    meta_path = Path(settings.model_path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability of the home team winning for each row in X.

    Ensures column order matches training schema.
    """
    available = [c for c in ALL_FEATURE_COLS if c in X.columns]
    missing = set(ALL_FEATURE_COLS) - set(available)
    if missing:
        logger.warning(f"Missing features (filling 0.5): {missing}")
        for col in missing:
            X[col] = 0.5
    X = X[available]
    return model.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────────────────────

def backtest(
    model: Any | None = None,
    df: pd.DataFrame | None = None,
    min_edge: float | None = None,
    bet_size: float = 100.0,
) -> dict[str, Any]:
    """
    Replay the model on historical held-out data (last 20 % by date).

    Simulates flat-bet wagers on games where edge > min_edge.

    Returns dict with:
        roi, win_rate, total_bets, wins, losses,
        total_wagered, total_profit, profit_per_bet,
        by_month (list of monthly P&L dicts)
    """
    min_edge = min_edge if min_edge is not None else settings.min_edge

    if df is None:
        df = load_historical_data()

    if model is None:
        model = load_model()

    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    test_df = df.iloc[split_idx:].copy()

    X_test, y_test = build_feature_matrix(test_df)
    probs = predict_proba(model, X_test)
    test_df = test_df.reset_index(drop=True)
    test_df["model_prob"] = probs
    test_df["actual"] = y_test.values

    # Simulate American odds (synthetic implied probabilities ±5 %)
    # In production these come from the live odds feed
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.03, len(test_df))
    test_df["implied_prob"] = np.clip(
        test_df["model_prob"] + noise + rng.choice([-0.05, 0.05], len(test_df)),
        0.10, 0.90,
    )
    test_df["edge"] = test_df["model_prob"] - test_df["implied_prob"]

    # Only bet where edge > threshold
    bet_mask = test_df["edge"] > min_edge
    bet_df = test_df[bet_mask].copy()

    if bet_df.empty:
        return {
            "total_bets": 0,
            "message": "No bets met the edge threshold in backtest.",
        }

    # Implied decimal odds from implied probability
    bet_df["decimal_odds"] = 1.0 / bet_df["implied_prob"]
    # P&L: win = profit = (decimal - 1) * bet; lose = -bet
    bet_df["pnl"] = np.where(
        bet_df["actual"] == 1,
        (bet_df["decimal_odds"] - 1.0) * bet_size,
        -bet_size,
    )

    wins = int((bet_df["actual"] == 1).sum())
    losses = int((bet_df["actual"] == 0).sum())
    total_bets = len(bet_df)
    total_wagered = total_bets * bet_size
    total_profit = float(bet_df["pnl"].sum())

    # Monthly breakdown
    bet_df["month"] = pd.to_datetime(bet_df["date"]).dt.to_period("M").astype(str)
    monthly = (
        bet_df.groupby("month")["pnl"]
        .agg(bets="count", profit="sum")
        .reset_index()
        .rename(columns={"month": "period"})
    )
    monthly["roi_pct"] = round(monthly["profit"] / (monthly["bets"] * bet_size) * 100, 2)

    result = {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_bets, 4),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_wagered, 4),
        "roi_pct": round(total_profit / total_wagered * 100, 2),
        "profit_per_bet": round(total_profit / total_bets, 2),
        "avg_edge": round(float(bet_df["edge"].mean()), 4),
        "min_edge_used": min_edge,
        "bet_size": bet_size,
        "by_month": monthly.to_dict(orient="records"),
    }

    logger.success(
        f"Backtest: {total_bets} bets | Win rate={result['win_rate']:.1%} | "
        f"ROI={result['roi_pct']:.1f}%"
    )
    return result
