"""
api.py — FastAPI backend

Endpoints:
  GET  /                    Health check
  GET  /predictions         Today's value bets (live odds + model)
  GET  /model-stats         Training metrics + backtest results
  GET  /odds                Raw best-available odds for today's slate
  GET  /teams/{team}/stats  Per-team aggregate stats from historical data
  POST /simulate            Simulate a custom bet (ad-hoc edge calculation)
  GET  /bankroll            Current bankroll + recommended stake sizes

Run with:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from data import fetch_live_odds, get_best_odds, load_historical_data, odds_to_dataframe
from features import build_feature_matrix, build_live_feature_row, engineer_features
from model import backtest, load_model, load_model_metrics, predict_proba
from odds import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_edge,
    find_value_bets,
    flat_bet,
    kelly_bet,
    overround,
    remove_vig,
)

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sports Betting Engine",
    description=(
        "Production-grade ML betting model API. "
        "Surfaces value bets by comparing XGBoost win probabilities "
        "against sportsbook-implied probabilities."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached resources (loaded once per worker process)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    return load_model()


@lru_cache(maxsize=1)
def _get_historical_df():
    return load_historical_data()


def _build_team_stats(hist_df: pd.DataFrame) -> dict[str, dict]:
    """Derive per-team stats from the last 40 completed games."""
    stats: dict[str, dict] = {}
    all_teams = set(hist_df["home_team"].tolist() + hist_df["away_team"].tolist())

    for team in all_teams:
        home_games = hist_df[hist_df["home_team"] == team].tail(40)
        away_games = hist_df[hist_df["away_team"] == team].tail(40)

        all_gf = list(home_games["home_score"]) + list(away_games["away_score"])
        all_ga = list(home_games["away_score"]) + list(away_games["home_score"])
        all_wins_home = list(home_games["home_win"])
        all_wins_away = [1 - w for w in list(away_games["home_win"])]
        all_wins = all_wins_home + all_wins_away

        last5 = all_wins[-5:] if len(all_wins) >= 5 else all_wins

        stats[team] = {
            "win_pct_season": round(sum(all_wins) / max(len(all_wins), 1), 4),
            "goals_for_avg": round(sum(all_gf) / max(len(all_gf), 1), 3),
            "goals_against_avg": round(sum(all_ga) / max(len(all_ga), 1), 3),
            "win_pct_L5": round(sum(last5) / max(len(last5), 1), 4),
            "rest_days": 2,  # default — update from schedule in production
            "h2h_win_pct": 0.5,
            "games_played": len(all_wins),
        }
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class ValueBet(BaseModel):
    game_id: str
    commence_time: str
    home_team: str
    away_team: str
    recommended_bet: str
    bet_side: str
    bet_team: str
    model_probability: float
    implied_probability: float
    edge: float
    edge_pct: float
    american_odds: int
    decimal_odds: float
    overround: float
    flat_bet_amount: float
    kelly_bet_amount: float


class PredictionsResponse(BaseModel):
    timestamp: str
    sport: str
    total_games_today: int
    value_bets_found: int
    min_edge_threshold: float
    bankroll: float
    value_bets: list[ValueBet]


class ModelStatsResponse(BaseModel):
    accuracy: float | None = None
    log_loss: float | None = None
    roc_auc: float | None = None
    brier_score: float | None = None
    cv_log_loss_mean: float | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    n_features: int | None = None
    trained_at: str | None = None
    backtest: dict[str, Any] | None = None
    feature_importances: dict[str, float] | None = None


class SimulateBetRequest(BaseModel):
    home_team: str
    away_team: str
    home_american_odds: float = Field(..., example=-130)
    away_american_odds: float = Field(..., example=110)
    bankroll: float = Field(1000.0, example=1000.0)


class SimulateBetResponse(BaseModel):
    home_team: str
    away_team: str
    model_home_prob: float
    model_away_prob: float
    fair_home_prob: float
    fair_away_prob: float
    home_edge: float
    away_edge: float
    home_edge_pct: float
    away_edge_pct: float
    recommended_side: str | None
    flat_bet: float
    kelly_bet: float
    american_odds_home: float
    american_odds_away: float
    overround: float


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "AI Sports Betting Engine",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/predictions", response_model=PredictionsResponse, tags=["Betting"])
async def get_predictions(
    sport: str = Query(None, description="Override sport key, e.g. icehockey_nhl"),
    min_edge: float = Query(None, description="Minimum edge threshold (0–1)"),
    bankroll: float = Query(None, description="Bankroll for sizing calculations"),
):
    """
    Fetch today's live odds, run the ML model, and return all value bets
    where edge ≥ min_edge threshold.
    """
    sport = sport or settings.sport
    min_edge = min_edge if min_edge is not None else settings.min_edge
    bankroll = bankroll if bankroll is not None else settings.bankroll

    try:
        raw_games = fetch_live_odds(sport=sport)
    except Exception as exc:
        logger.error(f"Odds API error: {exc}")
        raise HTTPException(status_code=502, detail=f"Odds API error: {exc}")

    if not raw_games:
        return PredictionsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sport=sport,
            total_games_today=0,
            value_bets_found=0,
            min_edge_threshold=min_edge,
            bankroll=bankroll,
            value_bets=[],
        )

    odds_df_raw = odds_to_dataframe(raw_games)
    if odds_df_raw.empty:
        raise HTTPException(status_code=404, detail="No h2h markets found in API response.")

    best_odds_df = get_best_odds(odds_df_raw)

    # Build features for each game using historical team stats
    hist_df = _get_historical_df()
    team_stats = _build_team_stats(hist_df)
    model = _get_model()

    pred_rows = []
    for _, row in best_odds_df.iterrows():
        feat_df = build_live_feature_row(row["home_team"], row["away_team"], team_stats)
        feat_df = engineer_features(feat_df)
        prob = float(predict_proba(model, feat_df)[0])
        pred_rows.append(
            {
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_prob": prob,
                "commence_time": str(row["commence_time"]),
            }
        )

    predictions_df = pd.DataFrame(pred_rows)

    value_bets_raw = find_value_bets(
        predictions_df, best_odds_df, bankroll=bankroll, min_edge=min_edge
    )

    return PredictionsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sport=sport,
        total_games_today=len(best_odds_df),
        value_bets_found=len(value_bets_raw),
        min_edge_threshold=min_edge,
        bankroll=bankroll,
        value_bets=[ValueBet(**b) for b in value_bets_raw],
    )


@app.get("/model-stats", response_model=ModelStatsResponse, tags=["Model"])
async def get_model_stats(
    run_backtest: bool = Query(False, description="Re-run backtest (slow)"),
):
    """
    Return model training metrics and optionally a fresh backtest.
    """
    metrics = load_model_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No trained model found. POST /train first.")

    bt_results = None
    if run_backtest:
        try:
            model = _get_model()
            bt_results = backtest(model=model)
        except Exception as exc:
            logger.error(f"Backtest failed: {exc}")
            bt_results = {"error": str(exc)}

    return ModelStatsResponse(
        accuracy=metrics.get("accuracy"),
        log_loss=metrics.get("log_loss"),
        roc_auc=metrics.get("roc_auc"),
        brier_score=metrics.get("brier_score"),
        cv_log_loss_mean=metrics.get("cv_log_loss_mean"),
        train_rows=metrics.get("train_rows"),
        test_rows=metrics.get("test_rows"),
        n_features=metrics.get("n_features"),
        trained_at=metrics.get("trained_at"),
        backtest=bt_results,
        feature_importances=metrics.get("feature_importances"),
    )


@app.get("/odds", tags=["Data"])
async def get_odds(
    sport: str = Query(None),
):
    """Return today's best available odds with vig-removed fair probabilities."""
    sport = sport or settings.sport
    try:
        raw = fetch_live_odds(sport=sport)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    df = odds_to_dataframe(raw)
    best = get_best_odds(df)

    if best.empty:
        return {"games": []}

    results = []
    for _, row in best.iterrows():
        fair_h, fair_a = remove_vig(row["best_home_price"], row["best_away_price"])
        results.append(
            {
                "game_id": row["game_id"],
                "commence_time": str(row["commence_time"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "best_home_price": int(row["best_home_price"]),
                "best_away_price": int(row["best_away_price"]),
                "fair_home_prob": round(fair_h, 4),
                "fair_away_prob": round(fair_a, 4),
                "overround": round(overround(row["best_home_price"], row["best_away_price"]), 4),
            }
        )

    return {"sport": sport, "games": results, "count": len(results)}


@app.get("/teams/{team_name}/stats", tags=["Data"])
async def get_team_stats(team_name: str):
    """
    Return aggregate stats for a specific team derived from historical data.
    URL-encode spaces: 'Boston%20Bruins'
    """
    hist_df = _get_historical_df()
    all_stats = _build_team_stats(hist_df)

    # Case-insensitive search
    matched = {k: v for k, v in all_stats.items() if k.lower() == team_name.lower()}
    if not matched:
        # Fuzzy fallback — partial match
        matched = {k: v for k, v in all_stats.items() if team_name.lower() in k.lower()}

    if not matched:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found.")

    return {"teams": matched}


@app.post("/simulate", response_model=SimulateBetResponse, tags=["Betting"])
async def simulate_bet(req: SimulateBetRequest):
    """
    Run an ad-hoc edge analysis for any matchup + odds you supply.
    Useful for frontend "what-if" tooling.
    """
    hist_df = _get_historical_df()
    team_stats = _build_team_stats(hist_df)
    model = _get_model()

    feat_df = build_live_feature_row(req.home_team, req.away_team, team_stats)
    feat_df = engineer_features(feat_df)

    try:
        home_prob = float(predict_proba(model, feat_df)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    away_prob = 1.0 - home_prob
    fair_h, fair_a = remove_vig(req.home_american_odds, req.away_american_odds)
    home_edge = home_prob - fair_h
    away_edge = away_prob - fair_a

    # Which side has positive edge?
    if home_edge >= settings.min_edge:
        rec_side = "home"
        rec_prob = home_prob
        rec_price = req.home_american_odds
    elif away_edge >= settings.min_edge:
        rec_side = "away"
        rec_prob = away_prob
        rec_price = req.away_american_odds
    else:
        rec_side = None
        rec_prob = max(home_prob, away_prob)
        rec_price = req.home_american_odds

    dec = american_to_decimal(rec_price)

    return SimulateBetResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        model_home_prob=round(home_prob, 4),
        model_away_prob=round(away_prob, 4),
        fair_home_prob=round(fair_h, 4),
        fair_away_prob=round(fair_a, 4),
        home_edge=round(home_edge, 4),
        away_edge=round(away_edge, 4),
        home_edge_pct=round(home_edge * 100, 2),
        away_edge_pct=round(away_edge * 100, 2),
        recommended_side=rec_side,
        flat_bet=flat_bet(req.bankroll),
        kelly_bet=kelly_bet(rec_prob, dec, req.bankroll),
        american_odds_home=req.home_american_odds,
        american_odds_away=req.away_american_odds,
        overround=round(overround(req.home_american_odds, req.away_american_odds), 4),
    )


@app.get("/bankroll", tags=["Betting"])
async def get_bankroll(
    bankroll: float = Query(settings.bankroll, description="Current bankroll"),
):
    """Return recommended stake sizes for both flat-bet and Kelly strategies."""
    return {
        "bankroll": bankroll,
        "flat_bet": {
            "pct": settings.flat_bet_pct,
            "amount": flat_bet(bankroll),
        },
        "kelly": {
            "fraction": settings.kelly_fraction,
            "note": "Kelly amount varies per bet — use /simulate or /predictions for per-game sizing",
        },
        "min_edge_threshold": settings.min_edge,
    }


@app.post("/train", tags=["Model"])
async def trigger_training(n_games: int = Query(2000, description="Synthetic games to generate")):
    """
    Re-generate synthetic data and retrain the model.
    In production, point this at your real data pipeline.
    """
    from data import generate_historical_data
    from model import train as train_model

    # Clear LRU cache so new model + data are picked up
    _get_model.cache_clear()
    _get_historical_df.cache_clear()

    try:
        df = generate_historical_data(n_games=n_games)
        model, metrics = train_model(df=df, save=True)
        return {"status": "success", "metrics": metrics}
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
