"""
scheduler.py — Background job runner

Runs two recurring jobs:
  1. Retrain model every Sunday at 03:00 UTC
  2. Refresh odds cache every 10 minutes (logs value bets to file)

Usage (run alongside the API):
  python scheduler.py &
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import schedule
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def job_retrain():
    logger.info("=== Scheduled retraining job started ===")
    try:
        from data import generate_historical_data
        from model import train
        df = generate_historical_data(n_games=2000)
        _, metrics = train(df=df, save=True)
        logger.success(f"Retraining complete. AUC={metrics.get('roc_auc')}")
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}")


def job_refresh_odds():
    logger.info("Refreshing live odds …")
    try:
        from data import fetch_live_odds, get_best_odds, load_historical_data, odds_to_dataframe
        from features import build_live_feature_row, engineer_features
        from model import load_model, predict_proba
        from odds import find_value_bets
        import pandas as pd

        raw = fetch_live_odds()
        if not raw:
            logger.info("No games on slate.")
            return

        odds_df = get_best_odds(odds_to_dataframe(raw))
        hist_df = load_historical_data()

        # Inline team stats (avoid circular import with api.py)
        all_teams = set(hist_df["home_team"].tolist() + hist_df["away_team"].tolist())
        team_stats = {}
        for team in all_teams:
            hg = hist_df[hist_df["home_team"] == team].tail(40)
            ag = hist_df[hist_df["away_team"] == team].tail(40)
            all_w = list(hg["home_win"]) + [1-w for w in list(ag["home_win"])]
            gf = list(hg["home_score"]) + list(ag["away_score"])
            ga = list(hg["away_score"]) + list(ag["home_score"])
            last5 = all_w[-5:] if all_w else []
            team_stats[team] = {
                "win_pct_season": sum(all_w)/max(len(all_w),1),
                "goals_for_avg": sum(gf)/max(len(gf),1),
                "goals_against_avg": sum(ga)/max(len(ga),1),
                "win_pct_L5": sum(last5)/max(len(last5),1),
                "rest_days": 2, "h2h_win_pct": 0.5,
            }

        model = load_model()
        pred_rows = []
        for _, row in odds_df.iterrows():
            feat = build_live_feature_row(row["home_team"], row["away_team"], team_stats)
            feat = engineer_features(feat)
            prob = float(predict_proba(model, feat)[0])
            pred_rows.append({
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_prob": prob,
                "commence_time": str(row["commence_time"]),
            })

        preds_df = pd.DataFrame(pred_rows)
        value_bets = find_value_bets(preds_df, odds_df)

        snapshot = {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "games": len(odds_df),
            "value_bets": value_bets,
        }

        out_path = LOG_DIR / "latest_value_bets.json"
        out_path.write_text(json.dumps(snapshot, indent=2))
        logger.success(f"Odds refreshed. {len(value_bets)} value bet(s) found → {out_path}")

    except Exception as exc:
        logger.error(f"Odds refresh failed: {exc}")


if __name__ == "__main__":
    logger.add(LOG_DIR / "scheduler_{time}.log", rotation="1 week")

    # Retrain every Sunday at 03:00
    schedule.every().sunday.at("03:00").do(job_retrain)

    # Refresh odds every 10 minutes
    schedule.every(10).minutes.do(job_refresh_odds)

    logger.info("Scheduler running. Ctrl+C to stop.")

    # Run once on startup
    job_refresh_odds()

    while True:
        schedule.run_pending()
        time.sleep(30)
