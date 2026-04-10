"""
main.py — CLI entry point

Usage:
  python main.py train        # generate data + train model
  python main.py backtest     # run backtest on held-out data
  python main.py predict      # fetch live odds + output value bets to console
  python main.py serve        # start FastAPI server (prod: use uvicorn directly)
"""

from __future__ import annotations

import sys
import json

from loguru import logger


def cmd_train():
    from data import generate_historical_data
    from model import train
    logger.info("=== Training pipeline ===")
    df = generate_historical_data(n_games=2000)
    model, metrics = train(df=df, save=True)
    print("\n── Model Metrics ──────────────────────────────")
    for k, v in metrics.items():
        if k not in ("feature_importances", "by_month"):
            print(f"  {k}: {v}")
    print("───────────────────────────────────────────────\n")


def cmd_backtest():
    from model import backtest, load_model
    logger.info("=== Backtest ===")
    model = load_model()
    results = backtest(model=model, bet_size=100.0)
    print("\n── Backtest Results ───────────────────────────")
    for k, v in results.items():
        if k != "by_month":
            print(f"  {k}: {v}")
    if "by_month" in results:
        print("\n  Monthly breakdown:")
        for m in results["by_month"]:
            print(f"    {m['period']}: {m['bets']} bets | P&L ${m['profit']:.2f} | ROI {m['roi_pct']}%")
    print("───────────────────────────────────────────────\n")


def cmd_predict():
    from data import fetch_live_odds, get_best_odds, load_historical_data, odds_to_dataframe
    from features import build_live_feature_row, engineer_features
    from model import load_model, predict_proba
    from odds import find_value_bets
    import pandas as pd

    logger.info("=== Live Prediction Run ===")
    model = load_model()
    hist_df = load_historical_data()

    # Build team stats
    from api import _build_team_stats
    team_stats = _build_team_stats(hist_df)

    try:
        raw_games = fetch_live_odds()
    except Exception as e:
        logger.error(f"Could not fetch odds: {e}")
        sys.exit(1)

    if not raw_games:
        print("No games on slate today.")
        return

    odds_df = odds_to_dataframe(raw_games)
    best_odds = get_best_odds(odds_df)

    pred_rows = []
    for _, row in best_odds.iterrows():
        feat_df = build_live_feature_row(row["home_team"], row["away_team"], team_stats)
        feat_df = engineer_features(feat_df)
        prob = float(predict_proba(model, feat_df)[0])
        pred_rows.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": prob,
            "commence_time": str(row["commence_time"]),
        })

    predictions_df = pd.DataFrame(pred_rows)
    value_bets = find_value_bets(predictions_df, best_odds)

    print(f"\n── Today's Slate: {len(best_odds)} games ────────────────────")
    for _, row in best_odds.iterrows():
        match = [p for p in pred_rows if p["home_team"] == row["home_team"]]
        prob = match[0]["home_win_prob"] if match else 0.5
        print(f"  {row['home_team']} vs {row['away_team']}  |  "
              f"Model: {prob:.1%}  |  "
              f"Odds: {int(row['best_home_price'])} / {int(row['best_away_price'])}")

    if value_bets:
        print(f"\n── Value Bets Found: {len(value_bets)} ──────────────────────")
        for bet in value_bets:
            print(
                f"  ★  {bet['recommended_bet']}\n"
                f"     Edge: {bet['edge_pct']}%  |  "
                f"Model: {bet['model_probability']:.1%}  |  "
                f"Implied: {bet['implied_probability']:.1%}  |  "
                f"Odds: {bet['american_odds']:+d}  |  "
                f"Kelly: ${bet['kelly_bet_amount']}"
            )
    else:
        print("\n  No value bets found today (edge < threshold).")
    print()


def cmd_serve():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


COMMANDS = {
    "train": cmd_train,
    "backtest": cmd_backtest,
    "predict": cmd_predict,
    "serve": cmd_serve,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
