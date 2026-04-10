"""
data.py — Data ingestion layer

Responsibilities:
  • Fetch live odds from The Odds API
  • Generate / load historical game data (synthetic for bootstrapping;
    plug in a real data source — e.g. NHL stats API — when available)
  • Persist raw data to disk as Parquet files
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger

from config import settings

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# NHL / NBA team rosters (used for synthetic data generation)
# ──────────────────────────────────────────────────────────────────────────────

NHL_TEAMS = [
    "Boston Bruins", "Buffalo Sabres", "Detroit Red Wings", "Florida Panthers",
    "Montreal Canadiens", "Ottawa Senators", "Tampa Bay Lightning", "Toronto Maple Leafs",
    "Carolina Hurricanes", "Columbus Blue Jackets", "New Jersey Devils", "New York Islanders",
    "New York Rangers", "Philadelphia Flyers", "Pittsburgh Penguins", "Washington Capitals",
    "Arizona Coyotes", "Chicago Blackhawks", "Colorado Avalanche", "Dallas Stars",
    "Minnesota Wild", "Nashville Predators", "St. Louis Blues", "Winnipeg Jets",
    "Anaheim Ducks", "Calgary Flames", "Edmonton Oilers", "Los Angeles Kings",
    "San Jose Sharks", "Seattle Kraken", "Vancouver Canucks", "Vegas Golden Knights",
]

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]


# ──────────────────────────────────────────────────────────────────────────────
# The Odds API helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_live_odds(sport: str | None = None) -> list[dict[str, Any]]:
    """
    Pull current odds from The Odds API for head-to-head markets.

    Returns a list of raw game dicts as returned by the API.
    Raises on non-200 responses so callers can handle gracefully.
    """
    sport = sport or settings.sport
    url = f"{settings.odds_api_base_url}/sports/{sport}/odds"
    params = {
        "apiKey": settings.odds_api_key,
        "regions": settings.regions,
        "markets": settings.markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    logger.info(f"Fetching live odds for {sport} …")
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    games = resp.json()
    logger.success(f"Received {len(games)} game(s) from The Odds API.")

    # Persist raw response
    raw_path = DATA_DIR / f"raw_odds_{sport}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    raw_path.write_text(json.dumps(games, indent=2))
    logger.debug(f"Raw odds saved → {raw_path}")

    return games


def odds_to_dataframe(raw_games: list[dict]) -> pd.DataFrame:
    """
    Flatten the nested Odds API response into a tidy DataFrame.

    Columns produced:
        game_id, sport, commence_time, home_team, away_team,
        bookmaker, home_price, away_price, last_update
    """
    rows: list[dict] = []
    for game in raw_games:
        for bm in game.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
                rows.append(
                    {
                        "game_id": game["id"],
                        "sport": game["sport_key"],
                        "commence_time": game["commence_time"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "bookmaker": bm["key"],
                        "home_price": prices.get(game["home_team"]),
                        "away_price": prices.get(game["away_team"]),
                        "last_update": bm["last_update"],
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["last_update"] = pd.to_datetime(df["last_update"])
    df["home_price"] = pd.to_numeric(df["home_price"], errors="coerce")
    df["away_price"] = pd.to_numeric(df["away_price"], errors="coerce")
    df.dropna(subset=["home_price", "away_price"], inplace=True)

    logger.info(f"Odds DataFrame: {len(df)} rows, {df['game_id'].nunique()} unique games.")
    return df


def get_best_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce multiple bookmakers to the best (highest) available price per side.
    Returns one row per game with the sharpest line.
    """
    if odds_df.empty:
        return odds_df

    best = (
        odds_df.groupby("game_id")
        .agg(
            sport=("sport", "first"),
            commence_time=("commence_time", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            best_home_price=("home_price", "max"),
            best_away_price=("away_price", "max"),
        )
        .reset_index()
    )
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Historical data  (synthetic bootstrap — replace with real NHL stats API)
# ──────────────────────────────────────────────────────────────────────────────

def _team_strength(team: str) -> float:
    """Deterministic pseudo-strength so simulated results are consistent."""
    rng = random.Random(sum(ord(c) for c in team))
    return rng.uniform(0.35, 0.65)


def generate_historical_data(
    n_games: int = 2000,
    sport: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesise a historical game log with realistic-ish stats.

    Each row = one completed game with:
        date, home_team, away_team,
        home_goals, away_goals, home_win (target),
        home_rest_days, away_rest_days,
        home_win_pct_season, away_win_pct_season,
        home_goals_for_avg, away_goals_for_avg,
        home_goals_against_avg, away_goals_against_avg,
        home_win_pct_L5, away_win_pct_L5,
        home_h2h_win_pct, away_h2h_win_pct
    """
    sport = sport or settings.sport
    teams = NHL_TEAMS if "hockey" in sport else NBA_TEAMS
    is_nhl = "hockey" in sport

    rng = np.random.default_rng(seed)
    start_date = datetime(2022, 10, 1)

    records: list[dict] = []
    # Running season stats per team
    season_stats: dict[str, dict] = {
        t: {
            "games": 0, "wins": 0,
            "gf": [], "ga": [],
            "last_game": None,
            "last5": [],
            "h2h": {},  # opponent -> [win/loss list]
        }
        for t in teams
    }

    current_date = start_date
    game_idx = 0

    while game_idx < n_games:
        # Pick two random teams
        home, away = rng.choice(teams, size=2, replace=False)
        sh = season_stats[home]
        sa = season_stats[away]

        # Compute rest days
        home_rest = (current_date - sh["last_game"]).days if sh["last_game"] else 3
        away_rest = (current_date - sa["last_game"]).days if sa["last_game"] else 3
        home_rest = min(home_rest, 14)
        away_rest = min(away_rest, 14)

        # Win probabilities driven by strength + home-ice / home-court advantage
        h_str = _team_strength(home) + 0.04  # home advantage
        a_str = _team_strength(away)
        h_prob = h_str / (h_str + a_str)
        h_prob = float(np.clip(h_prob + rng.normal(0, 0.05), 0.15, 0.85))

        home_win = int(rng.random() < h_prob)

        # Scores
        if is_nhl:
            home_goals = int(rng.poisson(2.9 + (h_prob - 0.5)))
            away_goals = int(rng.poisson(2.9 - (h_prob - 0.5)))
            if home_goals == away_goals:
                home_goals += int(rng.random() > 0.5)
        else:
            home_goals = int(rng.normal(112 + (h_prob - 0.5) * 10, 8))
            away_goals = int(rng.normal(108 - (h_prob - 0.5) * 10, 8))

        home_win = int(home_goals > away_goals)

        # Season win %
        h_win_pct = sh["wins"] / sh["games"] if sh["games"] > 0 else 0.5
        a_win_pct = sa["wins"] / sa["games"] if sa["games"] > 0 else 0.5

        # Last-5 win %
        h_l5 = np.mean(sh["last5"][-5:]) if sh["last5"] else 0.5
        a_l5 = np.mean(sa["last5"][-5:]) if sa["last5"] else 0.5

        # Goals averages
        h_gf_avg = np.mean(sh["gf"][-20:]) if sh["gf"] else (2.9 if is_nhl else 110)
        h_ga_avg = np.mean(sh["ga"][-20:]) if sh["ga"] else (2.9 if is_nhl else 110)
        a_gf_avg = np.mean(sa["gf"][-20:]) if sa["gf"] else (2.9 if is_nhl else 110)
        a_ga_avg = np.mean(sa["ga"][-20:]) if sa["ga"] else (2.9 if is_nhl else 110)

        # H2H
        h2h_key = away
        h_h2h_results = sh["h2h"].get(h2h_key, [])
        a_h2h_results = sa["h2h"].get(home, [])
        h_h2h_win_pct = np.mean(h_h2h_results[-10:]) if h_h2h_results else 0.5
        a_h2h_win_pct = np.mean(a_h2h_results[-10:]) if a_h2h_results else 0.5

        records.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_score": home_goals,
                "away_score": away_goals,
                "home_win": home_win,
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "home_win_pct_season": round(h_win_pct, 4),
                "away_win_pct_season": round(a_win_pct, 4),
                "home_goals_for_avg": round(h_gf_avg, 3),
                "home_goals_against_avg": round(h_ga_avg, 3),
                "away_goals_for_avg": round(a_gf_avg, 3),
                "away_goals_against_avg": round(a_ga_avg, 3),
                "home_win_pct_L5": round(h_l5, 4),
                "away_win_pct_L5": round(a_l5, 4),
                "home_h2h_win_pct": round(h_h2h_win_pct, 4),
                "away_h2h_win_pct": round(a_h2h_win_pct, 4),
            }
        )

        # Update running stats
        for stat, team, scored, conceded, won in [
            (sh, home, home_goals, away_goals, home_win),
            (sa, away, away_goals, home_goals, 1 - home_win),
        ]:
            stat["games"] += 1
            stat["wins"] += won
            stat["gf"].append(scored)
            stat["ga"].append(conceded)
            stat["last5"].append(won)
            stat["last_game"] = current_date

        sh["h2h"].setdefault(away, []).append(home_win)
        sa["h2h"].setdefault(home, []).append(1 - home_win)

        current_date += timedelta(days=int(rng.integers(0, 2)))
        game_idx += 1

    df = pd.DataFrame(records)
    path = DATA_DIR / f"historical_{sport}.parquet"
    df.to_parquet(path, index=False)
    logger.success(f"Generated {len(df)} historical games → {path}")
    return df


def load_historical_data(sport: str | None = None) -> pd.DataFrame:
    """Load from disk; generate if missing."""
    sport = sport or settings.sport
    path = DATA_DIR / f"historical_{sport}.parquet"
    if path.exists():
        logger.info(f"Loading historical data from {path}")
        return pd.read_parquet(path)
    logger.warning("No historical data found — generating synthetic data …")
    return generate_historical_data(sport=sport)
