"""
odds.py — Odds utilities

Covers:
  • American ↔ Decimal ↔ Implied probability conversions
  • Vig / overround removal (to get fair implied probabilities)
  • Edge calculation: model_prob - fair_implied_prob
  • Bet sizing: flat-bet and fractional Kelly
  • Value-bet identification
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import settings


# ──────────────────────────────────────────────────────────────────────────────
# Odds conversions
# ──────────────────────────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    """Convert American moneyline odds to decimal odds."""
    if american >= 100:
        return (american / 100.0) + 1.0
    else:
        return (100.0 / abs(american)) + 1.0


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American moneyline."""
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    else:
        return -100.0 / (decimal - 1.0)


def american_to_implied_prob(american: float) -> float:
    """
    Raw (vig-inclusive) implied probability from American odds.
    This is NOT the fair probability — use remove_vig for that.
    """
    if american >= 100:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def implied_prob_to_american(prob: float) -> float:
    """Convert a probability to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {prob}")
    decimal = 1.0 / prob
    return decimal_to_american(decimal)


def remove_vig(home_price: float, away_price: float) -> tuple[float, float]:
    """
    Remove the sportsbook's vig to obtain fair (no-vig) implied probabilities.

    Uses the multiplicative method (preferred over the additive method for
    markets close to even-money).

    Returns (fair_home_prob, fair_away_prob) that sum to 1.0.
    """
    raw_home = american_to_implied_prob(home_price)
    raw_away = american_to_implied_prob(away_price)
    total = raw_home + raw_away
    if total <= 0:
        return 0.5, 0.5
    return raw_home / total, raw_away / total


def overround(home_price: float, away_price: float) -> float:
    """Sportsbook's margin (overround). A fair market = 1.0."""
    raw_home = american_to_implied_prob(home_price)
    raw_away = american_to_implied_prob(away_price)
    return raw_home + raw_away


# ──────────────────────────────────────────────────────────────────────────────
# Edge calculation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_edge(
    model_prob: float,
    home_price: float,
    away_price: float,
    side: str = "home",
) -> tuple[float, float]:
    """
    Compute edge for a specific bet side.

    edge = model_prob - fair_implied_prob

    Returns (edge, fair_implied_prob).
    """
    fair_home, fair_away = remove_vig(home_price, away_price)
    fair_prob = fair_home if side == "home" else fair_away
    model_side_prob = model_prob if side == "home" else (1.0 - model_prob)
    edge = model_side_prob - fair_prob
    return edge, fair_prob


# ──────────────────────────────────────────────────────────────────────────────
# Bet sizing
# ──────────────────────────────────────────────────────────────────────────────

def flat_bet(bankroll: float, pct: float | None = None) -> float:
    """Return a flat wager amount (default: settings.flat_bet_pct of bankroll)."""
    pct = pct if pct is not None else settings.flat_bet_pct
    return round(bankroll * pct, 2)


def kelly_bet(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float | None = None,
) -> float:
    """
    Fractional Kelly criterion bet size.

    Kelly fraction f* = (b * p - q) / b
      where b = decimal_odds - 1,  p = win prob,  q = 1 - p

    We use `fraction` (default 0.25 = quarter-Kelly) to reduce variance.
    Returns 0.0 when Kelly is negative (no edge).
    """
    fraction = fraction if fraction is not None else settings.kelly_fraction
    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p
    kelly_full = (b * p - q) / b
    if kelly_full <= 0:
        return 0.0
    kelly_frac = kelly_full * fraction
    kelly_frac = min(kelly_frac, 0.10)  # hard cap at 10 % of bankroll
    return round(bankroll * kelly_frac, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Value bet pipeline
# ──────────────────────────────────────────────────────────────────────────────

def find_value_bets(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    bankroll: float | None = None,
    min_edge: float | None = None,
) -> list[dict[str, Any]]:
    """
    Cross-reference model predictions against live odds to surface value bets.

    Parameters
    ----------
    predictions_df : DataFrame with columns
        [home_team, away_team, home_win_prob, commence_time]
    odds_df : DataFrame from data.get_best_odds(), columns
        [game_id, home_team, away_team, best_home_price, best_away_price, ...]
    bankroll : current bankroll (defaults to settings.BANKROLL)
    min_edge : minimum edge to flag a bet (defaults to settings.MIN_EDGE)

    Returns list of bet dicts ready for API serialisation.
    """
    bankroll = bankroll if bankroll is not None else settings.bankroll
    min_edge = min_edge if min_edge is not None else settings.min_edge

    value_bets: list[dict] = []

    for _, pred_row in predictions_df.iterrows():
        home = pred_row["home_team"]
        away = pred_row["away_team"]
        home_prob = float(pred_row["home_win_prob"])

        # Match to odds row
        match = odds_df[
            (odds_df["home_team"] == home) & (odds_df["away_team"] == away)
        ]
        if match.empty:
            continue

        odds_row = match.iloc[0]
        home_price = float(odds_row["best_home_price"])
        away_price = float(odds_row["best_away_price"])

        # Evaluate both sides
        for side, side_prob, price in [
            ("home", home_prob, home_price),
            ("away", 1.0 - home_prob, away_price),
        ]:
            edge, fair_prob = calculate_edge(home_prob, home_price, away_price, side)

            if edge < min_edge:
                continue

            decimal = american_to_decimal(price)
            flat = flat_bet(bankroll)
            kelly = kelly_bet(side_prob, decimal, bankroll)
            team = home if side == "home" else away

            value_bets.append(
                {
                    "game_id": str(odds_row.get("game_id", "")),
                    "commence_time": str(pred_row.get("commence_time", "")),
                    "home_team": home,
                    "away_team": away,
                    "recommended_bet": f"{team} ML",
                    "bet_side": side,
                    "bet_team": team,
                    "model_probability": round(side_prob, 4),
                    "implied_probability": round(fair_prob, 4),
                    "edge": round(edge, 4),
                    "edge_pct": round(edge * 100, 2),
                    "american_odds": int(price),
                    "decimal_odds": round(decimal, 3),
                    "overround": round(overround(home_price, away_price), 4),
                    "flat_bet_amount": flat,
                    "kelly_bet_amount": kelly,
                }
            )

    # Sort by edge descending
    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    logger.info(f"Found {len(value_bets)} value bet(s) with edge ≥ {min_edge:.0%}.")
    return value_bets


def summarise_odds_df(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich an odds DataFrame with fair probabilities and overround columns.
    Useful for display / debugging.
    """
    if odds_df.empty:
        return odds_df

    df = odds_df.copy()
    df[["fair_home_prob", "fair_away_prob"]] = df.apply(
        lambda r: pd.Series(remove_vig(r["best_home_price"], r["best_away_price"])),
        axis=1,
    )
    df["overround"] = df.apply(
        lambda r: overround(r["best_home_price"], r["best_away_price"]), axis=1
    )
    return df
