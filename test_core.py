"""
tests/test_core.py — Core unit tests

Run with: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── odds.py ────────────────────────────────────────────────────────────────

from odds import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_edge,
    flat_bet,
    kelly_bet,
    overround,
    remove_vig,
)


class TestOddsConversions:
    def test_positive_american_to_decimal(self):
        assert american_to_decimal(100) == pytest.approx(2.0)
        assert american_to_decimal(200) == pytest.approx(3.0)
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_american_to_decimal(self):
        assert american_to_decimal(-110) == pytest.approx(1.909, abs=0.001)
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_implied_prob_positive(self):
        assert american_to_implied_prob(100) == pytest.approx(0.5)
        assert american_to_implied_prob(200) == pytest.approx(1 / 3, abs=0.001)

    def test_implied_prob_negative(self):
        prob = american_to_implied_prob(-110)
        assert 0.52 < prob < 0.53

    def test_remove_vig_sums_to_one(self):
        h, a = remove_vig(-110, -110)
        assert h + a == pytest.approx(1.0, abs=1e-6)
        assert h == pytest.approx(0.5, abs=0.01)

    def test_overround_above_one(self):
        assert overround(-110, -110) > 1.0

    def test_fair_market_overround(self):
        # Even-money market with no vig
        assert overround(100, 100) == pytest.approx(1.0)


class TestEdgeCalculation:
    def test_positive_edge(self):
        # Model says 60% but fair implied is 50% → edge = +10%
        edge, fair = calculate_edge(0.60, 100, 100, side="home")
        assert edge == pytest.approx(0.10, abs=0.01)
        assert fair == pytest.approx(0.50, abs=0.01)

    def test_negative_edge(self):
        edge, _ = calculate_edge(0.40, 100, 100, side="home")
        assert edge < 0


class TestBetSizing:
    def test_flat_bet(self):
        assert flat_bet(1000, pct=0.02) == pytest.approx(20.0)

    def test_kelly_positive_edge(self):
        stake = kelly_bet(0.60, 2.0, 1000, fraction=0.25)
        assert stake > 0

    def test_kelly_no_edge(self):
        # No edge → kelly = 0
        stake = kelly_bet(0.50, 2.0, 1000, fraction=0.25)
        assert stake == pytest.approx(0.0)

    def test_kelly_capped_at_10_pct(self):
        # Huge edge should still be capped at 10% of bankroll
        stake = kelly_bet(0.99, 10.0, 1000, fraction=1.0)
        assert stake <= 100.0


# ── features.py ────────────────────────────────────────────────────────────

from features import engineer_features, build_feature_matrix, ALL_FEATURE_COLS


def _make_sample_df(n=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "home_win_pct_season": rng.uniform(0.3, 0.7, n),
        "away_win_pct_season": rng.uniform(0.3, 0.7, n),
        "home_goals_for_avg": rng.uniform(2.0, 4.0, n),
        "home_goals_against_avg": rng.uniform(2.0, 4.0, n),
        "away_goals_for_avg": rng.uniform(2.0, 4.0, n),
        "away_goals_against_avg": rng.uniform(2.0, 4.0, n),
        "home_win_pct_L5": rng.uniform(0.0, 1.0, n),
        "away_win_pct_L5": rng.uniform(0.0, 1.0, n),
        "home_rest_days": rng.integers(1, 5, n).astype(float),
        "away_rest_days": rng.integers(1, 5, n).astype(float),
        "home_h2h_win_pct": rng.uniform(0.3, 0.7, n),
        "away_h2h_win_pct": rng.uniform(0.3, 0.7, n),
        "home_win": rng.integers(0, 2, n),
    })


class TestFeatureEngineering:
    def test_output_has_engineered_columns(self):
        df = _make_sample_df()
        out = engineer_features(df)
        assert "win_pct_diff_season" in out.columns
        assert "strength_diff" in out.columns
        assert "home_back2back" in out.columns

    def test_no_nan_in_features(self):
        df = _make_sample_df()
        out = engineer_features(df)
        cols = [c for c in ALL_FEATURE_COLS if c in out.columns]
        assert out[cols].isna().sum().sum() == 0

    def test_build_feature_matrix_shape(self):
        df = _make_sample_df(20)
        X, y = build_feature_matrix(df)
        assert len(X) == 20
        assert len(y) == 20
        assert X.shape[1] >= 12


# ── data.py ─────────────────────────────────────────────────────────────────

from data import generate_historical_data, odds_to_dataframe, get_best_odds


class TestDataGeneration:
    def test_generate_returns_correct_shape(self):
        df = generate_historical_data(n_games=50, seed=0)
        assert len(df) == 50
        assert "home_win" in df.columns

    def test_home_win_binary(self):
        df = generate_historical_data(n_games=100, seed=1)
        assert set(df["home_win"].unique()).issubset({0, 1})

    def test_odds_to_dataframe_empty(self):
        df = odds_to_dataframe([])
        assert df.empty

    def test_get_best_odds_single_book(self):
        raw = [
            {
                "id": "abc123",
                "sport_key": "icehockey_nhl",
                "commence_time": "2025-01-10T02:00:00Z",
                "home_team": "Boston Bruins",
                "away_team": "Toronto Maple Leafs",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "last_update": "2025-01-10T01:00:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Bruins", "price": -130},
                                    {"name": "Toronto Maple Leafs", "price": 110},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        df = odds_to_dataframe(raw)
        best = get_best_odds(df)
        assert len(best) == 1
        assert best.iloc[0]["home_team"] == "Boston Bruins"
        assert best.iloc[0]["best_home_price"] == -130
