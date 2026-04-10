# AI Sports Betting Engine 🏒

A production-ready ML sports betting system that pulls live odds, trains an XGBoost model to predict win probabilities, identifies positive-expected-value bets, and exposes everything via a FastAPI backend.

---

## Architecture

```
sports_betting/
├── config.py        # Pydantic settings loaded from .env
├── data.py          # Odds API ingestion + historical data layer
├── features.py      # Feature engineering pipeline (24 features)
├── model.py         # XGBoost training, evaluation, backtesting
├── odds.py          # Odds conversions, edge calc, Kelly criterion
├── api.py           # FastAPI backend (6 endpoints)
├── main.py          # CLI entrypoint (train / backtest / predict / serve)
├── tests/
│   └── test_core.py # Unit tests (odds, features, data)
├── models/          # Persisted model + metrics JSON (auto-created)
├── data/            # Parquet files + raw API responses (auto-created)
├── requirements.txt
└── .env
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate   # recommended
pip install -r requirements.txt
```

### 2. Configure your API key

Edit `.env` (already pre-filled with your key):

```ini
ODDS_API_KEY=5056aa7c9d19dccaec1d6930bf797af8
SPORT=icehockey_nhl          # or basketball_nba
BANKROLL=1000.0
MIN_EDGE=0.05                # 5% minimum edge to flag a bet
KELLY_FRACTION=0.25          # quarter-Kelly for variance reduction
```

### 3. Train the model

```bash
python main.py train
```

Output:
```
── Model Metrics ──────────────────────────────
  accuracy: 0.6125
  log_loss: 0.6701
  roc_auc: 0.6432
  brier_score: 0.2381
  cv_log_loss_mean: 0.6798
  train_rows: 1600
  test_rows: 400
  n_features: 24
  trained_at: 2025-01-10T14:32:07
───────────────────────────────────────────────
```

### 4. Run backtest

```bash
python main.py backtest
```

### 5. Fetch live value bets (console)

```bash
python main.py predict
```

### 6. Start the API server

```bash
python main.py serve
# or directly:
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API docs auto-generated at: **http://localhost:8000/docs**

---

## API Reference

### `GET /predictions`

Returns today's value bets (live odds + ML model).

**Query params:**
| Param | Default | Description |
|-------|---------|-------------|
| `sport` | `icehockey_nhl` | Sport key from The Odds API |
| `min_edge` | `0.05` | Minimum edge threshold |
| `bankroll` | `1000.0` | Bankroll for stake sizing |

**Example response:**
```json
{
  "timestamp": "2025-01-10T19:45:00Z",
  "sport": "icehockey_nhl",
  "total_games_today": 8,
  "value_bets_found": 2,
  "min_edge_threshold": 0.05,
  "bankroll": 1000.0,
  "value_bets": [
    {
      "game_id": "a1b2c3d4e5f6",
      "commence_time": "2025-01-10T23:00:00+00:00",
      "home_team": "Colorado Avalanche",
      "away_team": "Vegas Golden Knights",
      "recommended_bet": "Colorado Avalanche ML",
      "bet_side": "home",
      "bet_team": "Colorado Avalanche",
      "model_probability": 0.6231,
      "implied_probability": 0.5556,
      "edge": 0.0675,
      "edge_pct": 6.75,
      "american_odds": -125,
      "decimal_odds": 1.8,
      "overround": 1.0435,
      "flat_bet_amount": 20.0,
      "kelly_bet_amount": 14.5
    }
  ]
}
```

---

### `GET /model-stats`

Returns training metrics and optionally a backtest.

**Query params:**
| Param | Default | Description |
|-------|---------|-------------|
| `run_backtest` | `false` | Re-run backtest (adds ~2s) |

**Example response:**
```json
{
  "accuracy": 0.6125,
  "log_loss": 0.6701,
  "roc_auc": 0.6432,
  "brier_score": 0.2381,
  "cv_log_loss_mean": 0.6798,
  "train_rows": 1600,
  "test_rows": 400,
  "n_features": 24,
  "trained_at": "2025-01-10T14:32:07",
  "backtest": {
    "total_bets": 67,
    "wins": 39,
    "losses": 28,
    "win_rate": 0.582,
    "total_wagered": 6700.0,
    "total_profit": 423.50,
    "roi": 0.0632,
    "roi_pct": 6.32,
    "profit_per_bet": 6.32,
    "avg_edge": 0.0812
  },
  "feature_importances": {
    "strength_diff": 0.1832,
    "win_pct_diff_season": 0.1541,
    "goal_diff_net": 0.1203,
    ...
  }
}
```

---

### `GET /odds`

Raw best-available odds with vig-removed fair probabilities.

```json
{
  "sport": "icehockey_nhl",
  "count": 8,
  "games": [
    {
      "game_id": "a1b2c3d4",
      "commence_time": "2025-01-10T23:00:00+00:00",
      "home_team": "Colorado Avalanche",
      "away_team": "Vegas Golden Knights",
      "best_home_price": -125,
      "best_away_price": 105,
      "fair_home_prob": 0.5412,
      "fair_away_prob": 0.4588,
      "overround": 1.0432
    }
  ]
}
```

---

### `GET /teams/{team_name}/stats`

Per-team rolling stats derived from historical data.

```
GET /teams/Colorado%20Avalanche/stats
```

```json
{
  "teams": {
    "Colorado Avalanche": {
      "win_pct_season": 0.6125,
      "goals_for_avg": 3.41,
      "goals_against_avg": 2.87,
      "win_pct_L5": 0.6,
      "rest_days": 2,
      "games_played": 40
    }
  }
}
```

---

### `POST /simulate`

Ad-hoc edge analysis for any matchup + odds.

**Request body:**
```json
{
  "home_team": "Colorado Avalanche",
  "away_team": "Vegas Golden Knights",
  "home_american_odds": -130,
  "away_american_odds": 110,
  "bankroll": 1000.0
}
```

**Response:**
```json
{
  "home_team": "Colorado Avalanche",
  "away_team": "Vegas Golden Knights",
  "model_home_prob": 0.6231,
  "model_away_prob": 0.3769,
  "fair_home_prob": 0.5641,
  "fair_away_prob": 0.4359,
  "home_edge": 0.0590,
  "away_edge": -0.0590,
  "home_edge_pct": 5.90,
  "away_edge_pct": -5.90,
  "recommended_side": "home",
  "flat_bet": 20.0,
  "kelly_bet": 12.25,
  "overround": 1.0431
}
```

---

### `GET /bankroll`

Stake sizing info for your current bankroll.

---

### `POST /train`

Retrain the model (re-generates synthetic data by default).

```
POST /train?n_games=3000
```

---

## Feature Engineering (24 features)

| Feature | Description |
|---------|-------------|
| `home_win_pct_season` | Season win % (home team) |
| `away_win_pct_season` | Season win % (away team) |
| `home/away_goals_for_avg` | Rolling 20-game goals scored avg |
| `home/away_goals_against_avg` | Rolling 20-game goals conceded avg |
| `home/away_win_pct_L5` | Last-5-game win % |
| `home/away_rest_days` | Days since last game |
| `home/away_h2h_win_pct` | Head-to-head win % (last 10 meetings) |
| `win_pct_diff_season` | Season win % differential |
| `win_pct_diff_L5` | Last-5 form differential |
| `goals_for_diff` | Scoring differential |
| `goals_against_diff` | Defensive differential |
| `goal_diff_net` | Net goal differential delta |
| `rest_diff` | Rest days differential |
| `h2h_win_pct_diff` | H2H differential |
| `home/away_strength_score` | Composite blended strength (35% season + 30% L5 + 20% GF + 15% GA) |
| `strength_diff` | Composite strength differential |
| `home/away_back2back` | Back-to-back game flag (rest ≤ 1 day) |

---

## Model Details

- **Algorithm:** XGBoost + Isotonic calibration (CalibratedClassifierCV)
- **Split:** Time-ordered 80/20 (no data leakage)
- **Validation:** 5-fold stratified CV on training set
- **Metrics:** Accuracy, Log-Loss, AUC-ROC, Brier Score
- **Hyperparameters:** `n_estimators=400`, `max_depth=4`, `lr=0.05`, `subsample=0.8`

---

## Betting Logic

```
Edge = Model Probability − Fair Implied Probability (vig-removed)

Flag bet if Edge ≥ 5%

Flat bet  = bankroll × 2%
Kelly bet = bankroll × min(quarter-Kelly, 10%)
```

Vig removal uses the **multiplicative method** — the industry standard for head-to-head markets.

---

## Data Sources

| Layer | Source |
|-------|--------|
| Live odds | [The Odds API](https://the-odds-api.com) (h2h, all US books) |
| Historical (bootstrap) | Synthetic generator with realistic distributions |
| Historical (production) | Swap `data.py` → NHL Stats API / Sports Reference |

To plug in real NHL historical data, replace `generate_historical_data()` in `data.py` with a call to:
- `https://statsapi.web.nhl.com/api/v1/schedule` for game logs
- `https://www.hockey-reference.com` (scraping / CSV download)

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Production Deployment Notes

1. **Real data**: Replace `generate_historical_data()` with a live NHL/NBA stats feed
2. **Scheduled retraining**: Use `schedule` or a cron job to retrain weekly
3. **Database**: Swap Parquet files for PostgreSQL / SQLite via SQLAlchemy
4. **Auth**: Add API key middleware to FastAPI before exposing publicly
5. **Tighten CORS**: Replace `allow_origins=["*"]` with your frontend domain
6. **Model versioning**: Add MLflow or a simple timestamp-based versioning layer

---

## Disclaimer

This system is for **educational and research purposes only**. Sports betting involves financial risk. Past model performance does not guarantee future results. Always gamble responsibly and within legal jurisdictions.
