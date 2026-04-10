"""
config.py — Centralised settings loaded from .env
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API
    odds_api_key: str = Field(..., env="ODDS_API_KEY")
    odds_api_base_url: str = Field("https://api.the-odds-api.com/v4", env="ODDS_API_BASE_URL")

    # Sport / market config
    sport: str = Field("icehockey_nhl", env="SPORT")
    regions: str = Field("us", env="REGIONS")
    markets: str = Field("h2h", env="MARKETS")

    # Bankroll
    bankroll: float = Field(1000.0, env="BANKROLL")
    flat_bet_pct: float = Field(0.02, env="FLAT_BET_PCT")
    kelly_fraction: float = Field(0.25, env="KELLY_FRACTION")  # fractional Kelly

    # Model
    min_edge: float = Field(0.05, env="MIN_EDGE")
    model_path: str = Field("models/xgb_model.joblib", env="MODEL_PATH")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
