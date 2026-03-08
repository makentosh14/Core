#!/usr/bin/env python3
"""
config.py - Unified Configuration
===================================
Merged: Research Scanner config + Trading Bot config.

SECTIONS:
  1.  API / Credentials
  2.  Timeframe configuration       [scanner — do not change]
  3.  Warmup candles                [scanner — do not change]
  4.  Indicator parameters          [scanner]
  5.  V3 signal thresholds          [scanner]
  6.  Outcome labeling              [scanner]
  7.  Market regime classification  [scanner]
  8.  Scanner collection settings   [scanner]
  9.  Data storage paths            [scanner]
  10. Logging
  11. Trading mode and risk         [bot]
  12. Leverage and margin           [bot]
  13. Candle intervals              [bot]
  14. Signal score thresholds       [bot]
  15. Feature flags                 [bot]
  16. Scan speed control            [bot]
  17. Market types                  [bot]
  18. Smart systems                 [bot]
  19. Re-entry system               [bot]
  20. Altseason mode                [bot]
  21. Position limits               [bot]
  22. DCA settings                  [bot]
  23. Fast drop protection          [bot]
  24. Strategy activation and weights [bot]
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 1. API / CREDENTIALS
# ============================================================
# Bybit REST base URL — used by both scanner and bot
BYBIT_API     = "https://api.bybit.com"
BYBIT_API_URL = BYBIT_API          # alias used by bot code

# Auth keys — loaded from .env, never hard-coded here
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Telegram — loaded from .env
TELEGRAM_BOT_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID           = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_ASSISTANT_CHAT_ID = os.getenv("TELEGRAM_ASSISTANT_CHAT_ID", "")

# API behaviour (scanner)
RATE_DELAY  = 0.12   # seconds between API calls
API_TIMEOUT = 60     # total request timeout (seconds)
MAX_RETRIES = 3
BATCH_SIZE  = 10     # parallel symbol fetches


# ============================================================
# 2. TIMEFRAME CONFIGURATION  [scanner — do not change]
# ============================================================
TIMEFRAMES = {
    "5":   {"label": "5m",  "candles": 15, "interval_sec": 300},
    "15":  {"label": "15m", "candles": 12, "interval_sec": 900},
    "60":  {"label": "1h",  "candles": 7,  "interval_sec": 3600},
    "240": {"label": "4h",  "candles": 5,  "interval_sec": 14400},
}


# ============================================================
# 3. WARMUP CANDLES  [scanner — do not change]
# ============================================================
# Extra candles fetched beyond the analysis window so slow
# indicators are fully warmed up before the analysis window.
# Must match scanner_main.py validation exactly.
#
# Slowest indicators:
#   EMA200 = 200    ADX14 = 28    Ichimoku = 52
#   MACD   = 35     BB20  = 20    Stoch14  = 14
WARMUP_CANDLES = 220


# ============================================================
# 4. INDICATOR PARAMETERS  [scanner]
# ============================================================
RSI_PERIOD        = 14
EMA_FAST          = 9
EMA_MID           = 21
EMA_SLOW          = 55
EMA_200           = 200
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_PERIOD         = 20
BB_STD            = 2.0
SUPERTREND_PERIOD = 10
SUPERTREND_MULT   = 3.0
STOCH_RSI_PERIOD  = 14
STOCH_K           = 3
STOCH_D           = 3
ATR_PERIOD        = 14
VOL_MA_PERIOD     = 20
ADX_PERIOD        = 14
CCI_PERIOD        = 20
WILLIAMS_R_PERIOD = 14
KC_PERIOD         = 20
KC_MULT           = 1.5
ICHIMOKU_TENKAN   = 9
ICHIMOKU_KIJUN    = 26
ICHIMOKU_SENKOU_B = 52
PSAR_AF_START     = 0.02
PSAR_AF_MAX       = 0.2
MFI_PERIOD        = 14
CMF_PERIOD        = 20
OBV_MA_PERIOD     = 20
ROC_PERIOD        = 12
MOMENTUM_PERIOD   = 10


# ============================================================
# 5. V3 SIGNAL THRESHOLDS  [scanner]
# ============================================================
V3_COOLDOWN  = 12
V3_FLIP_MULT = 2.5


# ============================================================
# 6. OUTCOME LABELING  [scanner]
# ============================================================
# Bars to check after a signal (5m TF = 5/15/30/40/60 min)
OUTCOME_BARS      = [1, 3, 6, 8, 12]
OUTCOME_UP_PCTS   = [2.0, 3.0, 5.0, 10.0]
OUTCOME_DOWN_PCTS = [2.0, 3.0, 5.0, 10.0]
OUTCOME_SIG_MOVE  = 2.0   # min % to classify as significant move


# ============================================================
# 7. MARKET REGIME CLASSIFICATION  [scanner]
# ============================================================
REGIME_TRENDING_ADX = 25.0   # ADX above this = trending
REGIME_STRONG_ADX   = 40.0   # ADX above this = strongly trending
REGIME_HIGH_VOL_ATR = 0.03   # ATR/price > 3%   = high volatility
REGIME_LOW_VOL_ATR  = 0.008  # ATR/price < 0.8% = low volatility
REGIME_SQUEEZE_BW   = 3.0    # BB bandwidth % below this = squeeze


# ============================================================
# 8. SCANNER COLLECTION SETTINGS  [scanner]
# ============================================================
TOP_SYMBOLS         = 100
SCAN_INTERVAL       = 300         # seconds between scans (5 min)
MIN_24H_VOLUME_USDT = 5_000_000


# ============================================================
# 9. DATA STORAGE PATHS  [scanner]
# ============================================================
DATA_DIR         = "scanner_data"

RESEARCH_PARQUET = "research_events.parquet"
RESEARCH_CSV     = "research_events.csv"

OUTCOME_PARQUET  = "outcome_labels.parquet"
OUTCOME_CSV      = "outcome_labels.csv"

PATTERN_JSON     = "pattern_metadata.json"

# Legacy (kept for compatibility)
SNAPSHOT_CSV = "scanner_snapshots.csv"
SIGNAL_LOG   = "signal_log.csv"
PATTERN_DB   = "pattern_database.json"
SUMMARY_JSON = "scanner_summary.json"


# ============================================================
# 10. LOGGING
# ============================================================
LOG_LEVEL      = "INFO"
LOG_FILE       = "scanner.log"
CONSOLE_OUTPUT = True


# ============================================================
# 11. TRADING MODE AND RISK  [bot]
# ============================================================
TRADING_MODE   = "auto"   # "auto" or "signal"
RISK_SPOT      = 0.03     # 3% risk per spot trade
RISK_FUTURES   = 0.09     # 9% risk per futures trade
DAILY_MAX_LOSS = 0.1      # 10% max daily loss before auto-pause


# ============================================================
# 12. LEVERAGE AND MARGIN  [bot]
# ============================================================
DEFAULT_LEVERAGE = 5
MARGIN_MODE      = "CROSSED"   # or "ISOLATED"


# ============================================================
# 13. CANDLE INTERVALS  [bot]
# ============================================================
DEFAULT_INTERVAL    = "1"
SUPPORTED_INTERVALS = ["1", "3", "5", "15"]


# ============================================================
# 14. SIGNAL SCORE THRESHOLDS  [bot]
# ============================================================
MIN_SCALP_SCORE       = 9.0
MIN_INTRADAY_SCORE    = 11.0
MIN_SWING_SCORE       = 14.0
ALTSEASON_SCORE_BOOST = 0.5
MEME_SCORE_BOOST      = 0.5
ALWAYS_ALLOW_SWING    = False   # set True only to allow low-score swing trades


# ============================================================
# 15. FEATURE FLAGS  [bot]
# ============================================================
ENABLE_MEME_RADAR    = True
ENABLE_ALTS_SCALING  = True


# ============================================================
# 16. SCAN SPEED CONTROL  [bot]
# ============================================================
BASE_SCAN_INTERVAL      = 180   # seconds (3 min default)
ALTSEASON_SCAN_INTERVAL = 120
MEME_HYPE_SCAN_INTERVAL = 90


# ============================================================
# 17. MARKET TYPES  [bot]
# ============================================================
ENABLE_SPOT    = False   # spot trading disabled
ENABLE_FUTURES = True


# ============================================================
# 18. SMART SYSTEMS  [bot]
# ============================================================
ENABLE_SMART_EXIT            = True
ENABLE_AI_SIGNAL_MEMORY      = True
ENABLE_LIQUIDITY_TRAP_FILTER = True
ENABLE_WHALE_WATCH           = True
ENABLE_NEWS_REACTION         = True


# ============================================================
# 19. RE-ENTRY SYSTEM  [bot]
# ============================================================
ENABLE_AUTO_REENTRY = False   # master switch

# Cooldowns per trade type (cycles; 1 cycle ~= 5 seconds)
REENTRY_COOLDOWNS = {
    "Scalp": {
        "base_cooldown":    6,     # 30 s base
        "max_cooldown":     60,    # 5 min max
        "min_score":        7.5,
        "confidence_boost": 1.1,
    },
    "Intraday": {
        "base_cooldown":    12,    # 1 min base
        "max_cooldown":     180,   # 15 min max
        "min_score":        8.5,
        "confidence_boost": 1.1,
    },
    "Swing": {
        "base_cooldown":    24,    # 2 min base
        "max_cooldown":     360,   # 30 min max
        "min_score":        10.0,
        "confidence_boost": 1.1,
    },
}

REENTRY_EXIT_MULTIPLIERS = {
    "TP_Hit":       0.5,   # target hit  → fastest reentry
    "Trailing_SL":  0.7,
    "Breakeven_SL": 1.0,
    "SL_Hit":       2.0,   # stop loss   → slowest reentry
    "Time_Exit":    1.5,
    "Score_Exit":   1.8,
    "Manual":       1.0,
}

MIN_REENTRY_WIN_RATE         = 0.3   # 30% min historical win rate
MAX_DAILY_REENTRIES          = 3     # per symbol per day
MIN_PROFIT_FOR_QUICK_REENTRY = 2.0  # 2% profit enables faster reentry

REENTRY_CONFIRMATION_REQUIRED = 3
REENTRY_TECH_WEIGHTS = {
    "rsi":        0.20,
    "macd":       0.25,
    "bollinger":  0.20,
    "supertrend": 0.20,
    "volume":     0.15,
}

REENTRY_FEATURES = {
    "check_whale_activity":    True,
    "check_momentum":          True,
    "check_patterns":          True,
    "check_volume_profile":    True,
    "require_trend_alignment": True,
}


# ============================================================
# 20. ALTSEASON MODE  [bot]
# ============================================================
ALTSEASON_MODE = {
    "enabled":                   True,
    "min_volume_override":       True,   # allow lower-volume alts
    "max_positions":             15,
    "prefer_longs":              True,
    "tp_multiplier":             1.5,
    "sl_multiplier":             1.2,
    "confidence_reduction":      10,     # lower confidence requirement by 10%
    "score_threshold_reduction": 1.0,
    "risk_multiplier":           1.3,
    "scan_speed":                3,      # seconds
    "enable_micro_caps":         True,
    "momentum_bias":             0.3,
}


# ============================================================
# 21. POSITION LIMITS  [bot]
# ============================================================
NORMAL_MAX_POSITIONS = 10


# ============================================================
# 22. DCA (DOLLAR COST AVERAGING)  [bot]
# ============================================================
ENABLE_DCA                  = True
DCA_MAX_RISK_MULTIPLIER     = 2.0    # max total risk after all DCAs
DCA_COOLDOWN_MINUTES        = 5      # min minutes between DCA additions
DCA_MAX_COUNT_PER_TRADE     = 2      # max DCAs per trade
DCA_MAX_POSITION_MULTIPLIER = 2.0    # position can't grow more than 2x
DCA_DAILY_LIMIT             = 8      # max DCA operations per day
DCA_MAX_BALANCE_USAGE       = 0.25   # max fraction of balance for DCA
DCA_FAST_BUFFER             = 0.05


# ============================================================
# 23. FAST DROP PROTECTION  [bot]
# ============================================================
ENABLE_FAST_DROP_PROTECTION = True

FAST_DROP_PROTECTION = {
    "enabled":            True,
    "velocity_threshold": 0.3,    # % per minute to trigger
    "pause_duration":     120,    # seconds to pause SL after fast drop
    "dca_buffer":         0.1,    # % buffer before DCA trigger
    "enhanced_buffer":    0.001,  # extra buffer during fast drops
    "min_data_points":    5,
    "monitoring_window":  10,     # last N minutes of price data
}


# ============================================================
# 24. STRATEGY ACTIVATION AND WEIGHTS  [bot]
# ============================================================
ENABLE_CORE_STRATEGY   = True
ENABLE_MEAN_REVERSION  = True
ENABLE_BREAKOUT_SNIPER = True
ENABLE_RANGE_BREAK     = True
ENABLE_SWING_STRATEGY  = True

STRATEGY_RISK_WEIGHTS = {
    "core_strategy":   1.0,   # full risk
    "mean_reversion":  0.85,
    "breakout_sniper": 0.7,
    "swing":           0.8,
    "range_break":     0.9,
}

STRATEGY_MIN_SCORES = {
    "core_strategy":   7,
    "mean_reversion":  4,
    "breakout_sniper": 4,
    "swing":           8,
    "range_break":     6,
}

STRATEGY_REGIME_PREFERENCES = {
    "core_strategy":   ["trending", "volatile"],
    "mean_reversion":  ["ranging", "consolidating"],
    "breakout_sniper": ["volatile", "trending"],
    "swing":           ["trending", "ranging"],
    "range_break":     ["ranging", "consolidating"],
}
