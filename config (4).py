#!/usr/bin/env python3
"""
config.py - Research Scanner Configuration
===========================================
Central configuration for all scanner parameters.
Tuned for maximum research quality, not live trading.
"""

# ============================================================
# API SETTINGS
# ============================================================
BYBIT_API         = "https://api.bybit.com"
RATE_DELAY        = 0.12        # seconds between API calls
API_TIMEOUT       = 60          # total request timeout
MAX_RETRIES       = 3
BATCH_SIZE        = 10          # parallel symbol fetches

# ============================================================
# TIMEFRAME CONFIGURATION
# ============================================================
# key = str used by Bybit API; values define candle windows
TIMEFRAMES = {
    "5":   {"label": "5m",  "candles": 15,  "interval_sec": 300},
    "15":  {"label": "15m", "candles": 12,  "interval_sec": 900},
    "60":  {"label": "1h",  "candles": 7,   "interval_sec": 3600},
    "240": {"label": "4h",  "candles": 5,   "interval_sec": 14400},
}

# ============================================================
# WARMUP CANDLES
# ============================================================
# How many extra candles to fetch beyond the analysis window.
# Must cover the slowest indicator: EMA200 needs 200 bars minimum.
# We fetch WARMUP_CANDLES + analysis_window for every timeframe.
# Validation in process_symbol uses EXACTLY this number.
#
# Minimum required by indicator:
#   EMA200   = 200
#   ADX14    = 28  (2 * period)
#   Ichimoku = 52  (senkou_b lookback)
#   MACD     = 35  (26 slow + 9 signal)
#   BB20     = 20
#   Stoch14  = 14
#
# We use 220 to safely cover EMA200 + extra stability bars.
WARMUP_CANDLES = 220

# ============================================================
# INDICATOR PARAMETERS
# ============================================================
RSI_PERIOD          = 14
EMA_FAST            = 9
EMA_MID             = 21
EMA_SLOW            = 55
EMA_200             = 200
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
BB_PERIOD           = 20
BB_STD              = 2.0
SUPERTREND_PERIOD   = 10
SUPERTREND_MULT     = 3.0
STOCH_RSI_PERIOD    = 14
STOCH_K             = 3
STOCH_D             = 3
ATR_PERIOD          = 14
VOL_MA_PERIOD       = 20
ADX_PERIOD          = 14
CCI_PERIOD          = 20
WILLIAMS_R_PERIOD   = 14
KC_PERIOD           = 20
KC_MULT             = 1.5
ICHIMOKU_TENKAN     = 9
ICHIMOKU_KIJUN      = 26
ICHIMOKU_SENKOU_B   = 52
PSAR_AF_START       = 0.02
PSAR_AF_MAX         = 0.2
MFI_PERIOD          = 14
CMF_PERIOD          = 20
OBV_MA_PERIOD       = 20
ROC_PERIOD          = 12
MOMENTUM_PERIOD     = 10

# ============================================================
# V3 SIGNAL THRESHOLDS
# ============================================================
V3_COOLDOWN  = 12
V3_FLIP_MULT = 2.5

# ============================================================
# OUTCOME LABELING — what happens AFTER the signal
# ============================================================
# Bars to check for outcome labeling (on 5m TF = 5/15/30/40/60 min)
OUTCOME_BARS       = [1, 3, 6, 8, 12]

# % thresholds for labeling a move as "reached"
OUTCOME_UP_PCTS    = [2.0, 3.0, 5.0, 10.0]
OUTCOME_DOWN_PCTS  = [2.0, 3.0, 5.0, 10.0]

# Minimum % to classify as significant move
OUTCOME_SIG_MOVE   = 2.0

# ============================================================
# MARKET REGIME CLASSIFICATION
# ============================================================
# ADX thresholds
REGIME_TRENDING_ADX    = 25.0
REGIME_STRONG_ADX      = 40.0

# ATR/price ratio thresholds for volatility classification
REGIME_HIGH_VOL_ATR    = 0.03   # ATR > 3% of price = high vol
REGIME_LOW_VOL_ATR     = 0.008  # ATR < 0.8% of price = low vol

# BB squeeze: bandwidth < threshold = compression
REGIME_SQUEEZE_BW      = 3.0   # BB bandwidth % below this = squeeze

# ============================================================
# SCANNER SETTINGS
# ============================================================
TOP_SYMBOLS           = 100
SCAN_INTERVAL         = 300            # seconds (5 min)
MIN_24H_VOLUME_USDT   = 5_000_000

# ============================================================
# DATA STORAGE
# ============================================================
DATA_DIR        = "scanner_data"

# Research dataset (primary output — parquet preferred)
RESEARCH_PARQUET = "research_events.parquet"
RESEARCH_CSV     = "research_events.csv"    # fallback / optional export

# Outcome labels storage (filled in by outcome_labeler.py)
OUTCOME_PARQUET  = "outcome_labels.parquet"
OUTCOME_CSV      = "outcome_labels.csv"

# Pattern metadata
PATTERN_JSON     = "pattern_metadata.json"

# Legacy files (kept for compatibility)
SNAPSHOT_CSV     = "scanner_snapshots.csv"
SIGNAL_LOG       = "signal_log.csv"
PATTERN_DB       = "pattern_database.json"
SUMMARY_JSON     = "scanner_summary.json"

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL      = "INFO"
LOG_FILE       = "scanner.log"
CONSOLE_OUTPUT = True
