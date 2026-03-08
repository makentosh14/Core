#!/usr/bin/env python3
"""
config.py - Central Configuration for Advanced Multi-Timeframe Market Scanner
==============================================================================
All indicator parameters, thresholds, and scanner settings in one place.
Matches your existing bot's indicator parameters exactly.
"""

# ============================================================
# API & NETWORK
# ============================================================
BYBIT_API = "https://api.bybit.com"
RATE_DELAY = 0.08          # seconds between API calls
API_TIMEOUT = 30            # seconds
MAX_RETRIES = 3
BATCH_SIZE = 10             # symbols per concurrent batch

# ============================================================
# TIMEFRAME CONFIGURATION
# ============================================================
# Format: (interval_str, candle_count, label, tf_minutes)
TIMEFRAMES = {
    "5":   {"interval": "5",   "candles": 15,  "label": "5m",  "minutes": 5,   "role": "entry"},
    "15":  {"interval": "15",  "candles": 12,  "label": "15m", "minutes": 15,  "role": "confirm"},
    "60":  {"interval": "60",  "candles": 7,   "label": "1h",  "minutes": 60,  "role": "swing"},
    "240": {"interval": "240", "candles": 5,   "label": "4h",  "minutes": 240, "role": "trend"},
}

# How many extra candles to fetch for indicator warm-up
# e.g., RSI(14) needs 14 prior candles, EMA(200) needs 200 prior
WARMUP_CANDLES = 250

# ============================================================
# INDICATOR PARAMETERS (matching your bot exactly)
# ============================================================
RSI_PERIOD = 14
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 55
EMA_200 = 200

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2.0

SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

STOCH_RSI_PERIOD = 14
STOCH_K = 3
STOCH_D = 3

ATR_PERIOD = 14
VOL_MA_PERIOD = 20
ADX_PERIOD = 14
CCI_PERIOD = 20
WILLIAMS_R_PERIOD = 14

KC_PERIOD = 20
KC_MULT = 1.5

ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52

PSAR_AF_START = 0.02
PSAR_AF_MAX = 0.2

MFI_PERIOD = 14
CMF_PERIOD = 20
OBV_MA_PERIOD = 20
ROC_PERIOD = 12
MOMENTUM_PERIOD = 10

# ============================================================
# PUMP / DUMP DETECTION THRESHOLDS
# ============================================================
PUMP_PCT = 3.0              # minimum % move to count as pump
DUMP_PCT = -3.0             # minimum % move to count as dump
MOVE_WINDOW = 4             # candles to check for move
COOLDOWN_BARS = 8           # bars between events

STRONG_PUMP_PCT = 5.0
STRONG_DUMP_PCT = -5.0
STRONG_MOVE_WINDOW = 6
STRONG_COOLDOWN = 12

# ============================================================
# SCORING THRESHOLDS (v1 system)
# ============================================================
V1_PUMP_THRESH = 4.0
V1_STRONG_PUMP_THRESH = 7.0
V1_MIN_NET_GAP = 2.0
V1_COOLDOWN_BARS = 8

# ============================================================
# V3 COMBO SYSTEM THRESHOLDS
# ============================================================
V3_COOLDOWN = 12
V3_FLIP_MULT = 2.5

# ============================================================
# OUTCOME EVALUATION
# ============================================================
OUTCOME_CHECK_BARS = 8      # bars after signal to check outcome
OUTCOME_TARGET_PCT = 3.0    # % move required = "signal worked"

# ============================================================
# SCANNER SETTINGS
# ============================================================
TOP_SYMBOLS = 100            # how many symbols to scan (by 24h volume)
SCAN_INTERVAL = 300          # seconds between full scans (5 min = 300)
MIN_24H_VOLUME_USDT = 5_000_000  # skip low-volume coins

# ============================================================
# DATA STORAGE
# ============================================================
DATA_DIR = "scanner_data"
EVENTS_CSV = "scanner_events.csv"
SNAPSHOT_CSV = "scanner_snapshots.csv"
PATTERN_DB = "pattern_database.json"
SUMMARY_JSON = "scanner_summary.json"
SIGNAL_LOG = "signal_log.csv"

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"           # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "scanner.log"
CONSOLE_OUTPUT = True
