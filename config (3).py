import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_API_URL = "https://api.bybit.com"

TELEGRAM_BOT_TOKEN = "7803544014:AAGLJVwfTg4Ij5lzI8RIVRfrZkKG9uIZnh4"
TELEGRAM_CHAT_ID = "1806610681"
TELEGRAM_ASSISTANT_CHAT_ID = "-1000000000000"  # optional

# === TRADING MODE ===
TRADING_MODE = "auto"  # "auto" or "signal"
RISK_SPOT = 0.03
RISK_FUTURES = 0.09    # default 9% per futures trade
DAILY_MAX_LOSS = 0.1   # max daily loss before auto pause (10%)

# === LEVERAGE / MARGIN SETTINGS ===
DEFAULT_LEVERAGE = 5
MARGIN_MODE = "CROSSED"  # or "ISOLATED"

# === CANDLE INTERVAL SETTINGS ===
DEFAULT_INTERVAL = '1'
SUPPORTED_INTERVALS = ['1', '3', '5', '15']

# === SIGNAL THRESHOLDS ===
MIN_SCALP_SCORE = 9.5
MIN_INTRADAY_SCORE = 10.5
MIN_SWING_SCORE = 12
ALTSEASON_SCORE_BOOST = 0.5
MEME_SCORE_BOOST = 0.5
ALWAYS_ALLOW_SWING = False  # ❌ Disabled to prevent low-score trades - change back to True if needed

# === MEME / ALTSEASON DETECTION ===
ENABLE_MEME_RADAR = True
ENABLE_ALTS_SCALING = True

# === SCAN SPEED CONTROL ===
BASE_SCAN_INTERVAL = 180  # default: 3 minutes
ALTSEASON_SCAN_INTERVAL = 120
MEME_HYPE_SCAN_INTERVAL = 90

# === MARKET TYPES ===
ENABLE_SPOT = False  # 🔴 Spot trading disabled
ENABLE_FUTURES = True

# === SMART SYSTEMS ===
ENABLE_SMART_EXIT = True
ENABLE_AI_SIGNAL_MEMORY = True
ENABLE_LIQUIDITY_TRAP_FILTER = True
ENABLE_WHALE_WATCH = True
ENABLE_NEWS_REACTION = True

# Master switch for the reentry system
ENABLE_AUTO_REENTRY = False  # Set to False to disable completely

# Reentry cooldown settings (in monitoring cycles, 1 cycle = 5 seconds)
REENTRY_COOLDOWNS = {
    "Scalp": {
        "base_cooldown": 6,      # 30 seconds base cooldown
        "max_cooldown": 60,      # 5 minutes maximum
        "min_score": 7.5,        # Minimum score to consider reentry
        "confidence_boost": 1.1  # Confidence multiplier for reentries
    },
    "Intraday": {
        "base_cooldown": 12,     # 1 minute base cooldown
        "max_cooldown": 180,     # 15 minutes maximum
        "min_score": 8.5,
        "confidence_boost": 1.1
    },
    "Swing": {
        "base_cooldown": 24,     # 2 minutes base cooldown
        "max_cooldown": 360,     # 30 minutes maximum
        "min_score": 10.0,
        "confidence_boost": 1.1
    }
}

# Exit type multipliers for cooldown calculation
REENTRY_EXIT_MULTIPLIERS = {
    "TP_Hit": 0.5,          # Target hit - fastest reentry
    "Trailing_SL": 0.7,     # Trailing stop - fast reentry
    "Breakeven_SL": 1.0,    # Breakeven stop - normal
    "SL_Hit": 2.0,          # Stop loss - slow reentry
    "Time_Exit": 1.5,       # Time-based - slower
    "Score_Exit": 1.8,      # Score deterioration - slow
    "Manual": 1.0           # Manual exit - normal
}

# Performance thresholds
MIN_REENTRY_WIN_RATE = 0.3     # 30% minimum win rate to allow reentry
MAX_DAILY_REENTRIES = 3        # Maximum reentries per symbol per day
MIN_PROFIT_FOR_QUICK_REENTRY = 2.0  # 2% profit for faster reentry

# Technical confirmation requirements
REENTRY_CONFIRMATION_REQUIRED = 3  # Number of confirming indicators needed
REENTRY_TECH_WEIGHTS = {
    "rsi": 0.2,
    "macd": 0.25,
    "bollinger": 0.2,
    "supertrend": 0.2,
    "volume": 0.15
}

# Reentry feature flags
REENTRY_FEATURES = {
    "check_whale_activity": True,
    "check_momentum": True,
    "check_patterns": True,
    "check_volume_profile": True,
    "require_trend_alignment": True
}

ALTSEASON_MODE = {
    "enabled": True,  # Master switch
    "min_volume_override": True,  # Trade lower volume alts during altseason
    "max_positions": 15,  # Increase from normal 10
    "prefer_longs": True,  # Bias toward upside in altseason
    "tp_multiplier": 1.5,  # Bigger profit targets
    "sl_multiplier": 1.2,  # Wider stops for volatility
    "confidence_reduction": 10,  # Lower confidence requirements by 10%
    "score_threshold_reduction": 1.0,  # Reduce score requirements
    "risk_multiplier": 1.3,  # Increase risk per trade
    "scan_speed": 3,  # Faster scanning (seconds)
    "enable_micro_caps": True,  # Trade smaller alts
    "momentum_bias": 0.3,  # Extra score for momentum
}

NORMAL_MAX_POSITIONS = 10

# DCA (Dollar Cost Averaging) Settings
ENABLE_DCA = True  # Master switch for DCA strategy
DCA_MAX_RISK_MULTIPLIER = 2.0  # Maximum total risk after all DCAs (2x original)
DCA_COOLDOWN_MINUTES = 5  # Minimum time between DCA additions

# DCA Risk Limits
DCA_MAX_COUNT_PER_TRADE = 2      # Maximum 2 DCAs per trade
DCA_MAX_POSITION_MULTIPLIER = 2.0 # Position can't grow more than 2x
DCA_DAILY_LIMIT = 8              # Maximum 8 DCA operations per day
DCA_MAX_BALANCE_USAGE = 0.25  # Maximum DCA operations per day

ENABLE_FAST_DROP_PROTECTION = True  # Master switch for fast drop protection
FAST_DROP_PROTECTION = {
    "enabled": True,
    "velocity_threshold": 0.3,        # 0.3% per minute velocity threshold
    "pause_duration": 120,            # Pause SL for 2 minutes after fast drop
    "dca_buffer": 0.1,               # 0.1% buffer before DCA trigger
    "enhanced_buffer": 0.001,         # Extra 0.1% during fast drops
    "min_data_points": 5,            # Minimum price points for analysis
    "monitoring_window": 10,          # Monitor last 10 minutes of price data
}

DCA_FAST_BUFFER = 0.05

# === STRATEGY ACTIVATION SETTINGS ===
ENABLE_CORE_STRATEGY = True        # Your main trend-following strategy
ENABLE_MEAN_REVERSION = True       # Mean reversion strategy (for ranging markets)
ENABLE_BREAKOUT_SNIPER = True      # Breakout detection strategy (for volatile markets)
ENABLE_RANGE_BREAK = True          # Range breakout detection strategy
ENABLE_SWING_STRATEGY = True       # Swing trading strategy

# Strategy-specific risk settings
STRATEGY_RISK_WEIGHTS = {
    "core_strategy": 1.0,          # Full risk allocation
    "mean_reversion": 0.85,        # 85% of normal risk (more conservative)
    "breakout_sniper": 0.7,        # 70% of normal risk (high volatility)
    "swing": 0.8,                  # 80% of normal risk
    "range_break": 0.9             # 90% of normal risk
}

# Strategy score thresholds
STRATEGY_MIN_SCORES = {
    "core_strategy": 7,            # Your current core threshold
    "mean_reversion": 4,           # Lower threshold for mean reversion
    "breakout_sniper": 4,          # Lower threshold for breakouts
    "swing": 8,                    # Higher threshold for swing trades
    "range_break": 6               # Medium threshold for range breaks
}

# Market regime preferences for each strategy
STRATEGY_REGIME_PREFERENCES = {
    "core_strategy": ["trending", "volatile"],
    "mean_reversion": ["ranging", "consolidating"],
    "breakout_sniper": ["volatile", "trending"],
    "swing": ["trending", "ranging"],
    "range_break": ["ranging", "consolidating"]
}
