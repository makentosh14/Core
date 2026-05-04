#!/usr/bin/env python3
"""
main.py - Core Strategy Trading Bot
QUALITY PATCH (May 2026) — Win-rate optimization
====================================================
Changes vs previous version:
  - MIN_SCALP_SCORE        9.0  → 10.5
  - MIN_INTRADAY_SCORE    10.0  → 12.0
  - MIN_SWING_SCORE       14.0  → 15.5
  - Confidence early gate  60   → 68
  - Confidence per tier:   Scalp 65→70, Intraday 70→75, Swing 80→82
  - Trend strength tier:   Intraday 0.35→0.45, Swing 0.7→0.75
  - Confirmations needed:   2  → 3
  - _calculate_momentum_and_regime_bonus capped at 1.5 (was ~3.5)
"""

import asyncio
import traceback
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Core imports
from scanner import fetch_symbols
from websocket_candles import live_candles, stream_candles, SUPPORTED_INTERVALS
from score import (
    score_symbol, determine_direction, calculate_confidence,
    has_pump_potential, detect_momentum_strength
)
from telegram_bot import send_telegram_message, format_trade_signal, send_error_to_telegram
from trend_filters import monitor_btc_trend_accuracy, monitor_altseason_status, validate_short_signal
from trend_upgrade_integration import get_trend_context_cached
from signal_memory import log_signal, is_duplicate_signal
from config import DEFAULT_LEVERAGE, ALWAYS_ALLOW_SWING, ALTSEASON_MODE, NORMAL_MAX_POSITIONS
from performance_tracker import track_signal
from logger import log
from bybit_sync import sync_bot_with_bybit
from monitor_report import log_trade_result, send_daily_report
from trade_executor import calculate_dynamic_sl_tp, execute_trade_if_valid
from symbol_info import fetch_symbol_info
from activity_logger import write_log, log_trade_to_file
from monitor import (
    track_active_trade, monitor_trades, load_active_trades,
    check_and_restore_sl, active_trades, recover_active_trades_from_exchange,
    periodic_trade_sync, monitor_active_trades, get_current_price
)
from trade_lock_manager import trade_lock_manager
from scalp_hunter import (
    evaluate_scalp_setup,
    update_watchlist,
    update_btc_state,
    register_scalp_trade,
    record_scalp_result,
    format_scalp_signal,
    format_scalp_exit_message,
    SCALP_CONFIG,
    _btc_state,
)

log(f"🔍 main.py - Core Strategy Only - imported active_trades id: {id(active_trades)}")

# === CORE STRATEGY CONFIGURATION ===
TIMEFRAMES = SUPPORTED_INTERVALS

# Global state
active_signals: Dict[str, Any] = {}
recent_exits: Dict[str, float] = {}
signal_cooldown: Dict[str, float] = {}
startup_time = time.time()

# Timing constants
SIGNAL_COOLDOWN_TIME = 3600  # 1 hour cooldown after signal
EXIT_COOLDOWN = 120          # 2 minutes cooldown after exit

# ── QUALITY PATCH: tightened score thresholds ────────────────────────────────
MIN_SCALP_SCORE    = 10.5    # was 9.0   — only top scalps
MIN_INTRADAY_SCORE = 12.0    # was 10.0  — only top intraday
MIN_SWING_SCORE    = 15.5    # was 14.0  — only top swings

# QUALITY PATCH: tightened confidence/strength tiers (used in determine_core_strategy_type)
MIN_SCALP_CONFIDENCE    = 70    # was 65
MIN_INTRADAY_CONFIDENCE = 75    # was 70
MIN_SWING_CONFIDENCE    = 82    # was 80
MIN_INTRADAY_TREND_STR  = 0.45  # was 0.35
MIN_SWING_TREND_STR     = 0.75  # was 0.7

# Early confidence gate (in core_strategy_scan)
EARLY_CONFIDENCE_GATE = 68      # was 60

# Confirmations required
MIN_CONFIRMATIONS = 3           # was 2

# Core Strategy Risk Management
CORE_RISK_PERCENTAGES = {
    "Scalp":        0.025,   # 2.5% risk for scalps
    "Intraday":     0.02,    # 2% risk for intraday
    "Swing":        0.015,   # 1.5% risk for swing trades
    "CoreScalp":    0.025,
    "CoreIntraday": 0.02,
    "CoreSwing":    0.015,
}

# Core Strategy Position Limits
MAX_CORE_POSITIONS     = 3   # Maximum 3 concurrent positions
MAX_SCALP_POSITIONS    = 2   # Maximum 2 scalp positions
MAX_INTRADAY_POSITIONS = 1   # Maximum 1 intraday position
MAX_SWING_POSITIONS    = 1   # Maximum 1 swing position

# <<< SCALP HUNTER >>>
SCALP_HUNTER_ENABLED       = True
SCALP_HUNTER_SIGNALS_ONLY  = False
SCALP_RISK_PCT             = 0.01
MAX_SCALP_CONCURRENT       = 2
scalp_signal_cooldown: Dict[str, float] = {}
SCALP_COOLDOWN_SECONDS     = 300


# ─────────────────────────────────────────────────────────────────────────────
# CANDLES STRUCTURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fix_live_candles_structure(candles_data):
    """Ensures consistent format: {symbol: {timeframe: [candles]}}"""
    if not candles_data:
        return {}

    fixed = {}
    for symbol, data in candles_data.items():
        if not isinstance(data, dict):
            continue
        fixed[symbol] = {}
        for tf, candles in data.items():
            if candles is None:
                continue
            if hasattr(candles, '__iter__') and not isinstance(candles, (str, dict)):
                try:
                    candle_list = list(candles)
                    if candle_list:
                        fixed[symbol][tf] = candle_list
                except Exception:
                    continue
            elif isinstance(candles, list):
                fixed[symbol][tf] = candles
    return fixed


def safe_get_candles(symbol: str, source: dict) -> Optional[List]:
    """Safely extract candles from any timeframe"""
    try:
        if symbol not in source:
            return None
        for tf in ['1', '5', '15', '3', '30']:
            if tf in source[symbol]:
                candle_data = source[symbol][tf]
                if candle_data:
                    try:
                        candle_list = list(candle_data)
                        if len(candle_list) > 0:
                            return candle_list[-20:] if len(candle_list) >= 20 else candle_list
                    except Exception:
                        continue
        return None
    except Exception as e:
        log(f"❌ Safe candle extraction error for {symbol}: {e}", level="ERROR")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL FILTERING
# ─────────────────────────────────────────────────────────────────────────────

async def filter_core_symbols(symbols: List[str]) -> List[str]:
    """Filter symbols that have sufficient data for analysis"""
    source = fix_live_candles_structure(live_candles)
    log(f"✅ Fixed live_candles structure before filtering")

    filtered = []

    for symbol in symbols:
        try:
            if 'USDT' not in symbol:
                continue
            if symbol not in source:
                continue

            tf_count = 0
            for tf in ['1', '5', '15']:
                if tf in source[symbol] and source[symbol][tf]:
                    tf_data = source[symbol][tf]
                    if isinstance(tf_data, list) and len(tf_data) >= 12:
                        tf_count += 1
            if tf_count < 2:
                continue

            candles = None
            for tf in ['1', '5', '15']:
                if tf in source[symbol] and source[symbol][tf]:
                    tf_data = source[symbol][tf]
                    if isinstance(tf_data, list) and len(tf_data) >= 5:
                        candles = tf_data[-20:] if len(tf_data) >= 20 else tf_data
                        break

            if not candles or len(candles) < 5:
                continue

            volumes = [float(c.get('volume', 0)) for c in candles[-5:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            if avg_volume > 10000:
                filtered.append(symbol)
            if len(filtered) >= 50:
                break
        except Exception as e:
            log(f"⚠️ Error filtering {symbol}: {e}", level="WARN")
            continue

    log(f"✅ Filtered to {len(filtered)} symbols with relaxed criteria")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORE CALCULATION (QUALITY PATCH — bonus capped at 1.5)
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_core_score(symbol: str, core_candles: Dict, trend_context: Dict) -> float:
    """Calculate core score from base + small bonus"""
    try:
        base_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
            symbol, core_candles, trend_context
        )
        bonus = _calculate_momentum_and_regime_bonus(symbol, core_candles, trend_context)
        return base_score + bonus
    except Exception as e:
        log(f"❌ Error calculating score for {symbol}: {e}", level="ERROR")
        return 6.0


def _calculate_momentum_and_regime_bonus(symbol: str, core_candles: Dict, trend_context: Dict) -> float:
    """
    QUALITY PATCH: Cap free bonus at ~1.5 (was ~3.5).
    Score must come from real indicators, not regime alignment.
    """
    try:
        momentum_bonus = 0
        for tf, candles in core_candles.items():
            if candles and len(candles) >= 10:
                has_momentum, _, strength = detect_momentum_strength(candles)
                # Require STRONG momentum (was 0.6) and cap bonus
                if has_momentum and strength > 0.75:
                    momentum_bonus = min(strength * 1.0, 1.0)  # was * 1.5, no cap
                    break

        regime = trend_context.get("regime", "unknown").lower()
        if regime == "transitional":
            regime_bonus = 0.8         # was 2.0
        elif regime in ("trending", "strong_trending"):
            regime_bonus = 0.5         # was 1.5
        elif regime in ("range_bound", "consolidating"):
            regime_bonus = 0.0         # was 1.2 — don't bonus ranging markets
        else:
            regime_bonus = 0.0         # was 1.0

        return momentum_bonus + regime_bonus  # max ~1.5

    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTION & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def determine_core_direction(core_candles: Dict, trend_context: Dict) -> Optional[str]:
    """Determine direction from price action and trend context"""
    try:
        tf_scores = {}
        for tf, candles in core_candles.items():
            if not candles or len(candles) < 5:
                continue
            closes = [float(c.get('close', 0)) for c in candles[-10:]]
            if len(closes) < 5:
                continue
            avg = sum(closes[:-1]) / len(closes[:-1])
            pct_diff = (closes[-1] - avg) / avg if avg > 0 else 0
            tf_scores[tf] = pct_diff

        if not tf_scores:
            return None

        bullish = sum(1 for v in tf_scores.values() if v > 0)
        bearish = sum(1 for v in tf_scores.values() if v < 0)
        total = len(tf_scores)
        trend = trend_context.get("trend", "neutral")

        if trend in ("downtrend", "weak_downtrend"):
            if bearish >= total // 2:
                return "Short"
            return None
        if trend in ("uptrend", "weak_uptrend"):
            if bullish >= total // 2:
                return "Long"
            return None

        # Neutral: need 2+ TFs to agree
        if bullish > bearish and bullish >= 2:
            return "Long"
        elif bearish > bullish and bearish >= 2:
            return "Short"
        elif total >= 1 and '5' in tf_scores:
            return "Long" if tf_scores['5'] > 0 else "Short"

        return None
    except Exception as e:
        log(f"❌ Error determining direction for core: {e}", level="ERROR")
        return None


async def validate_core_conditions(symbol: str, core_candles: Dict, direction: str, trend_context: Dict) -> bool:
    """Validate trend alignment + minimum volume condition"""
    try:
        btc_trend = trend_context.get("btc_trend", "neutral")
        if direction == "Long" and btc_trend in ["bearish", "strong_bearish"]:
            log(f"⚠️ {symbol}: Long rejected - BTC trend is {btc_trend}")
            return False
        if direction == "Short" and btc_trend in ["bullish", "strong_bullish"]:
            log(f"⚠️ {symbol}: Short rejected - BTC trend is {btc_trend}")
            return False

        if direction == "Short":
            is_valid = await validate_short_signal(symbol, core_candles)
            if not is_valid:
                log(f"⚠️ {symbol}: Short signal validation failed")
                return False

        # Volume sanity: reject only if recent volume < 10% of own average (dead market)
        for tf in ['5', '1', '15']:
            candles = core_candles.get(tf)
            if candles and len(candles) >= 20:
                volumes = [float(c.get('volume', 0)) for c in candles[-20:]]
                avg_vol = sum(volumes) / len(volumes)
                recent_vol = sum(volumes[-3:]) / 3
                if avg_vol > 0 and recent_vol < avg_vol * 0.10:
                    log(f"⚠️ {symbol}: Volume collapsed on {tf}m "
                        f"(recent={recent_vol:.0f} vs avg={avg_vol:.0f})")
                    return False
                break  # only check one TF

        return True
    except Exception as e:
        log(f"❌ Error validating conditions for {symbol}: {e}", level="ERROR")
        return False


def validate_momentum_alignment(core_candles: Dict, direction: str) -> bool:
    """Validate momentum alignment across timeframes"""
    try:
        aligned_count = 0
        total_count = 0

        for tf, candles in core_candles.items():
            if not candles or len(candles) < 10:
                continue
            total_count += 1
            has_momentum, mom_direction, strength = detect_momentum_strength(candles)
            if has_momentum and strength > 0.5:
                if direction == "Long" and mom_direction == "bullish":
                    aligned_count += 1
                elif direction == "Short" and mom_direction == "bearish":
                    aligned_count += 1

        return aligned_count >= (total_count // 2) if total_count > 0 else False
    except Exception:
        return False


def validate_volume_breakout(core_candles: Dict) -> bool:
    """Validate volume breakout confirmation"""
    try:
        for tf, candles in core_candles.items():
            if not candles or len(candles) < 20:
                continue
            recent_vol = sum(float(c.get('volume', 0)) for c in candles[-5:]) / 5
            earlier_vol = sum(float(c.get('volume', 0)) for c in candles[-20:-5]) / 15
            if earlier_vol > 0 and recent_vol > earlier_vol * 1.5:
                return True
        return False
    except Exception:
        return False


def validate_price_levels(core_candles: Dict, direction: str) -> bool:
    """Validate price is respecting key levels"""
    try:
        for tf, candles in core_candles.items():
            if not candles or len(candles) < 20:
                continue

            highs = [float(c.get('high', 0)) for c in candles[-20:]]
            lows = [float(c.get('low', 0)) for c in candles[-20:]]
            current_price = float(candles[-1]['close'])

            recent_high = max(highs)
            recent_low = min(lows)
            price_range = recent_high - recent_low
            if price_range <= 0:
                continue

            position_in_range = (current_price - recent_low) / price_range
            if direction == "Long" and position_in_range < 0.7:
                return True
            elif direction == "Short" and position_in_range > 0.3:
                return True

        return False
    except Exception:
        return False


def validate_trend_coherence(core_candles: Dict, direction: str, trend_context: Dict) -> bool:
    """Validate trend coherence across indicators"""
    try:
        btc_trend = trend_context.get("btc_trend", "neutral")
        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))

        if direction == "Long" and btc_trend in ["bullish", "strong_bullish"]:
            return True
        elif direction == "Short" and btc_trend in ["bearish", "strong_bearish"]:
            return True

        if btc_trend == "neutral" and trend_strength >= 0.6:
            return True

        return False
    except Exception as e:
        log(f"Trend coherence validation error: {e}", level="WARN")
        return False


def detect_momentum_acceleration(core_candles: Dict, direction: str) -> bool:
    """Detect if momentum is accelerating"""
    try:
        if '5' not in core_candles or len(core_candles['5']) < 10:
            return False

        candles = core_candles['5'][-10:]
        closes = []
        for candle in candles:
            try:
                if isinstance(candle, dict):
                    close = float(candle.get('close', 0))
                else:
                    continue
                if close > 0:
                    closes.append(close)
            except Exception:
                continue

        if len(closes) < 5:
            return False

        first_price = closes[0]
        last_price = closes[-1]
        price_change = ((last_price - first_price) / first_price) * 100

        if direction.lower() == "long":
            return price_change > 0.5
        else:
            return price_change < -0.5
    except Exception:
        return False


async def get_core_confirmations(symbol: str, core_candles: Dict, direction: str, trend_context: Dict) -> List[str]:
    """Collect all confirmations passing for this symbol"""
    confirmations = []

    try:
        if validate_momentum_alignment(core_candles, direction):
            confirmations.append("momentum_alignment")

        if validate_volume_breakout(core_candles):
            confirmations.append("volume_breakout")

        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))
        if trend_strength > 0.5:
            confirmations.append("strong_trend")

        if validate_price_levels(core_candles, direction):
            confirmations.append("price_levels")

        if validate_trend_coherence(core_candles, direction, trend_context):
            confirmations.append("trend_coherence")

        if has_pump_potential(core_candles, direction):
            confirmations.append("pump_potential")
    except Exception as e:
        log(f"❌ Error getting confirmations for {symbol}: {e}", level="ERROR")

    return confirmations


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY TYPE DETERMINATION (QUALITY PATCH — tighter tiers)
# ─────────────────────────────────────────────────────────────────────────────

def determine_core_strategy_type(score: float, confidence: float, trend_strength: float) -> Optional[str]:
    """QUALITY PATCH: tighter quality tiers per strategy"""
    try:
        # Swing — premium quality only
        if score >= MIN_SWING_SCORE and confidence >= MIN_SWING_CONFIDENCE \
                and trend_strength >= MIN_SWING_TREND_STR:
            return "CoreSwing"
        # Intraday — strong quality
        elif score >= MIN_INTRADAY_SCORE and confidence >= MIN_INTRADAY_CONFIDENCE \
                and trend_strength >= MIN_INTRADAY_TREND_STR:
            return "CoreIntraday"
        # Scalp — needs confidence ≥ 70 (was 65)
        elif score >= MIN_SCALP_SCORE and confidence >= MIN_SCALP_CONFIDENCE:
            return "CoreScalp"
        else:
            log(f"⛔ No strategy type: score={score:.1f} "
                f"conf={confidence:.0f} strength={trend_strength:.2f}")
            return None
    except Exception as e:
        log(f"❌ Error determining strategy type: {e}", level="ERROR")
        return None


def check_strategy_position_limits(strategy_type: str) -> bool:
    """Check strategy-specific position limits"""
    try:
        total_active = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        if total_active >= MAX_CORE_POSITIONS:
            log(f"🚫 Global position limit reached: {total_active}/{MAX_CORE_POSITIONS}")
            return False

        strategy_counts = {"CoreScalp": 0, "CoreIntraday": 0, "CoreSwing": 0}
        for trade in active_trades.values():
            if not trade.get("exited", False):
                trade_type = trade.get("trade_type", "")
                if trade_type in strategy_counts:
                    strategy_counts[trade_type] += 1
                elif f"Core{trade_type}" in strategy_counts:
                    strategy_counts[f"Core{trade_type}"] += 1

        if strategy_type == "CoreScalp" and strategy_counts["CoreScalp"] >= MAX_SCALP_POSITIONS:
            log(f"🚫 Scalp position limit reached: "
                f"{strategy_counts['CoreScalp']}/{MAX_SCALP_POSITIONS}")
            return False
        elif strategy_type == "CoreIntraday" and strategy_counts["CoreIntraday"] >= MAX_INTRADAY_POSITIONS:
            log(f"🚫 Intraday position limit reached: "
                f"{strategy_counts['CoreIntraday']}/{MAX_INTRADAY_POSITIONS}")
            return False
        elif strategy_type == "CoreSwing" and strategy_counts["CoreSwing"] >= MAX_SWING_POSITIONS:
            log(f"🚫 Swing position limit reached: "
                f"{strategy_counts['CoreSwing']}/{MAX_SWING_POSITIONS}")
            return False

        return True
    except Exception as e:
        log(f"❌ Error checking position limits: {e}", level="ERROR")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SCALP HUNTER SCAN
# ─────────────────────────────────────────────────────────────────────────────

async def scalp_hunter_scan(symbols: List[str], trend_context: Dict) -> None:
    """1% Scalp Hunter scan loop. Runs after core strategy."""
    if not SCALP_HUNTER_ENABLED:
        return

    cfg = SCALP_CONFIG

    btc_candles_1m = list(live_candles.get("BTCUSDT", {}).get("1", []))
    btc_candles_5m = list(live_candles.get("BTCUSDT", {}).get("5", []))
    if btc_candles_1m:
        update_btc_state(btc_candles_1m, btc_candles_5m)

    from monitor import active_trades
    current_scalp_count = sum(
        1 for t in active_trades.values()
        if not t.get("exited", False) and t.get("trade_type") == "ScalpHunter"
    )
    if current_scalp_count >= MAX_SCALP_CONCURRENT:
        log(f"⛔ SCALP HUNTER: Max concurrent positions reached ({current_scalp_count})")
        return

    processed = 0
    for symbol in symbols:
        try:
            sym_candles = live_candles.get(symbol, {})
            candles_by_tf = {
                tf: list(sym_candles.get(tf, []))
                for tf in ["1", "3", "5"]
                if sym_candles.get(tf)
            }
            if not candles_by_tf:
                continue

            has_enough = any(len(v) >= 30 for v in candles_by_tf.values())
            if not has_enough:
                continue

            primary_tf = "1" if "1" in candles_by_tf else list(candles_by_tf.keys())[0]
            last_candle = candles_by_tf[primary_tf][-1]
            current_price = float(last_candle.get("close", 0))
            if not current_price:
                continue

            last_signal_ts = scalp_signal_cooldown.get(symbol, 0)
            if time.time() - last_signal_ts < SCALP_COOLDOWN_SECONDS:
                continue

            if symbol in active_trades and not active_trades[symbol].get("exited", False):
                continue

            result = evaluate_scalp_setup(symbol, candles_by_tf, current_price)
            result = update_watchlist(symbol, result)

            if result["should_trade"]:
                await _fire_scalp_signal(symbol, result, candles_by_tf, current_price)
                await asyncio.sleep(0.1)

            processed += 1
            if processed % 20 == 0:
                await asyncio.sleep(0.01)
        except Exception as e:
            log(f"⚠️ SCALP HUNTER: Error evaluating {symbol}: {e}", level="WARN")
            continue


async def _fire_scalp_signal(symbol: str, result: Dict, candles_by_tf: Dict, current_price: float) -> None:
    """Send Telegram signal and optionally place order for scalp setup"""
    direction  = result["direction"]
    sl_price   = result["sl_price"]
    tp1_price  = result["tp1_price"]
    sl_pct     = result["sl_pct"]
    tp1_pct    = result["tp1_pct"]
    score      = result["score"]
    confidence = result["confidence"]
    reasons    = result["reasons"]
    details    = result["details"]

    if is_duplicate_signal(symbol):
        log(f"🔁 SCALP HUNTER: Duplicate signal blocked for {symbol}")
        return

    log(f"🎯 SCALP HUNTER SIGNAL: {symbol} {direction.upper()} "
        f"| Score={score} | Conf={confidence:.1f}%")

    msg = format_scalp_signal(
        symbol=symbol, direction=direction, entry_price=current_price,
        sl_price=sl_price, tp1_price=tp1_price, sl_pct=sl_pct, tp1_pct=tp1_pct,
        score=score, confidence=confidence, reasons=reasons, details=details,
        btc_state=_btc_state.get("trend", "unknown"),
    )
    await send_telegram_message(msg)

    log_signal(symbol)
    scalp_signal_cooldown[symbol] = time.time()

    if not SCALP_HUNTER_SIGNALS_ONLY:
        signal_data = {
            "symbol": symbol, "direction": direction.capitalize(),
            "strategy": "ScalpHunter", "trade_type": "ScalpHunter",
            "score": float(score), "confidence": confidence,
            "regime": "scalp_hunter",
            "sl_price": sl_price, "tp1_price": tp1_price,
            "sl_pct": sl_pct, "tp1_pct": tp1_pct,
            "trailing_pct": 0.5,
            "tp1_partial_close": SCALP_CONFIG["tp1_partial_close"],
            "leverage": SCALP_CONFIG["leverage"],
            "candles": candles_by_tf,
            "is_scalp_hunter": True,
        }
        trade_result = await execute_trade_if_valid(signal_data, max_risk=SCALP_RISK_PCT)
        if trade_result and trade_result.get("success"):
            register_scalp_trade(symbol)
            track_signal(symbol, float(score))
            log(f"✅ SCALP HUNTER: Trade placed for {symbol}")
        else:
            log(f"❌ SCALP HUNTER: Trade placement failed for {symbol}")
    else:
        log(f"📡 SCALP HUNTER: Signal-only mode — no order placed for {symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE STRATEGY MAIN SCAN
# ─────────────────────────────────────────────────────────────────────────────

async def core_strategy_scan(symbols: List[str], trend_context: Dict):
    """PURE CORE STRATEGY — strict quality filters only"""
    source = fix_live_candles_structure(live_candles)

    from scanner import symbol_category_map
    for sym in symbols:
        if sym not in symbol_category_map:
            symbol_category_map[sym] = 'linear'

    try:
        if not symbols or len(symbols) == 0:
            log("⚠️ CORE STRATEGY: No symbols to scan", level="WARN")
            return

        current_positions = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        if current_positions >= MAX_CORE_POSITIONS:
            log(f"🚫 CORE STRATEGY: Max positions reached ({current_positions}/{MAX_CORE_POSITIONS})")
            return

        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))
        trend_direction = trend_context.get("trend", "neutral")
        log(f"📊 Trend strength resolved: {trend_strength:.2f}")

        strategy_rec = trend_context.get("recommendations", {}).get("primary_strategy", "")
        opportunity = trend_context.get("opportunity_score", 0.5)

        if strategy_rec == "wait_and_see" and opportunity < 0.35:
            log(f"⏸️ CORE STRATEGY: Skipping scan — WAIT_AND_SEE with low opportunity ({opportunity:.2f})")
            return

        log(f"📈 Strategy recommendation: {strategy_rec} | Opportunity: {opportunity:.2f}")
        log(f"🔍 CORE STRATEGY: Scanning {len(symbols)} symbols | Trend: {trend_direction} ({trend_strength:.2f})")

        scanned_count = 0
        core_signals_found = 0

        for symbol in symbols:
            try:
                _diag = symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

                # Check if we can process
                can_process, reason = await trade_lock_manager.can_process_symbol(symbol)
                if not can_process:
                    if _diag:
                        log(f"🔴 DIAG {symbol}: can_process=False | reason={reason}", level="DEBUG")
                    continue

                # Acquire lock
                if not await trade_lock_manager.acquire_trade_lock(symbol):
                    if _diag:
                        log(f"🔴 DIAG {symbol}: lock acquire failed", level="DEBUG")
                    continue

                try:
                    # Skip if already in active trade
                    if symbol in active_trades and not active_trades[symbol].get("exited", False):
                        if _diag:
                            log(f"🔴 DIAG {symbol}: active trade exists", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # Skip if in cooldown
                    if symbol in signal_cooldown:
                        time_diff = time.time() - signal_cooldown[symbol]
                        if time_diff < SIGNAL_COOLDOWN_TIME:
                            if _diag:
                                log(f"🔴 DIAG {symbol}: signal cooldown {time_diff:.0f}s "
                                    f"of {SIGNAL_COOLDOWN_TIME}s", level="DEBUG")
                            trade_lock_manager.release_trade_lock(symbol, False)
                            continue

                    # Skip if in recent exit cooldown
                    if symbol in recent_exits:
                        time_diff = time.time() - recent_exits[symbol]
                        if time_diff < EXIT_COOLDOWN:
                            if _diag:
                                log(f"🔴 DIAG {symbol}: exit cooldown {time_diff:.0f}s "
                                    f"of {EXIT_COOLDOWN}s", level="DEBUG")
                            trade_lock_manager.release_trade_lock(symbol, False)
                            continue

                    if _diag:
                        log(f"🟢 DIAG {symbol}: passed all pre-candle checks", level="DEBUG")

                    # Get candles for core timeframes
                    core_candles = {}
                    src = fix_live_candles_structure({symbol: live_candles.get(symbol, {})}).get(symbol, {})

                    for tf in ['1', '5', '15']:
                        tf_data = src.get(tf)
                        if tf_data:
                            candles = list(tf_data)
                            min_needed = 12 if tf in ('1', '5') else 8
                            if len(candles) >= min_needed:
                                core_candles[tf] = candles

                    if not all(tf in core_candles for tf in ('1', '5')):
                        missing = [tf for tf in ('1', '5') if tf not in core_candles]
                        available = {tf: len(list(src.get(tf, []))) for tf in ['1', '5', '15']}
                        log(f"⏭️ {symbol}: Missing TFs {missing} | "
                            f"candle counts: {available}", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    scanned_count += 1

                    # === CORE STRATEGY SIGNAL GENERATION ===

                    # 1. Get full scoring data ONCE
                    score_result = score_symbol(symbol, core_candles, trend_context)
                    score, tf_scores, trade_type, indicator_scores, used_indicators = score_result

                    # 2. Calculate core score (base + small bonus)
                    core_score = score + _calculate_momentum_and_regime_bonus(symbol, core_candles, trend_context)

                    # 3. Determine direction
                    direction = determine_core_direction(core_candles, trend_context)
                    if not direction:
                        log(f"⏭️ {symbol}: No direction determined", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 4. Calculate confidence — QUALITY PATCH: gate at 68
                    confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                    if confidence < EARLY_CONFIDENCE_GATE:
                        log(f"⏭️ {symbol}: Confidence too low ({confidence}% < {EARLY_CONFIDENCE_GATE}%)",
                            level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 5. Validate core strategy conditions
                    if not await validate_core_conditions(symbol, core_candles, direction, trend_context):
                        log(f"⏭️ {symbol}: Core conditions not met", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 6. Determine strategy type — QUALITY PATCH: tighter tiers
                    strategy_type = determine_core_strategy_type(core_score, confidence, trend_strength)
                    if not strategy_type:
                        log(f"⏭️ {symbol}: No strategy type "
                            f"(score={core_score:.1f}, conf={confidence}%)", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 7. Check strategy-specific position limits
                    if not check_strategy_position_limits(strategy_type):
                        log(f"⏭️ {symbol}: Position limits reached for {strategy_type}", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 8. Get confirmations — QUALITY PATCH: require 3+
                    confirmations = await get_core_confirmations(symbol, core_candles, direction, trend_context)
                    if len(confirmations) < MIN_CONFIRMATIONS:
                        log(f"⏭️ {symbol}: Insufficient confirmations "
                            f"({len(confirmations)} < {MIN_CONFIRMATIONS}): {confirmations}", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 9. Check for duplicate signal
                    if is_duplicate_signal(symbol):
                        log(f"⏭️ {symbol}: Duplicate signal blocked", level="DEBUG")
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # === SIGNAL VALIDATED — EXECUTE TRADE ===
                    core_signals_found += 1

                    log(f"🎯 CORE STRATEGY SIGNAL: {symbol}")
                    log(f"   Score: {core_score:.1f} | Confidence: {confidence}%")
                    log(f"   Direction: {direction} | Type: {strategy_type}")
                    log(f"   Confirmations: {', '.join(confirmations)}")

                    trade_result = await execute_core_trade(
                        symbol=symbol, direction=direction, strategy_type=strategy_type,
                        score=core_score, confidence=confidence, confirmations=confirmations,
                        core_candles=core_candles, trend_context=trend_context
                    )

                    if trade_result and trade_result.get("success"):
                        signal_cooldown[symbol] = time.time()
                        trade_lock_manager.release_trade_lock(symbol, True)
                    else:
                        signal_cooldown[symbol] = time.time() - (SIGNAL_COOLDOWN_TIME * 0.8)
                        trade_lock_manager.release_trade_lock(symbol, False)

                except Exception as e:
                    log(f"❌ CORE STRATEGY: Error processing {symbol}: {e}", level="ERROR")
                    trade_lock_manager.release_trade_lock(symbol, False)
                    continue

            except Exception as e:
                log(f"❌ CORE STRATEGY: Error with {symbol}: {e}", level="ERROR")
                continue

        log(f"📊 CORE STRATEGY SUMMARY: {scanned_count} scanned, "
            f"{core_signals_found} quality signals")

        # <<< SCALP HUNTER >>> — Run scalp hunter scan after core strategy
        if SCALP_HUNTER_ENABLED:
            await scalp_hunter_scan(symbols, trend_context)

    except Exception as e:
        log(f"❌ CORE STRATEGY: Error in scan: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

async def execute_core_trade(
    symbol: str, direction: str, strategy_type: str,
    score: float, confidence: float, confirmations: List[str],
    core_candles: Dict, trend_context: Dict
) -> Dict[str, Any]:
    """Execute a core strategy trade"""
    try:
        base_risk = CORE_RISK_PERCENTAGES.get(strategy_type, 0.02)

        if confidence >= 85:
            adjusted_risk = base_risk * 1.2
        elif confidence >= 75:
            adjusted_risk = base_risk
        else:
            adjusted_risk = base_risk * 0.8

        adjusted_risk = min(adjusted_risk, 0.03)

        if symbol in active_trades and not active_trades[symbol].get("exited", False):
            log(f"🚫 {symbol}: Trade execution blocked - position already exists")
            return {"success": False, "reason": "position_exists"}

        _price = 0.0
        for _tf in ['1', '5', '15']:
            if _tf in core_candles and core_candles[_tf]:
                try:
                    _price = float(core_candles[_tf][-1].get('close', 0))
                    if _price > 0:
                        break
                except Exception:
                    continue

        if _price == 0:
            log(f"❌ {symbol}: Cannot get entry price from candles", level="ERROR")
            return {"success": False, "reason": "no_price"}

        signal_data = {
            "symbol": symbol, "direction": direction,
            "strategy": f"CORE_{strategy_type}",
            "score": score, "confidence": confidence,
            "regime": "core_strategy", "confirmations": confirmations,
            "risk_adjusted": True, "core_strategy": True,
            "candles": core_candles,
            "trade_type": strategy_type.replace("Core", ""),
            "price": _price, "entry_price": _price,
        }

        log(f"🎯 CORE STRATEGY EXECUTION: {symbol}")
        log(f"   Direction: {direction} | Type: {strategy_type}")
        log(f"   Score: {score:.1f} | Confidence: {confidence}%")
        log(f"   Confirmations: {', '.join(confirmations) if confirmations else '—'}")
        log(f"   Risk: {adjusted_risk:.2%}")

        trade_result = await execute_trade_if_valid(signal_data, adjusted_risk)

        if trade_result and trade_result.get("success"):
            log_signal(symbol)
            track_signal(symbol, score)
            await send_core_strategy_notification(signal_data, trade_result)
            log(f"✅ CORE STRATEGY: Trade executed successfully for {symbol}")
        else:
            log(f"❌ CORE STRATEGY: Trade execution failed for {symbol}")

        return trade_result

    except Exception as e:
        log(f"❌ CORE STRATEGY: Error executing trade for {symbol}: {e}", level="ERROR")
        return {"success": False, "reason": str(e)}


async def send_core_strategy_notification(signal_data: Dict, trade_result: Dict):
    """Send core strategy specific notification"""
    try:
        symbol = signal_data["symbol"]
        direction = signal_data["direction"]
        strategy = signal_data["strategy"]
        score = signal_data["score"]
        confidence = signal_data["confidence"]
        confirmations = signal_data.get("confirmations", [])

        msg = f"🎯 <b>CORE STRATEGY EXECUTED</b>\n\n"
        msg += f"Symbol: <b>{symbol}</b>\n"
        msg += f"Direction: <b>{direction.upper()}</b>\n"
        msg += f"Strategy: <b>{strategy}</b>\n"
        msg += f"Score: <b>{score:.1f}</b>\n"
        msg += f"Confidence: <b>{confidence}%</b>\n"

        if confirmations:
            msg += f"Confirmations ({len(confirmations)}):\n"
            for conf in confirmations:
                msg += f"   • {conf}\n"

        if trade_result:
            msg += f"\n💰 Entry: <b>{trade_result.get('entry_price', 'N/A')}</b>"
            msg += f"\n🛡️ Stop Loss: <b>{trade_result.get('sl_price', 'N/A')}</b>"
            msg += f"\n🎯 Take Profit: <b>{trade_result.get('tp_price', 'N/A')}</b>"

        await send_telegram_message(msg)
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error sending notification: {e}", level="ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# MONITOR LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def core_monitor_loop():
    """Core strategy status logger only.
    All exit logic lives in monitor.py → unified_exit_manager.process_trade_exits().
    """
    log("🔍 Starting core_monitor_loop (status logging only)...")
    await asyncio.sleep(15)

    while True:
        try:
            active = {k: v for k, v in active_trades.items() if not v.get("exited", False)}
            if not active:
                await asyncio.sleep(30)
                continue

            log(f"📊 CORE STATUS: {len(active)} active trades")
            for symbol, trade in list(active.items()):
                try:
                    pnl = trade.get("current_pnl_pct", 0)
                    tp1_hit = "✅" if trade.get("tp1_hit") else "⏳"
                    log(f"   {symbol}: P&L={pnl:+.2f}% | TP1={tp1_hit}")
                except Exception:
                    pass

            await asyncio.sleep(30)
        except Exception as e:
            log(f"❌ CORE STATUS: Error in status loop: {e}", level="ERROR")
            await asyncio.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# BOT LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

async def run_core_bot():
    """Core strategy bot - simplified and focused"""
    log("🚀 CORE STRATEGY BOT starting...")

    await fetch_symbol_info()
    symbols = await fetch_symbols()
    log(f"✅ CORE STRATEGY: Fetched {len(symbols)} symbols.")

    load_active_trades()
    await sync_bot_with_bybit(send_telegram=True)

    if len(active_trades) == 0:
        await recover_active_trades_from_exchange()

    asyncio.create_task(stream_candles(symbols))
    asyncio.create_task(core_monitor_loop())
    asyncio.create_task(monitor_active_trades())
    asyncio.create_task(monitor_btc_trend_accuracy())
    asyncio.create_task(monitor_altseason_status())
    asyncio.create_task(periodic_trade_sync())
    asyncio.create_task(bybit_sync_loop(120))
    asyncio.create_task(lock_manager_maintenance())

    await asyncio.sleep(5)
    log("🚀 CORE STRATEGY BOT fully initialized - starting main loop")

    while True:
        try:
            trend_context = await get_trend_context_cached()
            await core_strategy_scan(symbols, trend_context)
            await send_daily_report()
        except Exception as e:
            log(f"❌ CORE STRATEGY: Error in main loop: {e}", level="ERROR")
            write_log(f"CORE STRATEGY MAIN LOOP ERROR: {str(e)}", level="ERROR")
            await send_error_to_telegram(traceback.format_exc())

        await asyncio.sleep(1.0)
        await trade_lock_manager.sync_with_exchange()


async def lock_manager_maintenance():
    """Periodic maintenance for lock manager"""
    while True:
        try:
            await trade_lock_manager.sync_with_exchange()
            await trade_lock_manager.cleanup_stale_locks()
        except Exception as e:
            log(f"❌ Lock manager maintenance error: {e}", level="ERROR")
        await asyncio.sleep(30)


async def bybit_sync_loop(interval_sec: int = 120):
    """Periodic sync with Bybit exchange"""
    while True:
        try:
            await sync_bot_with_bybit(send_telegram=False)
        except Exception as e:
            await send_error_to_telegram(f"Core strategy sync error: {e}")
        await asyncio.sleep(interval_sec)


def debug_live_link():
    """Debug function to check live_candles state"""
    try:
        from websocket_candles import live_candles as ws_live
    except Exception:
        ws_live = None
    try:
        log(f"🔗 live_candles id(main)={id(live_candles)} | "
            f"id(ws)={id(ws_live) if ws_live else 'N/A'}")
    except Exception:
        pass

    if ws_live:
        shown = 0
        for sym, tfs in ws_live.items():
            try:
                c1 = len(tfs.get('1', []))
                c5 = len(tfs.get('5', []))
                c15 = len(tfs.get('15', []))
                log(f"📊 {sym}: 1m={c1}, 5m={c5}, 15m={c15}")
            except Exception:
                continue
            shown += 1
            if shown >= 5:
                break


# === ENTRY POINT ===
if __name__ == "__main__":
    import sys
    import platform

    log("🔧 DEBUG: CORE STRATEGY main.py is running...")
    log(f"🔍 CORE STRATEGY thresholds - "
        f"Scalp: {MIN_SCALP_SCORE}, Intraday: {MIN_INTRADAY_SCORE}, Swing: {MIN_SWING_SCORE}")
    log(f"🔒 CORE STRATEGY limits - Max positions: {MAX_CORE_POSITIONS}, "
        f"Scalp: {MAX_SCALP_POSITIONS}, Intraday: {MAX_INTRADAY_POSITIONS}, "
        f"Swing: {MAX_SWING_POSITIONS}")

    async def restart_forever():
        """Core strategy restart mechanism with crash recovery"""
        while True:
            try:
                await run_core_bot()
            except Exception as e:
                err_msg = f"🔁 Restarting CORE STRATEGY bot due to crash:\n{traceback.format_exc()}"
                log(err_msg, level="ERROR")
                await send_error_to_telegram(err_msg)
                await asyncio.sleep(10)

    # Windows: switch to SelectorEventLoop to avoid noisy ConnectionResetError
    if platform.system() == "Windows":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    try:
        asyncio.run(restart_forever())
    except KeyboardInterrupt:
        log("⏹️ CORE STRATEGY BOT stopped by user")
        sys.exit(0)
