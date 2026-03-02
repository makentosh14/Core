#!/usr/bin/env python3
"""
main.py - Core Strategy Trading Bot
FIXED VERSION - All issues resolved
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
EXIT_COOLDOWN = 120  # 2 minutes cooldown after exit

# Core Strategy Thresholds - Enhanced for Quality
MIN_SCALP_SCORE = 9.0       # High quality scalps only
MIN_INTRADAY_SCORE = 10.0   # High quality intraday only  
MIN_SWING_SCORE = 14.0      # High quality swings only

# Core Strategy Risk Management - Conservative
CORE_RISK_PERCENTAGES = {
    "Scalp": 0.025,      # 2.5% risk for scalps
    "Intraday": 0.02,    # 2% risk for intraday  
    "Swing": 0.015,      # 1.5% risk for swing trades
    "CoreScalp": 0.025,
    "CoreIntraday": 0.02,
    "CoreSwing": 0.015
}

# Core Strategy Position Limits
MAX_CORE_POSITIONS = 3       # Maximum 3 concurrent positions
MAX_SCALP_POSITIONS = 2      # Maximum 2 scalp positions
MAX_INTRADAY_POSITIONS = 1   # Maximum 1 intraday position
MAX_SWING_POSITIONS = 1      # Maximum 1 swing position

# <<< SCALP HUNTER >>> — Scalp Hunter Configuration
SCALP_HUNTER_ENABLED = True          # Master switch
SCALP_HUNTER_SIGNALS_ONLY = True    # True = Telegram only, no exchange orders
SCALP_RISK_PCT = 0.01                # 1% account risk per scalp trade
MAX_SCALP_CONCURRENT = 2            # Max concurrent scalp positions
scalp_signal_cooldown: Dict[str, float] = {}   # Per-symbol cooldown tracker
SCALP_COOLDOWN_SECONDS = 300         # 5 minutes between signals per symbol


def fix_live_candles_structure(candles_data):
    """
    Fix and normalize live candles data structure.
    Ensures consistent format: {symbol: {timeframe: [candles]}}
    """
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
                
            # Convert to list if needed
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
    """Safely extract candles for a symbol from any available timeframe"""
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
                    except:
                        continue
        return None
    except Exception as e:
        log(f"❌ Safe candle extraction error for {symbol}: {e}", level="ERROR")
        return None


async def filter_core_symbols(symbols: List[str]) -> List[str]:
    """
    Simple filter - focus on basic criteria only.
    Returns symbols that have sufficient data for analysis.
    """
    source = fix_live_candles_structure(live_candles)
    log(f"✅ Fixed live_candles structure before filtering")
    
    filtered = []
    
    for symbol in symbols:
        try:
            # Basic checks
            if 'USDT' not in symbol:
                continue
                
            if symbol not in source:
                continue
            
            # Check for available candles in required timeframes
            tf_count = 0
            for tf in ['1', '5', '15']:
                if tf in source[symbol] and source[symbol][tf]:
                    tf_data = source[symbol][tf]
                    if isinstance(tf_data, list) and len(tf_data) >= 12:
                        tf_count += 1
            
            # Need at least 2 timeframes with data
            if tf_count < 2:
                continue
            
            # Get any available candles for volume check
            candles = None
            for tf in ['1', '5', '15']:
                if tf in source[symbol] and source[symbol][tf]:
                    tf_data = source[symbol][tf]
                    if isinstance(tf_data, list) and len(tf_data) >= 5:
                        candles = tf_data[-20:] if len(tf_data) >= 20 else tf_data
                        break
            
            if not candles or len(candles) < 5:
                continue
            
            # Very basic volume check
            volumes = [float(c.get('volume', 0)) for c in candles[-5:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            # Low threshold - just need some activity
            if avg_volume > 10000:
                filtered.append(symbol)
                
            # Stop at 50 symbols for efficiency
            if len(filtered) >= 50:
                break
                
        except Exception as e:
            log(f"⚠️ Error filtering {symbol}: {e}", level="WARN")
            continue
    
    log(f"✅ Filtered to {len(filtered)} symbols with relaxed criteria")
    return filtered


async def calculate_core_score(symbol: str, core_candles: Dict, trend_context: Dict) -> float:
    try:
        base_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
            symbol, core_candles, trend_context
        )

        momentum_bonus = 0
        for tf, candles in core_candles.items():
            if candles and len(candles) >= 10:
                has_momentum, direction, strength = detect_momentum_strength(candles)
                if has_momentum and strength > 0.6:
                    momentum_bonus = strength * 1.5
                    break

        # Enhanced system uses lowercase regime value
        regime = trend_context.get("regime", "unknown").lower()

        if regime == "transitional":
            regime_bonus = 2.0
        elif regime in ("trending", "strong_trending"):
            regime_bonus = 1.5
        elif regime in ("range_bound", "consolidating"):
            regime_bonus = 1.2
        else:
            regime_bonus = 1.0

        final_score = base_score + momentum_bonus + regime_bonus
        return final_score

    except Exception as e:
        log(f"❌ Error calculating score for {symbol}: {e}", level="ERROR")
        return 6.0


def determine_core_direction(core_candles: Dict, trend_context: Dict) -> Optional[str]:
    try:
        tf_scores = {}
        for tf, candles in core_candles.items():
            if not candles or len(candles) < 5:
                continue
            closes = [float(c.get('close', 0)) for c in candles[-10:]]
            if len(closes) < 5:
                continue
            # Simple momentum: compare last close to 5-candle average
            avg = sum(closes[:-1]) / len(closes[:-1])
            tf_scores[tf] = closes[-1] - avg

        if not tf_scores:
            return None

        # Count bullish vs bearish timeframes
        bullish = sum(1 for v in tf_scores.values() if v > 0)
        bearish = sum(1 for v in tf_scores.values() if v < 0)
        total = len(tf_scores)

        trend = trend_context.get("trend", "neutral")

        # In downtrend: allow Short signals
        if trend in ("downtrend", "weak_downtrend"):
            if bearish >= total // 2:
                return "Short"
            return None  # Don't force Long against downtrend

        # In uptrend: allow Long signals
        if trend in ("uptrend", "weak_uptrend"):
            if bullish >= total // 2:
                return "Long"
            return None

        # Neutral/choppy: need majority agreement
        if bullish > bearish and bullish >= 2:
            return "Long"
        elif bearish > bullish and bearish >= 2:
            return "Short"

        return None

    except Exception as e:
        log(f"❌ Error determining direction for core: {e}", level="ERROR")
        return None

async def validate_core_conditions(symbol: str, core_candles: Dict, direction: str, trend_context: Dict) -> bool:
    try:
        btc_trend = trend_context.get("btc_trend", "neutral")

        if direction == "Long" and btc_trend in ["bearish", "strong_bearish"]:
            log(f"⚠️ {symbol}: Long rejected - BTC trend is {btc_trend}")
            return False

        if direction == "Short" and btc_trend in ["bullish", "strong_bullish"]:
            log(f"⚠️ {symbol}: Short rejected - BTC trend is {btc_trend}")
            return False

        # FIX: validate_short_signal is async — must be awaited.
        # Also pass correct arguments matching the async signature in trend_filters.py
        if direction == "Short":
            is_valid = await validate_short_signal(symbol, core_candles)
            if not is_valid:
                log(f"⚠️ {symbol}: Short signal validation failed")
                return False

        for tf, candles in core_candles.items():
            if candles and len(candles) >= 5:
                volumes = [float(c.get('volume', 0)) for c in candles[-5:]]
                avg_vol = sum(volumes) / len(volumes)
                if avg_vol < 5000:
                    log(f"⚠️ {symbol}: Low volume on {tf}m timeframe")
                    return False
                break

        return True

    except Exception as e:
        log(f"❌ Error validating conditions for {symbol}: {e}", level="ERROR")
        return False


def determine_core_strategy_type(score: float, confidence: float, trend_strength: float) -> Optional[str]:
    """Determine core strategy type with strict requirements"""
    try:
        if score >= MIN_SWING_SCORE and confidence >= 80 and trend_strength >= 0.7:
            return "CoreSwing"
        elif score >= MIN_INTRADAY_SCORE and confidence >= 70 and trend_strength >= 0.35:
            return "CoreIntraday"
        elif score >= MIN_SCALP_SCORE and confidence >= 65:
            return "CoreScalp"
        else:
            log(f"⛔ No strategy type: score={score:.1f} conf={confidence:.0f} strength={trend_strength:.2f}")
            return None


def check_strategy_position_limits(strategy_type: str) -> bool:
    """Check strategy-specific position limits"""
    try:
        # FIX: Count ALL non-exited positions for the global cap,
        # regardless of trade_type (covers recovered/synced trades too)
        total_active = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        
        if total_active >= MAX_CORE_POSITIONS:
            log(f"🚫 Global position limit reached: {total_active}/{MAX_CORE_POSITIONS}")
            return False

        # Per-strategy limits (only count Core-typed trades for sub-limits)
        strategy_counts = {"CoreScalp": 0, "CoreIntraday": 0, "CoreSwing": 0}

        for trade in active_trades.values():
            if not trade.get("exited", False):
                trade_type = trade.get("trade_type", "")
                # Normalize: "Scalp" → "CoreScalp", "Intraday" → "CoreIntraday" etc.
                if trade_type in strategy_counts:
                    strategy_counts[trade_type] += 1
                elif f"Core{trade_type}" in strategy_counts:
                    strategy_counts[f"Core{trade_type}"] += 1

        if strategy_type == "CoreScalp" and strategy_counts["CoreScalp"] >= MAX_SCALP_POSITIONS:
            log(f"🚫 Scalp position limit reached: {strategy_counts['CoreScalp']}/{MAX_SCALP_POSITIONS}")
            return False
        elif strategy_type == "CoreIntraday" and strategy_counts["CoreIntraday"] >= MAX_INTRADAY_POSITIONS:
            log(f"🚫 Intraday position limit reached: {strategy_counts['CoreIntraday']}/{MAX_INTRADAY_POSITIONS}")
            return False
        elif strategy_type == "CoreSwing" and strategy_counts["CoreSwing"] >= MAX_SWING_POSITIONS:
            log(f"🚫 Swing position limit reached: {strategy_counts['CoreSwing']}/{MAX_SWING_POSITIONS}")
            return False

        return True

    except Exception as e:
        log(f"❌ Error checking position limits: {e}", level="ERROR")
        return False  # FIX: Return False on error (was True — was allowing trades on exception!)


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
        
    except Exception as e:
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
        
    except Exception as e:
        return False


def validate_price_levels(core_candles: Dict, direction: str) -> bool:
    """Validate price is respecting key levels"""
    try:
        for tf, candles in core_candles.items():
            if not candles or len(candles) < 20:
                continue
                
            # Find recent high/low
            highs = [float(c.get('high', 0)) for c in candles[-20:]]
            lows = [float(c.get('low', 0)) for c in candles[-20:]]
            current_price = float(candles[-1]['close'])
            
            recent_high = max(highs)
            recent_low = min(lows)
            price_range = recent_high - recent_low
            
            if price_range <= 0:
                continue
            
            # Check position in range
            position_in_range = (current_price - recent_low) / price_range
            
            if direction == "Long" and position_in_range < 0.7:
                return True  # Good for long - not at top
            elif direction == "Short" and position_in_range > 0.3:
                return True  # Good for short - not at bottom
        
        return False
        
    except Exception as e:
        return False


def validate_trend_coherence(core_candles: Dict, direction: str, trend_context: Dict) -> bool:
    """Validate trend coherence across indicators"""
    try:
        btc_trend = trend_context.get("btc_trend", "neutral")
        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))
        
        # Strong trend alignment gives bonus
        if direction == "Long" and btc_trend in ["bullish", "strong_bullish"]:
            return True
        elif direction == "Short" and btc_trend in ["bearish", "strong_bearish"]:
            return True
        
        # Neutral is acceptable with good strength
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
            except:
                continue
        
        if len(closes) < 5:
            return False
        
        # Simple acceleration check
        first_price = closes[0]
        last_price = closes[-1]
        price_change = ((last_price - first_price) / first_price) * 100
        
        if direction.lower() == "long":
            return price_change > 0.5
        else:
            return price_change < -0.5
        
    except Exception as e:
        return False


async def get_core_confirmations(symbol: str, core_candles: Dict, direction: str, trend_context: Dict) -> List[str]:
    """Get confirmations specific to core strategy"""
    confirmations = []
    
    try:
        # 1. Multi-timeframe momentum alignment
        if validate_momentum_alignment(core_candles, direction):
            confirmations.append("momentum_alignment")
        
        # 2. Volume breakout confirmation
        if validate_volume_breakout(core_candles):
            confirmations.append("volume_breakout")
        
        # 3. Trend strength confirmation
        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))
        if trend_strength > 0.5:  # lowered from 0.6 → 0.5 to match real downtrend strength ~0.37
            confirmations.append("strong_trend")
        
        # 4. Price level confirmation
        if validate_price_levels(core_candles, direction):
            confirmations.append("price_levels")
        
        # 5. Trend coherence
        if validate_trend_coherence(core_candles, direction, trend_context):
            confirmations.append("trend_coherence")
        
        # 6. Check for pump potential
        if has_pump_potential(core_candles, direction):
            confirmations.append("pump_potential")
        
    except Exception as e:
        log(f"❌ Error getting confirmations for {symbol}: {e}", level="ERROR")
    
    return confirmations

# <<< SCALP HUNTER >>> — New async function, add to main.py

async def scalp_hunter_scan(symbols: List[str], trend_context: Dict) -> None:
    """
    1% Scalp Hunter scan loop.
    Called from core_strategy_scan() after core strategy processing.
    Evaluates every symbol for compression → expansion setups.
    """
    if not SCALP_HUNTER_ENABLED:
        return

    cfg = SCALP_CONFIG

    # ── Update BTC state first (needed for alt filtering) ──────────
    btc_candles_1m = list(live_candles.get("BTCUSDT", {}).get("1", []))
    btc_candles_5m = list(live_candles.get("BTCUSDT", {}).get("5", []))
    if btc_candles_1m:
        update_btc_state(btc_candles_1m, btc_candles_5m)

    # Count current scalp positions
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
            # ── Get candles ────────────────────────────────────────
            sym_candles = live_candles.get(symbol, {})
            candles_by_tf = {
                tf: list(sym_candles.get(tf, []))
                for tf in ["1", "3", "5"]
                if sym_candles.get(tf)
            }
            if not candles_by_tf:
                continue

            # Need at least one TF with sufficient candles
            has_enough = any(len(v) >= 30 for v in candles_by_tf.values())
            if not has_enough:
                continue

            # Get current price
            primary_tf = "1" if "1" in candles_by_tf else list(candles_by_tf.keys())[0]
            last_candle = candles_by_tf[primary_tf][-1]
            current_price = float(last_candle.get("close", 0))
            if not current_price:
                continue

            # ── Signal cooldown check ──────────────────────────────
            last_signal_ts = scalp_signal_cooldown.get(symbol, 0)
            if time.time() - last_signal_ts < SCALP_COOLDOWN_SECONDS:
                continue

            # ── Skip if already in active trade ───────────────────
            if symbol in active_trades and not active_trades[symbol].get("exited", False):
                continue

            # ── Evaluate scalp setup ───────────────────────────────
            result = evaluate_scalp_setup(symbol, candles_by_tf, current_price)

            # ── Update watchlist ───────────────────────────────────
            result = update_watchlist(symbol, result)

            # ── Process result ─────────────────────────────────────
            if result["should_trade"]:
                await _fire_scalp_signal(symbol, result, candles_by_tf, current_price)
                # Yield after signal to avoid blocking
                await asyncio.sleep(0.1)

            processed += 1
            # Small yield every 20 symbols to stay async-friendly
            if processed % 20 == 0:
                await asyncio.sleep(0.01)

        except Exception as e:
            log(f"⚠️ SCALP HUNTER: Error evaluating {symbol}: {e}", level="WARN")
            continue


async def _fire_scalp_signal(
    symbol: str,
    result: Dict,
    candles_by_tf: Dict,
    current_price: float,
) -> None:
    """
    Send Telegram signal and optionally place order for scalp setup.
    """
    direction = result["direction"]
    sl_price = result["sl_price"]
    tp1_price = result["tp1_price"]
    sl_pct = result["sl_pct"]
    tp1_pct = result["tp1_pct"]
    score = result["score"]
    confidence = result["confidence"]
    reasons = result["reasons"]
    details = result["details"]

    # ── Duplicate guard ──────────────────────────────────────────
    if is_duplicate_signal(symbol):
        log(f"🔁 SCALP HUNTER: Duplicate signal blocked for {symbol}")
        return

    log(f"🎯 SCALP HUNTER SIGNAL: {symbol} {direction.upper()} | Score={score} | Conf={confidence:.1f}%")

    # ── Format and send Telegram message ─────────────────────────
    msg = format_scalp_signal(
        symbol=symbol,
        direction=direction,
        entry_price=current_price,
        sl_price=sl_price,
        tp1_price=tp1_price,
        sl_pct=sl_pct,
        tp1_pct=tp1_pct,
        score=score,
        confidence=confidence,
        reasons=reasons,
        details=details,
        btc_state=_btc_state.get("trend", "unknown"),
    )
    await send_telegram_message(msg)

    # Record signal to prevent duplicates + cooldown
    log_signal(symbol)
    scalp_signal_cooldown[symbol] = time.time()

    # ── Place trade (if not signals-only mode) ────────────────────
    if not SCALP_HUNTER_SIGNALS_ONLY:
        signal_data = {
            "symbol": symbol,
            "direction": direction.capitalize(),
            "strategy": "ScalpHunter",
            "trade_type": "ScalpHunter",
            "score": float(score),
            "confidence": confidence,
            "regime": "scalp_hunter",
            "sl_price": sl_price,
            "tp1_price": tp1_price,
            "sl_pct": sl_pct,
            "tp1_pct": tp1_pct,
            "trailing_pct": 0.5,          # 0.5% trailing after TP1
            "tp1_partial_close": SCALP_CONFIG["tp1_partial_close"],
            "leverage": SCALP_CONFIG["leverage"],
            "candles": candles_by_tf,
            "is_scalp_hunter": True,       # Flag for executor
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


async def core_strategy_scan(symbols: List[str], trend_context: Dict):
    """
    PURE CORE STRATEGY - Single focused trading approach.
    Only the most reliable signals with strict quality filters.
    """
    source = fix_live_candles_structure(live_candles)
    
    try:
        if not symbols or len(symbols) == 0:
            log("⚠️ CORE STRATEGY: No symbols to scan", level="WARN")
            return

        # Check position limits first
        current_positions = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        if current_positions >= MAX_CORE_POSITIONS:
            log(f"🚫 CORE STRATEGY: Max positions reached ({current_positions}/{MAX_CORE_POSITIONS})")
            return

        # Enhanced system stores under "strength", legacy uses "trend_strength"
        trend_strength = trend_context.get("strength", trend_context.get("trend_strength", 0.5))
        trend_direction = trend_context.get("trend", "neutral")
        log(f"📊 Trend strength resolved: {trend_strength:.2f} (raw keys: strength={trend_context.get('strength', 'N/A')}, trend_strength={trend_context.get('trend_strength', 'N/A')})")
        # Don't block trading on WAIT_AND_SEE unless opportunity is very low
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
                # Check if we can process this symbol
                can_process, reason = await trade_lock_manager.can_process_symbol(symbol)
                if not can_process:
                    continue

                # Acquire lock
                if not await trade_lock_manager.acquire_trade_lock(symbol):
                    continue

                try:
                    # Skip if already in active trade
                    if symbol in active_trades and not active_trades[symbol].get("exited", False):
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # Skip if in cooldown
                    if symbol in signal_cooldown:
                        time_diff = time.time() - signal_cooldown[symbol]
                        if time_diff < SIGNAL_COOLDOWN_TIME:
                            trade_lock_manager.release_trade_lock(symbol, False)
                            continue

                    # Skip if in recent exit cooldown
                    if symbol in recent_exits:
                        time_diff = time.time() - recent_exits[symbol]
                        if time_diff < EXIT_COOLDOWN:
                            trade_lock_manager.release_trade_lock(symbol, False)
                            continue

                    # Get candles for core timeframes
                    core_candles = {}
                    src = source.get(symbol, {})
                    
                    for tf in ['1', '5', '15']:
                        tf_data = src.get(tf)
                        if tf_data:
                            candles = list(tf_data)
                            min_needed = 12 if tf in ('1', '5') else 8
                            if len(candles) >= min_needed:
                                core_candles[tf] = candles

                    # Must have at least 1m and 5m to proceed
                    if not all(tf in core_candles for tf in ('1', '5')):
                        # --- ADD THIS ---
                        missing = [tf for tf in ('1', '5') if tf not in core_candles]
                        available = {tf: len(list(src.get(tf, []))) for tf in ['1', '5', '15']}
                        log(f"⏭️ {symbol}: Missing TFs {missing} | candle counts: {available}", level="DEBUG")
                        # ----------------
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    scanned_count += 1

                    # === CORE STRATEGY SIGNAL GENERATION ===
                    
                    # 1. Get full scoring data
                    score_result = score_symbol(symbol, core_candles, trend_context)
                    score, tf_scores, trade_type, indicator_scores, used_indicators = score_result

                    # 2. Calculate core score with momentum bonus
                    core_score = await calculate_core_score(symbol, core_candles, trend_context)
                    if core_score < MIN_SCALP_SCORE:
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 3. Determine direction
                    direction = determine_core_direction(core_candles, trend_context)
                    if not direction:
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 4. Calculate confidence
                    confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                    if confidence < 60:
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 5. Validate core strategy conditions
                    if not await validate_core_conditions(symbol, core_candles, direction, trend_context):
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 6. Determine strategy type
                    strategy_type = determine_core_strategy_type(core_score, confidence, trend_strength)
                    if not strategy_type:
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 7. Check strategy-specific position limits
                    if not check_strategy_position_limits(strategy_type):
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 8. Get confirmations
                    confirmations = await get_core_confirmations(symbol, core_candles, direction, trend_context)
                    if len(confirmations) < 2:
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # 9. Check for duplicate signal
                    if is_duplicate_signal(symbol):
                        trade_lock_manager.release_trade_lock(symbol, False)
                        continue

                    # === SIGNAL VALIDATED - EXECUTE TRADE ===
                    core_signals_found += 1
                    
                    log(f"🎯 CORE STRATEGY SIGNAL: {symbol}")
                    log(f"   Score: {core_score:.1f} | Confidence: {confidence}%")
                    log(f"   Direction: {direction} | Type: {strategy_type}")
                    log(f"   Confirmations: {', '.join(confirmations)}")

                    # Execute trade
                    trade_result = await execute_core_trade(
                        symbol=symbol,
                        direction=direction,
                        strategy_type=strategy_type,
                        score=core_score,
                        confidence=confidence,
                        confirmations=confirmations,
                        core_candles=core_candles,
                        trend_context=trend_context
                    )

                    if trade_result and trade_result.get("success"):
                        signal_cooldown[symbol] = time.time()
                        trade_lock_manager.release_trade_lock(symbol, True)
                    else:
                        # Shorter cooldown on failure
                        signal_cooldown[symbol] = time.time() - (SIGNAL_COOLDOWN_TIME * 0.8)
                        trade_lock_manager.release_trade_lock(symbol, False)

                except Exception as e:
                    log(f"❌ CORE STRATEGY: Error processing {symbol}: {e}", level="ERROR")
                    trade_lock_manager.release_trade_lock(symbol, False)
                    continue

            except Exception as e:
                log(f"❌ CORE STRATEGY: Error with {symbol}: {e}", level="ERROR")
                continue

        log(f"📊 CORE STRATEGY SUMMARY: {scanned_count} scanned, {core_signals_found} quality signals")

                         # <<< SCALP HUNTER >>> — Run scalp hunter scan after core strategy
        if SCALP_HUNTER_ENABLED:
            await scalp_hunter_scan(symbols, trend_context)

    except Exception as e:
        log(f"❌ CORE STRATEGY: Error in scan: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")


async def execute_core_trade(
    symbol: str,
    direction: str,
    strategy_type: str,
    score: float,
    confidence: float,
    confirmations: List[str],
    core_candles: Dict,
    trend_context: Dict
) -> Dict[str, Any]:
    """Execute a core strategy trade"""
    try:
        # Get risk percentage for strategy type
        base_risk = CORE_RISK_PERCENTAGES.get(strategy_type, 0.02)
        
        # Adjust risk based on confidence
        if confidence >= 85:
            adjusted_risk = base_risk * 1.2
        elif confidence >= 75:
            adjusted_risk = base_risk
        else:
            adjusted_risk = base_risk * 0.8
        
        # Cap risk at 3%
        adjusted_risk = min(adjusted_risk, 0.03)

        # Guard: don't open if position already exists
        if symbol in active_trades and not active_trades[symbol].get("exited", False):
            log(f"🚫 {symbol}: Trade execution blocked - position already exists")
            return {"success": False, "reason": "position_exists"}

        # Prepare signal data
        signal_data = {
            "symbol": symbol,
            "direction": direction,
            "strategy": f"CORE_{strategy_type}",
            "score": score,
            "confidence": confidence,
            "regime": "core_strategy",
            "confirmations": confirmations,
            "risk_adjusted": True,
            "core_strategy": True,
            "candles": core_candles,
            "trade_type": strategy_type.replace("Core", "")  # "CoreScalp" -> "Scalp"
        }

        log(f"🎯 CORE STRATEGY EXECUTION: {symbol}")
        log(f"   Direction: {direction} | Type: {strategy_type}")
        log(f"   Score: {score:.1f} | Confidence: {confidence}%")
        log(f"   Confirmations: {', '.join(confirmations) if confirmations else '—'}")
        log(f"   Risk: {adjusted_risk:.2%}")

        # Execute trade
        trade_result = await execute_trade_if_valid(signal_data, adjusted_risk)

        if trade_result and trade_result.get("success"):
            # Track signal
            log_signal(symbol)
            track_signal(symbol, score)

            # Send notification
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


async def core_monitor_loop():
    """
    Core strategy status logger only.
    
    NOTE: All exit logic is handled by monitor.py -> unified_exit_manager.py
    This function only logs trade status for visibility.
    DO NOT add exit logic here to avoid duplication!
    """
    log("🔍 Starting core_monitor_loop (status logging only)...")

    await asyncio.sleep(15)
    
    while True:
        try:
            # Get active (non-exited) trades
            active = {k: v for k, v in active_trades.items() if not v.get("exited", False)}
            
            if not active:
                await asyncio.sleep(30)  # Less frequent when no trades
                continue
            
            # Just log status - no exit logic here!
            log(f"📊 CORE STATUS: {len(active)} active trades")
            
            for symbol, trade in list(active.items()):
                try:
                    pnl = trade.get("current_pnl_pct", 0)
                    tp1_hit = "✅" if trade.get("tp1_hit") else "⏳"
                    log(f"   {symbol}: P&L={pnl:+.2f}% | TP1={tp1_hit}")
                except:
                    pass
            
            await asyncio.sleep(30)  # Status update every 30 seconds
            
        except Exception as e:
            log(f"❌ CORE STATUS: Error in status loop: {e}", level="ERROR")
            await asyncio.sleep(60)


# NOTE: monitor_core_trade() REMOVED
# All exit logic is handled by:
#   monitor.py -> monitor_active_trades() -> unified_exit_manager.process_trade_exits()
# This prevents duplicate exit logic and keeps the code clean.
# 
# The exit flow is:
#   1. monitor_active_trades() gets current price
#   2. Calls unified_exit_manager.process_trade_exits()
#   3. unified_exit_manager handles: SL check, TP1 trigger, breakeven, trailing
#   4. If exit triggered, closes position on exchange


async def run_core_bot():
    """Core strategy bot - simplified and focused"""
    log("🚀 CORE STRATEGY BOT starting...")
    
    # Initialize
    await fetch_symbol_info()
    symbols = await fetch_symbols()
    log(f"✅ CORE STRATEGY: Fetched {len(symbols)} symbols.")
    
    # Load active trades
    load_active_trades()
    
    # Sync with exchange
    await sync_bot_with_bybit(send_telegram=True)
    
    # Recover trades if needed
    if len(active_trades) == 0:
        await recover_active_trades_from_exchange()
    
    # Start background tasks
    asyncio.create_task(stream_candles(symbols))
    asyncio.create_task(core_monitor_loop())
    asyncio.create_task(monitor_active_trades())  # From monitor.py
    asyncio.create_task(monitor_btc_trend_accuracy())
    asyncio.create_task(monitor_altseason_status())
    asyncio.create_task(periodic_trade_sync())
    asyncio.create_task(bybit_sync_loop(120))
    asyncio.create_task(lock_manager_maintenance())

    # Wait for initial data
    await asyncio.sleep(5)
    
    log("🚀 CORE STRATEGY BOT fully initialized - starting main loop")

    # Main loop
    while True:
        try:
            # Get trend context
            trend_context = await get_trend_context_cached()
            
            # Run core strategy scan
            await core_strategy_scan(symbols, trend_context)
            
            # Send daily report if due
            await send_daily_report()
            
        except Exception as e:
            log(f"❌ CORE STRATEGY: Error in main loop: {e}", level="ERROR")
            write_log(f"CORE STRATEGY MAIN LOOP ERROR: {str(e)}", level="ERROR")
            await send_error_to_telegram(traceback.format_exc())
        
        # Scan interval
        await asyncio.sleep(1.0)
        
        # Sync lock manager
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
        log(f"🔗 live_candles id(main)={id(live_candles)} | id(ws)={id(ws_live) if ws_live else 'N/A'}")
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
# === ENTRY POINT ===
if __name__ == "__main__":
    import sys
    import platform

    log("🔧 DEBUG: CORE STRATEGY main.py is running...")
    log(f"🔍 CORE STRATEGY thresholds - Scalp: {MIN_SCALP_SCORE}, Intraday: {MIN_INTRADAY_SCORE}, Swing: {MIN_SWING_SCORE}")
    log(f"🔒 CORE STRATEGY limits - Max positions: {MAX_CORE_POSITIONS}, Scalp: {MAX_SCALP_POSITIONS}, Intraday: {MAX_INTRADAY_POSITIONS}, Swing: {MAX_SWING_POSITIONS}")

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

    # ── Windows fix ──────────────────────────────────────────────────────────
    # On Windows, asyncio defaults to ProactorEventLoop which raises
    # "ConnectionResetError: [WinError 10054]" as a noisy but harmless
    # callback error whenever a remote host closes a TCP connection abruptly.
    # Switching to SelectorEventLoop eliminates this entirely.
    # On Linux/Mac this block is skipped — no change in behaviour there.
    if platform.system() == "Windows":
        # 1. Use SelectorEventLoop instead of ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # 2. Suppress the residual WinError 10054 that may still slip through
        #    during shutdown of existing connections on startup.
        def _silence_connection_reset(loop, context):
            exception = context.get("exception")
            if isinstance(exception, ConnectionResetError):
                # Harmless: remote host closed the connection — ignore silently
                return
            # All other exceptions go through the default handler
            loop.default_exception_handler(context)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(_silence_connection_reset)

        try:
            loop.run_until_complete(restart_forever())
        finally:
            loop.close()
    else:
        # Linux / Mac — run normally, no changes needed
        asyncio.run(restart_forever())









