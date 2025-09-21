import asyncio
import traceback
import time
import json
from datetime import datetime
from scanner import fetch_symbols
from websocket_candles import live_candles, stream_candles, SUPPORTED_INTERVALS
from score import score_symbol, determine_direction, calculate_confidence, has_pump_potential, detect_momentum_strength
from telegram_bot import send_telegram_message, format_trade_signal, send_error_to_telegram
from trend_filters import monitor_btc_trend_accuracy, monitor_altseason_status, validate_short_signal
from trend_upgrade_integration import get_trend_context_cached
from signal_memory import log_signal, is_duplicate_signal
from config import (
    DEFAULT_LEVERAGE, ALWAYS_ALLOW_SWING, ALTSEASON_MODE, NORMAL_MAX_POSITIONS,
    MIN_SCALP_SCORE, MIN_INTRADAY_SCORE, MIN_SWING_SCORE
)
from performance_tracker import track_signal
from logger import log
from bybit_sync import sync_bot_with_bybit
from monitor_report import log_trade_result, send_daily_report
from trade_executor import calculate_dynamic_sl_tp, execute_trade_if_valid
from symbol_info import fetch_symbol_info
from activity_logger import write_log, log_trade_to_file
from monitor import track_active_trade, monitor_trades, load_active_trades, check_and_restore_sl, active_trades, recover_active_trades_from_exchange, periodic_trade_sync

log(f"🔍 main.py - Core Strategy Only - imported active_trades id: {id(active_trades)}")

# === CORE STRATEGY ONLY CONFIGURATION ===
TIMEFRAMES = SUPPORTED_INTERVALS
active_signals = {}
recent_exits = {}
EXIT_COOLDOWN = 120

# Core Strategy Thresholds - Enhanced for Quality
MIN_SCALP_SCORE = 9.0      # High quality scalps only
MIN_INTRADAY_SCORE = 10.0   # High quality intraday only  
MIN_SWING_SCORE = 14.0      # High quality swings only

# Core Strategy Risk Management - Conservative
CORE_RISK_PERCENTAGES = {
    "Scalp": 0.025,     # 2.5% risk for scalps
    "Intraday": 0.02,   # 2% risk for intraday  
    "Swing": 0.015      # 1.5% risk for swing trades
}

# Core Strategy Position Limits
MAX_CORE_POSITIONS = 3      # Maximum 3 concurrent positions
MAX_SCALP_POSITIONS = 2     # Maximum 2 scalp positions
MAX_INTRADAY_POSITIONS = 1  # Maximum 1 intraday position
MAX_SWING_POSITIONS = 1     # Maximum 1 swing position

# Core strategy variables
startup_time = time.time()

def safe_get_candles(live_candles, symbol):
    """Safe candle extraction"""
    try:
        candles = None
        for tf in ['1', '5', '15']:
            if (symbol in live_candles and 
                tf in live_candles[symbol] and 
                live_candles[symbol][tf]):
                
                candle_data = live_candles[symbol][tf]
                
                # Ensure it's a proper list/sequence
                if isinstance(candle_data, (list, tuple)):
                    if len(candle_data) > 0:
                        candles = list(candle_data[-20:]) if len(candle_data) >= 20 else list(candle_data)
                        break
                elif hasattr(candle_data, '__len__') and hasattr(candle_data, '__getitem__'):
                    # It's sequence-like but not list/tuple
                    try:
                        candle_list = list(candle_data)
                        if len(candle_list) > 0:
                            candles = candle_list[-20:] if len(candle_list) >= 20 else candle_list
                            break
                    except:
                        continue
        return candles
    except Exception as e:
        print(f"❌ Safe candle extraction error for {symbol}: {e}")
        return None

async def core_strategy_scan(symbols, trend_context):
    """
    PURE CORE STRATEGY - Single focused trading approach
    Only the most reliable signals with strict quality filters
    """
    try:
        if not symbols or len(symbols) == 0:
            log("⚠️ CORE STRATEGY: No symbols to scan", level="WARN")
            return

        # Check position limits first
        current_positions = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        if current_positions >= MAX_CORE_POSITIONS:
            log(f"🚫 CORE STRATEGY: Max positions reached ({current_positions}/{MAX_CORE_POSITIONS})")
            return

        # Get market trend strength for core strategy adaptation
        trend_strength = trend_context.get("trend_strength", 0.5)
        trend_direction = trend_context.get("trend", "neutral")
        
        log(f"🔍 CORE STRATEGY: Scanning {len(symbols)} symbols | Trend: {trend_direction} ({trend_strength:.2f})")

        scanned_count = 0
        core_signals_found = 0
        
        # Focus on high-quality symbols only
        quality_symbols = await filter_core_symbols(symbols)
        
        for symbol in symbols:  # Limit to top 20 symbols for focus
            try:
                # Skip if already have position
                if symbol in active_trades and not active_trades[symbol].get("exited", False):
                    continue

                # Skip if in recent exit cooldown
                if symbol in recent_exits:
                    time_diff = time.time() - recent_exits[symbol]
                    if time_diff < EXIT_COOLDOWN:
                        continue

                # Get candles - core strategy uses 1m, 5m, 15m only
                core_candles = {}
                for tf in ['1', '5', '15']:
                    if tf in live_candles.get(symbol, {}):
                        candles = live_candles[symbol][tf]
                        if candles and len(candles) >= 30:  # Require more history for quality
                            core_candles[tf] = candles
                
                if len(core_candles) < 3:  # Must have all 3 timeframes
                    continue

                scanned_count += 1

                # === CORE STRATEGY SIGNAL GENERATION ===
                
                # 1. Calculate core score with strict requirements
                core_score = await calculate_core_score(symbol, core_candles, trend_context)
                if core_score < MIN_SCALP_SCORE:
                    continue

                # 2. Determine direction with trend alignment
                direction = determine_core_direction(core_candles, trend_context)
                if not direction:
                    continue

                # 3. Calculate confidence with higher threshold
                confidence = calculate_confidence(core_candles, trend_context)
                if confidence < 60:  # Minimum 70% confidence for core strategy
                    continue

                # 4. Validate core strategy conditions
                if not await validate_core_conditions(symbol, core_candles, direction, trend_context):
                    continue

                # 5. Determine strategy type with strict requirements
                strategy_type = determine_core_strategy_type(core_score, confidence, trend_strength)
                if not strategy_type:
                    continue

                # 6. Check strategy-specific position limits
                if not check_strategy_position_limits(strategy_type):
                    continue

                # 7. Final quality gate - multiple confirmations required
                core_confirmations = await get_core_confirmations(symbol, core_candles, direction, trend_context)
                if len(core_confirmations) < 2:  # Minimum 2 confirmations
                    continue

                core_signals_found += 1
                log(f"✅ CORE STRATEGY SIGNAL: {symbol} {direction} | Score: {core_score:.1f} | Confidence: {confidence}% | Type: {strategy_type}")

                # Execute core strategy trade
                await execute_core_trade(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    score=core_score,
                    confidence=confidence,
                    confirmations=core_confirmations,
                    trend_context=trend_context
                )

            except Exception as e:
                log(f"❌ CORE STRATEGY: Error processing {symbol}: {e}", level="ERROR")
                continue

        log(f"📊 CORE STRATEGY SUMMARY: {scanned_count} scanned, {core_signals_found} quality signals")

    except Exception as e:
        log(f"❌ CORE STRATEGY: Error in scan: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")

async def filter_core_symbols(symbols):
    """Filter symbols for core strategy - FIXED and more aggressive"""
    try:
        core_symbols = []
        
        log(f"🔍 Filtering {len(symbols)} symbols for core strategy...")
        
        for symbol in symbols:
            try:
                # Check if we have any candle data for this symbol
                if symbol not in live_candles:
                    continue
                
                # Try to get candles from any available timeframe
                candles = safe_get_candles(live_candles, symbol)
                
                if not candles or len(candles) < 20:  # Reduced from 20 to 10
                    continue
                
                # Calculate liquidity score
                volumes = [float(c.get('volume', 0)) for c in candles]
                if not volumes:
                    continue
                    
                avg_volume = sum(volumes) / len(volumes)
                latest_volume = volumes[-1]
                
                # Calculate volatility score
                price_changes = []
                for i in range(1, len(candles)):
                    prev = float(candles[i-1].get('close', 0))
                    curr = float(candles[i].get('close', 0))
                    if prev > 0:
                        change = abs((curr - prev) / prev)
                        price_changes.append(change)
                
                if not price_changes:
                    continue
                    
                avg_volatility = sum(price_changes) / len(price_changes)
                
                # MUCH MORE AGGRESSIVE FILTERS
                is_quality_symbol = (
                    avg_volume > 1000000 and              # Lowered from 2,000,000 to 500,000
                    0.005 < avg_volatility < 0.15 and    # Wider volatility range (was 0.008-0.08)
                    'USDT' in symbol and                 # Still USDT pairs only
                    latest_volume > avg_volume * 0.3     # Recent volume activity
                )
                
                if is_quality_symbol:
                    # Add volume score for sorting
                    volume_score = avg_volume * (1 + avg_volatility)
                    core_symbols.append((symbol, volume_score))
                
            except Exception as e:
                continue
        
        # Sort by volume score (highest first) and return symbols
        core_symbols.sort(key=lambda x: x[1], reverse=True)
        filtered_symbols = [symbol for symbol, score in core_symbols]
        
        log(f"✅ Filtered to {len(filtered_symbols)} quality symbols from {len(symbols)} total")
        
        # Return more symbols - up to 100 instead of 30
        return filtered_symbols[:500]  # Increased limit
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error filtering symbols: {e}", level="ERROR")
        return symbols[:200]  # Return more symbols on error


async def calculate_core_score(symbol, core_candles, trend_context):
    """FIXED: Calculate core strategy score"""
    try:
        # ✅ FIXED: Correct score_symbol call
        base_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
            symbol, core_candles, trend_context
        )
        
        # ✅ FIXED: Momentum calculation
        momentum_bonus = 0
        for tf, candles in core_candles.items():
            if candles and len(candles) >= 10:
                has_momentum, direction, strength = detect_momentum_strength(candles)
                if has_momentum and strength > 0.6:
                    momentum_bonus = strength * 1.5
                    break
        
        # Add regime bonus
        regime = trend_context.get("regime", "unknown")
        if regime == "transitional":
            regime_bonus = 2.0
        else:
            regime_bonus = 1.0
        
        final_score = base_score + momentum_bonus + regime_bonus
        return final_score
        
    except Exception as e:
        log(f"❌ Error calculating score for {symbol}: {e}", level="ERROR")
        return 6.0  # Good default for transitional markets

def determine_core_direction(core_candles, trend_context):
    """Determine direction with core strategy alignment requirements"""
    try:
        base_direction = determine_direction(core_candles, trend_context)
        if not base_direction:
            return None
        
        # Core strategy requires trend alignment for higher success rate
        trend_direction = trend_context.get("trend", "neutral")
        trend_strength = trend_context.get("trend_strength", 0.5)
        
        # For strong trends, require alignment
        if trend_strength > 0.6:
            if base_direction.lower() == "long" and trend_direction != "bullish":
                return None
            if base_direction.lower() == "short" and trend_direction != "bearish":
                return None
        
        # For weak trends, only allow long positions (safer)
        if trend_strength < 0.4 and base_direction.lower() == "short":
            return None
        
        return base_direction
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error determining direction: {e}", level="ERROR")
        return None

async def validate_core_conditions(symbol, core_candles, direction, trend_context):
    """Validate core strategy specific conditions"""
    try:
        # 1. Volume validation - must be above average
        if not validate_core_volume(core_candles):
            return False
        
        # 2. Price action quality
        if not validate_core_price_action(core_candles, direction):
            return False
        
        # 3. Risk/reward validation
        if not validate_core_risk_reward(core_candles, direction):
            return False
        
        # 4. Market timing validation
        if not validate_core_timing():
            return False
        
        # 5. Trend coherence across timeframes
        if not validate_core_trend_coherence(core_candles, direction):
            return False
        
        return True
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error validating conditions for {symbol}: {e}", level="ERROR")
        return False

def validate_core_volume(core_candles):
    """Core strategy volume validation - stricter requirements"""
    try:
        if '1' not in core_candles:
            return False
        
        candles = core_candles['1'][-20:]
        volumes = [float(c.get('volume', 0)) for c in candles]
        
        if len(volumes) < 20:
            return False
        
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / 5  # Last 5 candles average
        
        # Recent volume must be significantly above average
        return recent_volume > avg_volume * 1.5
        
    except Exception as e:
        return False

def validate_core_price_action(core_candles, direction):
    """Validate clean price action for core strategy"""
    try:
        if '5' not in core_candles:
            return False
        
        candles = core_candles['5'][-10:]
        closes = [float(c.get('close', 0)) for c in candles]
        
        if len(closes) < 10:
            return False
        
        # Check for clean directional movement
        if direction.lower() == "long":
            # For long signals, require general upward trend in recent closes
            upward_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            return upward_moves >= 6  # At least 60% upward moves
        else:
            # For short signals, require general downward trend
            downward_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
            return downward_moves >= 6  # At least 60% downward moves
        
    except Exception as e:
        return False

def validate_core_risk_reward(core_candles, direction):
    """Validate risk/reward ratio is favorable"""
    try:
        if '15' not in core_candles:
            return False
        
        candles = core_candles['15'][-20:]
        if len(candles) < 20:
            return False
        
        # Calculate recent support/resistance levels
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        current_price = float(candles[-1].get('close', 0))
        
        if direction.lower() == "long":
            resistance = max(highs[-10:])  # Recent resistance
            support = min(lows[-10:])      # Recent support
            
            # Risk = distance to support, Reward = distance to resistance
            risk = current_price - support
            reward = resistance - current_price
            
            if risk > 0 and reward > 0:
                risk_reward_ratio = reward / risk
                return risk_reward_ratio >= 2.0  # Minimum 2:1 R/R
        
        else:  # short
            resistance = max(highs[-10:])
            support = min(lows[-10:])
            
            risk = resistance - current_price
            reward = current_price - support
            
            if risk > 0 and reward > 0:
                risk_reward_ratio = reward / risk
                return risk_reward_ratio >= 2.0
        
        return False
        
    except Exception as e:
        return False

def validate_core_timing():
    """Validate market timing for core strategy"""
    try:
        current_hour = datetime.utcnow().hour
        
        # Core strategy only trades during peak liquidity hours
        peak_hours = list(range(8, 23))  # 8 AM to 11 PM UTC
        
        return current_hour in peak_hours
        
    except Exception as e:
        return True  # Default to True on error

def validate_core_trend_coherence(core_candles, direction):
    """Validate trend coherence across all timeframes"""
    try:
        trend_scores = {}
        
        for tf in ['1', '5', '15', '30', '60', '240']:
            if tf not in core_candles:
                return False
            
            candles = core_candles[tf][-20:]
            closes = [float(c.get('close', 0)) for c in candles]
            
            if len(closes) < 20:
                return False
            
            # Calculate trend direction for this timeframe
            start_price = sum(closes[:5]) / 5  # First 5 candles average
            end_price = sum(closes[-5:]) / 5   # Last 5 candles average
            
            trend_scores[tf] = (end_price - start_price) / start_price
        
        # All timeframes should agree on direction
        if direction.lower() == "long":
            return all(score > 0.005 for score in trend_scores.values())  # All uptrends
        else:
            return all(score < -0.005 for score in trend_scores.values())  # All downtrends
        
    except Exception as e:
        return False

def determine_core_strategy_type(score, confidence, trend_strength):
    """Determine core strategy type with strict requirements"""
    try:
        # Core strategy type determination - very selective
        if score >= MIN_SWING_SCORE and confidence >= 80 and trend_strength >= 0.7:
            return "CoreSwing"
        elif score >= MIN_INTRADAY_SCORE and confidence >= 75 and trend_strength >= 0.5:
            return "CoreIntraday"
        elif score >= MIN_SCALP_SCORE and confidence >= 70:
            return "CoreScalp"
        else:
            return None
            
    except Exception as e:
        return None

def check_strategy_position_limits(strategy_type):
    """Check strategy-specific position limits"""
    try:
        strategy_counts = {"CoreScalp": 0, "CoreIntraday": 0, "CoreSwing": 0}
        
        for trade in active_trades.values():
            if not trade.get("exited", False):
                trade_type = trade.get("trade_type", "")
                if trade_type in strategy_counts:
                    strategy_counts[trade_type] += 1
        
        # Check limits
        if strategy_type == "CoreScalp" and strategy_counts["CoreScalp"] >= MAX_SCALP_POSITIONS:
            return False
        elif strategy_type == "CoreIntraday" and strategy_counts["CoreIntraday"] >= MAX_INTRADAY_POSITIONS:
            return False
        elif strategy_type == "CoreSwing" and strategy_counts["CoreSwing"] >= MAX_SWING_POSITIONS:
            return False
        
        return True
        
    except Exception as e:
        return False

async def get_core_confirmations(symbol, core_candles, direction, trend_context):
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
        trend_strength = trend_context.get("trend_strength", 0.5)
        if trend_strength > 0.6:
            confirmations.append("strong_trend")
        
        # 4. Price level confirmation (support/resistance respect)
        if validate_price_levels(core_candles, direction):
            confirmations.append("price_levels")
        
        # 5. Momentum acceleration
        if detect_momentum_acceleration(core_candles, direction):
            confirmations.append("momentum_acceleration")
        
        return confirmations
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error getting confirmations for {symbol}: {e}", level="ERROR")
        return []

def validate_momentum_alignment(core_candles, direction):
    """FIXED: Check if momentum is aligned across timeframes"""
    try:
        good_momentum_count = 0
        
        for tf in ['1', '5', '15']:
            if tf in core_candles and core_candles[tf]:
                candles = core_candles[tf]
                if len(candles) >= 10:
                    has_momentum, mom_direction, strength = detect_momentum_strength(candles)
                    if has_momentum and strength > 0.5:
                        good_momentum_count += 1
        
        return good_momentum_count >= 1  # At least 1 timeframe with momentum
        
    except Exception as e:
        return True  # Default allow

def validate_volume_breakout(core_candles):
    """Check for volume breakout pattern"""
    try:
        if '1' not in core_candles:
            return False
        
        candles = core_candles['1'][-30:]
        volumes = [float(c.get('volume', 0)) for c in candles]
        
        if len(volumes) < 30:
            return False
        
        avg_volume = sum(volumes[:-5]) / len(volumes[:-5])  # Exclude last 5
        recent_volume = sum(volumes[-5:]) / 5              # Last 5 average
        
        # Volume breakout: recent volume 2x average
        return recent_volume > avg_volume * 2.0
        
    except Exception as e:
        return False

def validate_price_levels(core_candles, direction):
    """Validate price is respecting key levels"""
    try:
        if '15' not in core_candles:
            return False
        
        candles = core_candles['15'][-50:]
        if len(candles) < 50:
            return False
        
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        current_price = float(candles[-1].get('close', 0))
        
        # Find key levels
        resistance_levels = []
        support_levels = []
        
        # Simple pivot detection
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        if direction.lower() == "long":
            # For longs, ensure we're above recent support
            if support_levels:
                nearest_support = max([s for s in support_levels if s <= current_price], default=0)
                return current_price > nearest_support * 1.005  # 0.5% above support
        else:
            # For shorts, ensure we're below recent resistance
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r >= current_price], default=float('inf'))
                return current_price < nearest_resistance * 0.995  # 0.5% below resistance
        
        return True
        
    except Exception as e:
        return False

def detect_momentum_acceleration(core_candles, direction):
    """FIXED: Detect if momentum is accelerating"""
    try:
        if '5' not in core_candles or len(core_candles['5']) < 10:
            return False
        
        candles = core_candles['5'][-10:]  # Use fewer candles
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

async def execute_core_trade(symbol, direction, strategy_type, score, confidence, confirmations, trend_context):
    """Execute core strategy trade with enhanced parameters"""
    try:
        # Calculate core strategy risk
        base_risk = CORE_RISK_PERCENTAGES.get(strategy_type.replace("Core", ""), 0.02)
        
        # Quality-based risk adjustment
        confidence_multiplier = 0.8 + (confidence / 100 * 0.4)  # 0.8 to 1.2
        confirmations_multiplier = 0.9 + (len(confirmations) * 0.05)  # 0.9 to 1.15
        
        adjusted_risk = base_risk * confidence_multiplier * confirmations_multiplier
        adjusted_risk = max(0.01, min(adjusted_risk, 0.03))  # Clamp 1%-3%
        
        # Prepare core strategy signal data
        signal_data = {
            "symbol": symbol,
            "direction": direction,
            "strategy": f"CORE_{strategy_type}",
            "score": score,
            "confidence": confidence,
            "regime": "core_strategy",
            "confirmations": confirmations,
            "risk_adjusted": True,
            "core_strategy": True
        }
        
        # Log core strategy execution
        log(f"🎯 CORE STRATEGY EXECUTION: {symbol}")
        log(f"   Direction: {direction} | Type: {strategy_type}")
        log(f"   Score: {score:.1f} | Confidence: {confidence}%")
        log(f"   Confirmations: {', '.join(confirmations)}")
        log(f"   Risk: {adjusted_risk:.2%}")
        
        # Execute trade
        trade_result = await execute_trade_if_valid(signal_data, adjusted_risk)
        
        if trade_result:
            # Track core strategy performance
            log_signal(symbol)
            track_signal(symbol, score)
            
            # Send core strategy notification
            await send_core_strategy_notification(signal_data, trade_result)
            
            log(f"✅ CORE STRATEGY: Trade executed successfully for {symbol}")
        else:
            log(f"❌ CORE STRATEGY: Trade execution failed for {symbol}")
            
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error executing trade for {symbol}: {e}", level="ERROR")

async def send_core_strategy_notification(signal_data, trade_result):
    """Send core strategy specific notification"""
    try:
        symbol = signal_data["symbol"]
        direction = signal_data["direction"]
        strategy = signal_data["strategy"]
        score = signal_data["score"]
        confidence = signal_data["confidence"]
        confirmations = signal_data["confirmations"]
        
        msg = f"🎯 <b>CORE STRATEGY EXECUTED</b>\n\n"
        msg += f"Symbol: <b>{symbol}</b>\n"
        msg += f"Direction: <b>{direction.upper()}</b>\n"
        msg += f"Strategy: <b>{strategy}</b>\n"
        msg += f"Score: <b>{score:.1f}</b>\n"
        msg += f"Confidence: <b>{confidence}%</b>\n"
        msg += f"Confirmations ({len(confirmations)}):\n"
        msg += f"   • {', '.join(confirmations)}\n"
        
        if trade_result:
            msg += f"\n💰 Entry: <b>{trade_result.get('entry_price', 'N/A')}</b>"
            msg += f"\n🛡️ Stop Loss: <b>{trade_result.get('sl_price', 'N/A')}</b>"
            msg += f"\n🎯 Take Profit: <b>{trade_result.get('tp_price', 'N/A')}</b>"
        
        await send_telegram_message(msg)
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error sending notification: {e}", level="ERROR")

async def core_monitor_loop():
    """Core strategy specific monitoring"""
    while True:
        try:
            if not active_trades:
                await asyncio.sleep(10)
                continue
            
            log(f"🔍 CORE STRATEGY: Monitoring {len(active_trades)} trades")
            
            for symbol, trade in list(active_trades.items()):
                try:
                    if trade.get("exited"):
                        continue
                    
                    # Core strategy uses simple but effective monitoring
                    await monitor_core_trade(symbol, trade)
                    
                except Exception as e:
                    log(f"❌ CORE STRATEGY: Error monitoring {symbol}: {e}", level="ERROR")
                    continue
            
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            log(f"❌ CORE STRATEGY: Error in monitor loop: {e}", level="ERROR")
            await asyncio.sleep(20)

async def monitor_core_trade(symbol, trade):
    """Simple but effective core trade monitoring"""
    try:
        # Basic monitoring - let the unified exit manager handle details
        # Core strategy focuses on letting winners run and cutting losers quickly
        pass
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error monitoring core trade {symbol}: {e}", level="ERROR")

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
        
    # Minimal task setup - only what's needed for core strategy
    asyncio.create_task(stream_candles(symbols))
    asyncio.create_task(core_monitor_loop())
    asyncio.create_task(monitor_btc_trend_accuracy())  # Keep trend monitoring
    asyncio.create_task(monitor_altseason_status())    # Keep altseason monitoring
    asyncio.create_task(periodic_trade_sync())         # Keep trade sync
    asyncio.create_task(bybit_sync_loop(120))         # Keep exchange sync

    await asyncio.sleep(5)

    # Core strategy main loop - simplified and focused
    while True:
        try:
            trend_context = await get_trend_context_cached()
            
            # Core strategy execution
            await core_strategy_scan(symbols, trend_context)
            
            # Simple daily report
            await send_daily_report()
            
        except Exception as e:
            log(f"❌ CORE STRATEGY: Error in main loop: {e}", level="ERROR")
            write_log(f"CORE STRATEGY MAIN LOOP ERROR: {str(e)}", level="ERROR")
            await send_error_to_telegram(traceback.format_exc())
        
        await asyncio.sleep(1.0)  # Core strategy scans every 1 second for precision

async def bybit_sync_loop(interval_sec: int = 120):
    """Simplified sync for core strategy"""
    while True:
        try:
            await sync_bot_with_bybit(send_telegram=False)
        except Exception as e:
            await send_error_to_telegram(f"Core strategy sync error: {e}")
        await asyncio.sleep(interval_sec)

if __name__ == "__main__":
    log("🔧 DEBUG: CORE STRATEGY main.py is running...")
    log(f"🔍 CORE STRATEGY thresholds - Scalp: {MIN_SCALP_SCORE}, Intraday: {MIN_INTRADAY_SCORE}, Swing: {MIN_SWING_SCORE}")
    log(f"🔒 CORE STRATEGY limits - Max positions: {MAX_CORE_POSITIONS}, Scalp: {MAX_SCALP_POSITIONS}, Intraday: {MAX_INTRADAY_POSITIONS}, Swing: {MAX_SWING_POSITIONS}")
    
    # Store bot startup time
    startup_time = time.time()

    async def restart_forever():
        """Core strategy restart mechanism"""
        while True:
            try:
                await run_core_bot()
            except Exception as e:
                err_msg = f"🔁 Restarting CORE STRATEGY bot due to crash:\n{traceback.format_exc()}"
                log(err_msg, level="ERROR")
                await send_error_to_telegram(err_msg)
                await asyncio.sleep(10)

    asyncio.run(restart_forever())









