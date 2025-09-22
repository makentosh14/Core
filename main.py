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

def fix_live_candles_structure(live_candles):
    """
    Convert generators/iterators to lists in live_candles structure
    This fixes all slice errors by ensuring proper list types
    """
    try:
        if not live_candles:
            return {}
            
        fixed_candles = {}
        
        for symbol, timeframes in live_candles.items():
            if not isinstance(timeframes, dict):
                continue
                
            fixed_candles[symbol] = {}
            
            for tf, candle_data in timeframes.items():
                if candle_data is None:
                    fixed_candles[symbol][tf] = []
                    continue
                
                # Convert any iterable to a proper list
                try:
                    if isinstance(candle_data, list):
                        # Already a proper list
                        fixed_candles[symbol][tf] = candle_data
                    elif hasattr(candle_data, '__iter__') and not isinstance(candle_data, (str, bytes, dict)):
                        # It's an iterable (generator, iterator, tuple, etc.) - convert to list
                        candle_list = list(candle_data)
                        fixed_candles[symbol][tf] = candle_list
                        log(f"🔧 Fixed {symbol}[{tf}]: converted {type(candle_data)} to list with {len(candle_list)} items")
                    else:
                        # Not iterable or wrong type
                        log(f"⚠️ Skipping {symbol}[{tf}]: unsupported type {type(candle_data)}", level="WARN")
                        fixed_candles[symbol][tf] = []
                        
                except Exception as e:
                    log(f"❌ Error converting {symbol}[{tf}]: {e}", level="ERROR")
                    fixed_candles[symbol][tf] = []
        
        return fixed_candles
        
    except Exception as e:
        log(f"❌ Error fixing live_candles structure: {e}", level="ERROR")
        return live_candles  # Return original if fix fails

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
    source = fix_live_candles_structure(live_candles)

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
        
        for symbol in quality_symbols:  # Limit to top 20 symbols for focus
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
                    min_needed = {'1': 30, '5': 30, '15': 8}
                    if tf in source.get(symbol, {}):
                        candles = list(source[symbol][tf])
                        if candles and len(candles) >= 30:  # Require more history for quality
                            core_candles[tf] = candles
                
                if len(core_candles) < 3:  # Must have all 3 timeframes
                    continue

                scanned_count += 1

                # === CORE STRATEGY SIGNAL GENERATION ===
                
                # 1. Get full scoring data (we need this anyway)
                score_result = score_symbol(symbol, core_candles, trend_context) 
                score, tf_scores, trade_type, indicator_scores, used_indicators = score_result

                # 2. Calculate core score with momentum bonus
                core_score = await calculate_core_score(symbol, core_candles, trend_context)
                if core_score < MIN_SCALP_SCORE:
                    continue

                # 3. Determine direction
                direction = determine_core_direction(core_candles, trend_context)
                if not direction:
                    continue

                # 4. Calculate confidence (now we have all required parameters)
                confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                if confidence < 60:
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
    """Simple filter - focus on basic criteria only"""
    log(f"✅ Fixed live_candles structure before filtering")
    
    filtered = []
    
    for symbol in symbols:
        try:
            # Basic checks
            if 'USDT' not in symbol:
                continue
                
            if symbol not in live_candles:
                continue
            
            # Get any available candles
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
            if avg_volume > 10000:  # Very low: 10k volume
                filtered.append(symbol)
                
            # Stop at 50 symbols for efficiency
            if len(filtered) >= 50:
                break
                
        except Exception as e:
            log(f"⚠️ Error filtering {symbol}: {e}", level="WARN")
            continue
    
    log(f"✅ Filtered to {len(filtered)} symbols with relaxed criteria")
    return filtered


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
    """Simple direction determination based on momentum"""
    try:
        # Simple momentum-based direction
        candles_5m = core_candles.get('5', [])
        if not candles_5m or len(candles_5m) < 10:
            return None
            
        # Check recent price movement
        recent_closes = [float(c['close']) for c in candles_5m[-10:]]
        start_price = recent_closes[0]
        end_price = recent_closes[-1]
        
        price_change = (end_price - start_price) / start_price
        
        # Simple direction based on price movement
        if price_change > 0.002:  # 0.2% up
            direction = "Long"
        elif price_change < -0.002:  # 0.2% down
            direction = "Short"
        else:
            return None  # No clear direction
        
        # Apply trend context filters
        trend_direction = trend_context.get("trend", "neutral")
        trend_strength = trend_context.get("trend_strength", 0.5)
        
        # For strong downtrend, don't allow long positions
        if trend_strength > 0.6 and trend_direction == "bearish" and direction == "Long":
            return None
            
        return direction
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error determining direction: {e}", level="ERROR")
        return None

async def validate_core_conditions(symbol, core_candles, direction, trend_context):
    """FIXED: Validate core strategy specific conditions with relaxed requirements"""
    try:
        log(f"🔍 Validating core conditions for {symbol} {direction}")
        
        validation_score = 0
        max_score = 5
        
        # 1. Volume validation (weight: 2 points)
        volume_ok = validate_core_volume(core_candles)
        if volume_ok:
            validation_score += 2
            log(f"   Volume check: ✅ (+2 points)")
        else:
            log(f"   Volume check: ❌ (0 points)")
        
        # 2. Price action quality (weight: 1 point)
        price_action_ok = validate_core_price_action(core_candles, direction)
        if price_action_ok:
            validation_score += 1
            log(f"   Price action check: ✅ (+1 point)")
        else:
            log(f"   Price action check: ❌ (0 points)")
        
        # 3. Risk/reward validation (weight: 1 point) - with fallback
        risk_reward_ok = validate_core_risk_reward(core_candles, direction)
        if risk_reward_ok:
            validation_score += 1
            log(f"   Risk/reward check: ✅ (+1 point)")
        else:
            # Fallback: Check if at least price is trending in right direction
            if check_price_direction_alignment(core_candles, direction):
                validation_score += 0.5
                log(f"   Risk/reward check: ⚠️ (fallback +0.5 points)")
            else:
                log(f"   Risk/reward check: ❌ (0 points)")
        
        # 4. Market timing validation (weight: 0.5 points)
        timing_ok = validate_core_timing()
        if timing_ok:
            validation_score += 0.5
            log(f"   Timing check: ✅ (+0.5 points)")
        else:
            log(f"   Timing check: ❌ (0 points)")
        
        # 5. Trend coherence (weight: 0.5 points)
        trend_coherence_ok = validate_core_trend_coherence(core_candles, direction)
        if trend_coherence_ok:
            validation_score += 0.5
            log(f"   Trend coherence check: ✅ (+0.5 points)")
        else:
            log(f"   Trend coherence check: ❌ (0 points)")
        
        # Pass if score >= 3.0 out of 5.0 (60% threshold)
        passing_threshold = 3.0
        result = validation_score >= passing_threshold
        
        log(f"📊 Validation score: {validation_score}/{max_score} (need {passing_threshold})")
        log(f"🎯 Final result: {'✅ PASS' if result else '❌ FAIL'}")
        
        return result
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error validating conditions for {symbol}: {e}", level="ERROR")
        return False

# Replace your validate_core_volume function in main.py with this debug version:

def validate_core_volume(core_candles):
    """
    Improved volume validation with better logic and thresholds
    """
    try:
        log(f"🔍 DEBUG: validate_core_volume called with keys: {list(core_candles.keys())}")
        
        if '1' not in core_candles:
            log(f"❌ DEBUG: No '1' timeframe in core_candles")
            return False
        
        log(f"🔍 DEBUG: Found '1' timeframe, type: {type(core_candles['1'])}")
        
        candles = core_candles['1'][-30:]  # Use 30 candles for better average
        log(f"🔍 DEBUG: Extracted {len(candles)} candles from last 30")
        
        if len(candles) < 20:
            log(f"❌ DEBUG: Not enough candles: {len(candles)} < 20")
            return False
        
        # Debug: Show first and last candle
        log(f"🔍 DEBUG: First candle: {candles[0] if candles else 'None'}")
        log(f"🔍 DEBUG: Last candle: {candles[-1] if candles else 'None'}")
        
        volumes = [float(c.get('volume', 0)) for c in candles]
        log(f"🔍 DEBUG: Extracted {len(volumes)} volumes")
        log(f"🔍 DEBUG: First 5 volumes: {volumes[:5]}")
        log(f"🔍 DEBUG: Last 5 volumes: {volumes[-5:]}")
        
        if len(volumes) < 20:
            log(f"❌ DEBUG: Not enough volumes: {len(volumes)} < 20")
            return False
        
        # Improved volume analysis
        # Use median instead of mean to avoid skew from outliers
        sorted_volumes = sorted(volumes[:-5])  # Exclude recent 5 candles
        median_volume = sorted_volumes[len(sorted_volumes)//2] if sorted_volumes else 0
        avg_volume = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) > 5 else 0
        
        # Use average of last 3 candles instead of 5 to be more sensitive
        recent_volume = sum(volumes[-3:]) / 3
        
        # Calculate both ratios
        avg_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        median_ratio = recent_volume / median_volume if median_volume > 0 else 0
        
        log(f"🔍 DEBUG: Average volume (excluding recent): {avg_volume}")
        log(f"🔍 DEBUG: Median volume (excluding recent): {median_volume}")
        log(f"🔍 DEBUG: Recent volume (last 3): {recent_volume}")
        log(f"🔍 DEBUG: Avg ratio: {avg_ratio:.3f} (needs > 1.5)")
        log(f"🔍 DEBUG: Median ratio: {median_ratio:.3f} (needs > 1.5)")
        
        # More lenient volume requirement - either ratio needs to be good
        volume_threshold = 1.5  # Relaxed from 2.0
        result = avg_ratio > volume_threshold or median_ratio > volume_threshold
        
        log(f"🔍 DEBUG: Volume validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG: Volume validation error: {e}", level="ERROR")
        import traceback
        log(f"❌ DEBUG: Traceback: {traceback.format_exc()}", level="ERROR")
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
        
        # RELAXED: Check for general directional movement
        if direction.lower() == "long":
            # For long signals, require some upward movement
            upward_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            return upward_moves >= 4  # At least 40% upward moves (was 60%)
        else:
            # For short signals, require some downward movement
            downward_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
            return downward_moves >= 4  # At least 40% downward moves
        
    except Exception as e:
        log(f"Price action validation error: {e}", level="WARN")
        return False

# Replace your validate_core_risk_reward function in main.py with this debug version:

def validate_core_risk_reward(core_candles, direction):
    """
    COMPLETE FIXED VERSION: Risk/reward validation with proper target calculation
    Addresses the core issue where LONG positions have targets too close to current price
    """
    try:
        log(f"🔍 DEBUG RR: validate_core_risk_reward called, direction={direction}")
        log(f"🔍 DEBUG RR: Available timeframes: {list(core_candles.keys())}")
        
        if '15' not in core_candles:
            log(f"❌ DEBUG RR: No '15' timeframe in core_candles")
            return False
        
        candles = core_candles['15'][-20:]
        log(f"🔍 DEBUG RR: Got {len(candles)} candles from 15m timeframe")
        
        if len(candles) < 10:
            log(f"❌ DEBUG RR: Not enough candles: {len(candles)} < 10")
            return False
        
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        closes = [float(c.get('close', 0)) for c in candles]
        
        current_price = closes[-1]
        log(f"🔍 DEBUG RR: Current price: {current_price}")
        
        if direction.lower() == "long":
            # FIXED APPROACH FOR LONG POSITIONS
            
            # 1. Find proper support level
            recent_lows = lows[-10:]  # Last 10 candles
            support_candidate = min(recent_lows)
            
            # Cap maximum risk at 3% for safety
            max_risk_percent = 0.03
            min_support_level = current_price * (1 - max_risk_percent)
            effective_support = max(support_candidate, min_support_level)
            
            # 2. Calculate risk first
            potential_risk = current_price - effective_support
            
            # 3. FIXED TARGET CALCULATION - Multiple approaches
            approaches = []
            
            # Approach A: Risk-based target (1.5x risk for 1.5 R/R ratio)
            risk_based_target = current_price + (potential_risk * 1.5)
            approaches.append(("risk_1.5x", risk_based_target))
            
            # Approach B: Percentage-based target (minimum 4% above current)
            percentage_target = current_price * 1.04
            approaches.append(("percent_4%", percentage_target))
            
            # Approach C: Recent resistance with buffer
            recent_highs = highs[-10:]
            recent_resistance = max(recent_highs)
            if recent_resistance > current_price * 1.01:  # At least 1% above
                buffered_resistance = recent_resistance * 1.01  # Add 1% buffer
                approaches.append(("resistance_buffered", buffered_resistance))
            
            # Approach D: Price range expansion
            price_range = max(highs) - min(lows)
            range_target = current_price + (price_range * 0.3)  # 30% of range
            approaches.append(("range_30%", range_target))
            
            # Choose the highest reasonable target
            effective_resistance = max(target for _, target in approaches)
            best_approach = max(approaches, key=lambda x: x[1])
            
            log(f"🔍 DEBUG RR: LONG - Support: {effective_support:.2f}")
            log(f"🔍 DEBUG RR: LONG - Target approaches: {[(name, f'{target:.2f}') for name, target in approaches]}")
            log(f"🔍 DEBUG RR: LONG - Selected: {best_approach[0]} = {effective_resistance:.2f}")
            
            potential_reward = effective_resistance - current_price
            
        else:  # SHORT POSITIONS - IMPROVED LOGIC
            # FIXED APPROACH FOR SHORT POSITIONS - Mirror the LONG logic
            
            # 1. Find proper resistance level
            recent_highs = highs[-10:]  # Last 10 candles
            resistance_candidate = max(recent_highs)
            
            # Cap maximum risk at 3% for safety
            max_risk_percent = 0.03
            max_resistance_level = current_price * (1 + max_risk_percent)
            effective_resistance = min(resistance_candidate, max_resistance_level)
            
            # 2. Calculate risk first
            potential_risk = effective_resistance - current_price
            
            # 3. FIXED TARGET CALCULATION - Multiple approaches for SHORT
            approaches = []
            
            # Approach A: Risk-based target (1.5x risk for 1.5 R/R ratio)
            risk_based_target = current_price - (potential_risk * 1.5)
            approaches.append(("risk_1.5x", risk_based_target))
            
            # Approach B: Percentage-based target (minimum 4% below current)
            percentage_target = current_price * 0.96
            approaches.append(("percent_4%", percentage_target))
            
            # Approach C: Recent support with buffer
            recent_lows = lows[-10:]
            recent_support = min(recent_lows)
            if recent_support < current_price * 0.99:  # At least 1% below
                buffered_support = recent_support * 0.99  # Subtract 1% buffer
                approaches.append(("support_buffered", buffered_support))
            
            # Approach D: Price range expansion
            price_range = max(highs) - min(lows)
            range_target = current_price - (price_range * 0.3)  # 30% of range down
            approaches.append(("range_30%", range_target))
            
            # Choose the lowest reasonable target (for shorts, lower is better)
            effective_support = min(target for _, target in approaches if target > 0)
            best_approach = min(approaches, key=lambda x: x[1] if x[1] > 0 else float('inf'))
            
            log(f"🔍 DEBUG RR: SHORT - Resistance: {effective_resistance:.4f}")
            log(f"🔍 DEBUG RR: SHORT - Target approaches: {[(name, f'{target:.4f}') for name, target in approaches if target > 0]}")
            log(f"🔍 DEBUG RR: SHORT - Selected: {best_approach[0]} = {effective_support:.4f}")
            
            potential_reward = current_price - effective_support
        
        log(f"🔍 DEBUG RR: Potential reward: {potential_reward:.2f}")
        log(f"🔍 DEBUG RR: Potential risk: {potential_risk:.2f}")
        
        # Validation checks
        if potential_reward <= 0 or potential_risk <= 0:
            log(f"❌ DEBUG RR: Invalid reward/risk: {potential_reward:.2f}/{potential_risk:.2f}")
            return False
        
        # Calculate risk/reward ratio
        rr_ratio = potential_reward / potential_risk
        log(f"🔍 DEBUG RR: Risk/Reward ratio: {rr_ratio:.3f} (needs >= 1.2)")
        
        # Additional quality checks - IMPROVED FOR ALL ASSET TYPES
        min_reward_threshold = current_price * 0.01  # Reduced to 1% (was 1.5%)
        if potential_reward < min_reward_threshold:
            log(f"❌ DEBUG RR: Reward too small: {potential_reward:.4f} < {min_reward_threshold:.4f}")
            return False
        
        max_risk_threshold = current_price * 0.04  # Maximum 4% risk
        if potential_risk > max_risk_threshold:
            log(f"❌ DEBUG RR: Risk too large: {potential_risk:.4f} > {max_risk_threshold:.4f}")
            return False
        
        result = rr_ratio >= 1.2
        log(f"🔍 DEBUG RR: Risk/reward validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG RR: Risk/reward validation error: {e}", level="ERROR")
        import traceback
        log(f"❌ DEBUG RR: Traceback: {traceback.format_exc()}", level="ERROR")
        return False


# ADDITIONAL HELPER FUNCTION
def debug_risk_reward_calculation(core_candles, direction, symbol="TEST"):
    """
    Debug helper to visualize what's happening in risk/reward calculation
    """
    print(f"\n🔬 DEBUGGING RISK/REWARD FOR {symbol} {direction}")
    print("=" * 60)
    
    if '15' not in core_candles:
        print("❌ No 15m timeframe data")
        return
    
    candles = core_candles['15'][-20:]
    highs = [float(c.get('high', 0)) for c in candles]
    lows = [float(c.get('low', 0)) for c in candles]
    closes = [float(c.get('close', 0)) for c in candles]
    current_price = closes[-1]
    
    print(f"📊 Current Price: {current_price:.2f}")
    print(f"📊 Recent Range: {min(lows):.2f} - {max(highs):.2f}")
    print(f"📊 Range Size: {max(highs) - min(lows):.2f} ({((max(highs) - min(lows))/current_price*100):.1f}%)")
    
    if direction.lower() == "long":
        recent_lows = lows[-10:]
        recent_highs = highs[-10:]
        support = min(recent_lows)
        resistance = max(recent_highs)
        
        print(f"\n🎯 LONG Analysis:")
        print(f"   Recent Support: {support:.2f} ({((support-current_price)/current_price*100):+.1f}%)")
        print(f"   Recent Resistance: {resistance:.2f} ({((resistance-current_price)/current_price*100):+.1f}%)")
        print(f"   Current vs Support: {current_price - support:.2f} risk")
        print(f"   Current vs Resistance: {resistance - current_price:.2f} reward")
        
        if resistance > current_price:
            basic_rr = (resistance - current_price) / (current_price - support)
            print(f"   Basic R/R Ratio: {basic_rr:.3f}")
        
    # Run the actual validation
    result = validate_core_risk_reward(core_candles, direction)
    print(f"\n✅ Final Result: {'PASS' if result else 'FAIL'}")
    
    return result
def remove_duplicate_levels(levels, threshold):
    """Remove levels that are too close to each other"""
    if not levels:
        return levels
    
    sorted_levels = sorted(levels)
    unique_levels = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        if abs(level - unique_levels[-1]) > threshold:
            unique_levels.append(level)
    
    return unique_levels

def validate_core_timing():
    """Validate market timing for core strategy"""
    try:
        from datetime import datetime
        current_hour = datetime.utcnow().hour
        
        # RELAXED: Extended trading hours (was 8-23, now 6-23)
        peak_hours = list(range(6, 24))  # 6 AM to 11 PM UTC
        
        return current_hour in peak_hours
        
    except Exception as e:
        log(f"Timing validation error: {e}", level="WARN")
        return True  # Default to True on error

def validate_core_trend_coherence(core_candles, direction):
    """FIXED: Validate trend coherence across AVAILABLE timeframes only"""
    try:
        trend_scores = {}
        
        # FIXED: Only check timeframes that are actually available
        available_timeframes = [tf for tf in ['1', '5', '15'] if tf in core_candles]
        
        if len(available_timeframes) < 2:  # Need at least 2 timeframes
            return False
        
        for tf in available_timeframes:
            candles = core_candles[tf][-20:]
            closes = [float(c.get('close', 0)) for c in candles]
            
            if len(closes) < 10:  # RELAXED: Reduced from 20 to 10
                continue
            
            # Calculate trend direction for this timeframe
            start_price = sum(closes[:3]) / 3  # RELAXED: First 3 candles (was 5)
            end_price = sum(closes[-3:]) / 3   # RELAXED: Last 3 candles (was 5)
            
            trend_scores[tf] = (end_price - start_price) / start_price
        
        # RELAXED: Majority of timeframes should agree on direction
        if direction.lower() == "long":
            positive_trends = sum(1 for score in trend_scores.values() if score > 0.001)  # RELAXED: 0.1% threshold
            return positive_trends >= len(trend_scores) * 0.6  # 60% agreement
        else:
            negative_trends = sum(1 for score in trend_scores.values() if score < -0.001)
            return negative_trends >= len(trend_scores) * 0.6  # 60% agreement
        
    except Exception as e:
        log(f"Trend coherence validation error: {e}", level="WARN")
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


def debug_live_link():
    """Prints IDs of live_candles in main and websocket module + a few counts."""
    try:
        from websocket_candles import live_candles as ws_live
    except Exception:
        ws_live = None
    try:
        log(f"🔗 live_candles id(main)={{id(live_candles)}} | id(ws)={{id(ws_live) if ws_live else 'N/A'}}")
    except Exception:
        pass
    if ws_live:
        shown = 0
        for sym, tfs in ws_live.items():
            try:
                c1 = len(tfs.get('1', []))
                c5 = len(tfs.get('5', []))
                c15 = len(tfs.get('15', []))
                log(f"📊 {sym}: 1m={{c1}}, 5m={{c5}}, 15m={{c15}}")
            except Exception:
                continue
            shown += 1
            if shown >= 5:
                break


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


























