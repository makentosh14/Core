# pattern_matcher.py - COMPLETE FIXED VERSION
# Enhanced pattern matching with historical move prediction

import asyncio
import json
import os
from datetime import datetime
from collections import defaultdict
from logger import log, write_log
from error_handler import send_telegram_message
from pattern_detector import detect_pattern

# Pattern matching statistics
pattern_stats = {
    "scans": 0,
    "matches": 0,
    "trades": 0
}

# Pattern match memory (to avoid duplicate triggers)
recent_pattern_matches = {}
MATCH_COOLDOWN = 2  # hours before allowing same pattern to trigger again

# FIXED: Use same file as pattern discovery
PATTERN_DB_PATH = "pattern_memory.json"

def load_pattern_memory():
    """Load historical pattern database"""
    if os.path.exists(PATTERN_DB_PATH):
        try:
            with open(PATTERN_DB_PATH, "r") as f:
                data = json.load(f)
            log(f"📚 Loaded pattern database: {len(data)} historical patterns")
            return data
        except Exception as e:
            log(f"❌ Failed to load pattern memory: {e}", level="ERROR")
    else:
        log("⚠️ Pattern database not found - will be created as patterns are discovered")
    return []

def save_cooldown_memory(data):
    """Save pattern match memory (cooldowns)"""
    try:
        # Note: This saves cooldown data, not the main pattern database
        cooldown_path = "pattern_match_cooldowns.json"
        with open(cooldown_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"❌ Failed to save pattern match memory: {e}", level="ERROR")

def load_cooldown_memory():
    """Load pattern match cooldowns separately"""
    cooldown_path = "pattern_match_cooldowns.json"
    if os.path.exists(cooldown_path):
        try:
            with open(cooldown_path, "r") as f:
                return json.load(f)
        except Exception as e:
            log(f"❌ Failed to load cooldown memory: {e}", level="ERROR")
    return {}

async def pattern_match_scan(symbols):
    """
    Main pattern matching function - scans for patterns and predicts moves
    """
    global recent_pattern_matches
    
    pattern_stats["scans"] += 1
    
    # Load historical patterns and cooldowns
    patterns_db = load_pattern_memory()
    recent_pattern_matches = load_cooldown_memory()
    
    if not patterns_db:
        log("⚠️ No pattern database found - patterns need time to build up")
        return
    
    log(f"🔍 Scanning {len(symbols)} symbols against {len(patterns_db)} historical patterns")
    
    # Group patterns by type for statistics
    pattern_performance = defaultdict(list)
    for pattern_record in patterns_db:
        pattern_type = pattern_record.get('pattern')
        if pattern_type:
            pattern_performance[pattern_type].append(pattern_record)
    
    matches_found = 0
    patterns_detected = 0
    
    for symbol in symbols:
        try:
            # Get current candles
            from websocket_candles import live_candles
            
            if symbol not in live_candles or not live_candles[symbol].get("5"):
                continue
                
            candles = list(live_candles[symbol]["5"])
            if len(candles) < 30:
                continue
                
            # Detect current pattern
            current_pattern = detect_pattern(candles)
            if not current_pattern:
                continue
                
            patterns_detected += 1
            log(f"🎯 Pattern detected: {symbol} - {current_pattern}")
            
            # Check if we have historical data for this pattern
            if current_pattern not in pattern_performance:
                log(f"❌ No historical data for pattern: {current_pattern}")
                continue
            
            # Check cooldown
            if is_pattern_in_cooldown(symbol, current_pattern):
                log(f"⏰ {symbol} - {current_pattern} still in cooldown")
                continue
            
            # Analyze historical performance of this pattern
            historical_data = pattern_performance[current_pattern]
            pattern_stats_analysis = analyze_pattern_historical_performance(historical_data)
            
            # Only proceed if pattern has decent historical performance
            if pattern_stats_analysis['win_rate'] < 0.5:  # Less than 50% win rate
                log(f"📉 {symbol} - {current_pattern} has poor win rate: {pattern_stats_analysis['win_rate']:.1%}")
                continue
            
            # Calculate context similarity with best historical matches
            best_matches = find_best_context_matches(candles, historical_data)
            
            if best_matches:
                best_match = best_matches[0]  # Take the best match
                similarity_score = best_match['similarity']
                predicted_move = best_match['predicted_move']
                predicted_direction = best_match['predicted_direction']
                
                if similarity_score > 0.4:  # 40% similarity threshold (lowered for crypto volatility)
                    matches_found += 1
                    pattern_stats["matches"] += 1
                    
                    # Record this match to prevent repeats
                    record_pattern_match(symbol, current_pattern)
                    save_cooldown_memory(recent_pattern_matches)
                    
                    # Send alert with prediction
                    await send_enhanced_pattern_alert(
                        symbol, current_pattern, similarity_score, 
                        predicted_move, predicted_direction, pattern_stats_analysis
                    )
                    
                    # Execute trade if conditions are very favorable
                    if similarity_score > 0.6 and pattern_stats_analysis['win_rate'] > 0.65:
                        trade_executed = await execute_pattern_based_trade(
                            symbol, predicted_direction, predicted_move, current_pattern
                        )
                        if trade_executed:
                            pattern_stats["trades"] += 1
                            log(f"✅ Pattern trade executed: {symbol} - {current_pattern}")
                
        except Exception as e:
            log(f"❌ Pattern match error for {symbol}: {e}", level="ERROR")
            continue
    
    log(f"📊 Pattern scan complete: {patterns_detected} patterns detected, {matches_found} matches found")

def analyze_pattern_historical_performance(historical_data):
    """Analyze how well this pattern has performed historically"""
    if not historical_data:
        return {'win_rate': 0, 'avg_move': 0, 'sample_size': 0}
    
    total_trades = len(historical_data)
    profitable_trades = sum(1 for record in historical_data if record.get('move_pct', 0) > 1.0)
    avg_move = sum(record.get('move_pct', 0) for record in historical_data) / total_trades
    
    return {
        'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
        'avg_move': avg_move,
        'sample_size': total_trades
    }

def find_best_context_matches(current_candles, historical_data, max_matches=3):
    """Find historical patterns with most similar context to current situation"""
    current_context = analyze_pattern_context(current_candles)
    matches = []
    
    for historical_record in historical_data:
        historical_context = historical_record.get('context', {})
        similarity = calculate_context_similarity(current_context, historical_context)
        
        if similarity > 0.2:  # Minimum similarity threshold
            matches.append({
                'similarity': similarity,
                'predicted_move': historical_record.get('move_pct', 0),
                'predicted_direction': historical_record.get('direction', 'unknown'),
                'historical_record': historical_record
            })
    
    # Sort by similarity (best first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:max_matches]

def is_pattern_in_cooldown(symbol, pattern):
    """Check if this symbol/pattern combination is in cooldown"""
    if symbol not in recent_pattern_matches:
        return False
    
    if pattern not in recent_pattern_matches[symbol]:
        return False
    
    try:
        last_match_time = datetime.strptime(
            recent_pattern_matches[symbol][pattern], 
            "%Y-%m-%d %H:%M:%S"
        )
        hours_since = (datetime.now() - last_match_time).total_seconds() / 3600
        return hours_since < MATCH_COOLDOWN
    except:
        return False

def record_pattern_match(symbol, pattern):
    """Record that we matched this pattern to prevent immediate repeats"""
    if symbol not in recent_pattern_matches:
        recent_pattern_matches[symbol] = {}
    
    recent_pattern_matches[symbol][pattern] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_pattern_memory(recent_pattern_matches)

def analyze_pattern_context(candles):
    """Extract context features from candles for pattern matching"""
    try:
        if len(candles) < 20:
            return {
                "volume_ratio": 0,
                "price_velocity": 0,
                "range_expansion": 0,
                "pre_pattern_trend": "neutral"
            }
        
        # Recent volume vs previous
        recent_vol = sum(float(c["volume"]) for c in candles[-5:])
        prev_vol = sum(float(c["volume"]) for c in candles[-10:-5])
        volume_ratio = recent_vol / prev_vol if prev_vol > 0 else 1
        
        # Price velocity (normalized rate of change)
        close_prices = [float(c["close"]) for c in candles[-10:]]
        first_price = close_prices[0]
        last_price = close_prices[-1]
        price_velocity = (last_price - first_price) / first_price if first_price > 0 else 0
        
        # Range expansion
        recent_ranges = [float(c["high"]) - float(c["low"]) for c in candles[-5:]]
        prev_ranges = [float(c["high"]) - float(c["low"]) for c in candles[-10:-5]]
        avg_recent_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
        avg_prev_range = sum(prev_ranges) / len(prev_ranges) if prev_ranges else 0
        range_expansion = avg_recent_range / avg_prev_range if avg_prev_range > 0 else 1
        
        # Pre-pattern trend
        pre_pattern_change = (close_prices[-2] - close_prices[0]) / close_prices[0] if close_prices[0] > 0 else 0
        if pre_pattern_change > 0.01:
            trend = "bullish"
        elif pre_pattern_change < -0.01:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "volume_ratio": round(volume_ratio, 2),
            "price_velocity": round(price_velocity, 4),
            "range_expansion": round(range_expansion, 2),
            "pre_pattern_trend": trend
        }
        
    except Exception as e:
        log(f"❌ Error analyzing pattern context: {e}", level="ERROR")
        return {"volume_ratio": 0, "price_velocity": 0, "range_expansion": 0, "pre_pattern_trend": "neutral"}

def calculate_context_similarity(current_context, stored_context):
    """Calculate similarity between current and historical pattern context"""
    try:
        if not current_context or not stored_context:
            return 0
            
        score = 0
        
        # Volume similarity (25% of total score)
        vol_diff = abs(current_context.get("volume_ratio", 0) - stored_context.get("volume_ratio", 0))
        vol_score = max(0, 0.25 - vol_diff * 0.125)
        
        # Price velocity similarity (25% of total score)
        vel_diff = abs(current_context.get("price_velocity", 0) - stored_context.get("price_velocity", 0))
        vel_score = max(0, 0.25 - vel_diff * 5)
        
        # Range expansion similarity (25% of total score)
        range_diff = abs(current_context.get("range_expansion", 0) - stored_context.get("range_expansion", 0))
        range_score = max(0, 0.25 - range_diff * 0.125)
        
        # Trend direction match (25% of total score)
        trend_match = current_context.get("pre_pattern_trend") == stored_context.get("pre_pattern_trend")
        trend_score = 0.25 if trend_match else 0
        
        total_score = vol_score + vel_score + range_score + trend_score
        return round(total_score, 2)
        
    except Exception as e:
        log(f"❌ Error calculating similarity: {e}", level="ERROR")
        return 0

async def send_enhanced_pattern_alert(symbol, pattern, similarity, predicted_move, predicted_direction, stats):
    """Send enhanced Telegram alert with historical prediction"""
    
    confidence_emoji = "🔥" if similarity > 0.8 else "⚡" if similarity > 0.6 else "💡"
    
    message = (
        f"{confidence_emoji} <b>Pattern Prediction</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Pattern:</b> {pattern}\n"
        f"<b>Similarity:</b> {similarity * 100:.1f}%\n"
        f"<b>Predicted Move:</b> {predicted_direction} {abs(predicted_move):.1f}%\n"
        f"<b>Historical Win Rate:</b> {stats['win_rate']:.1%}\n"
        f"<b>Avg Historical Move:</b> {stats['avg_move']:.1f}%\n"
        f"<b>Sample Size:</b> {stats['sample_size']} similar patterns\n"
        f"\n🎯 Based on {stats['sample_size']} similar historical cases, expecting {predicted_direction} move"
    )
    
    await send_telegram_message(message)
    log(f"🧬 Pattern prediction sent: {symbol} - {pattern} (similarity: {similarity:.2f})")

async def execute_pattern_based_trade(symbol, predicted_direction, predicted_move, pattern, similarity_score=None):
    """Execute trade based on pattern prediction with your existing trade system"""
    try:
        # Convert to your trade direction format
        trade_direction = "Long" if predicted_direction == "pump" else "Short"
        
        # Calculate confidence from predicted move size
        confidence = min(abs(predicted_move) * 10, 95)
        
        # Determine trade type from move size
        if abs(predicted_move) < 3:
            trade_type = "Scalp"
        elif abs(predicted_move) < 6:
            trade_type = "Intraday" 
        else:
            trade_type = "Swing"
        
        # Create signal data for your existing trade executor
        signal_data = {
            "symbol": symbol,
            "direction": trade_direction,
            "score": 12.0,  # High score for pattern trades
            "confidence": confidence,
            "strategy": f"pattern_{pattern}",
            "predicted_move": predicted_move,
            "pattern_based": True,
            "trade_type": trade_type
        }
        
        # Use your existing trade execution system
        from trade_executor import execute_trade_if_valid
        trade_result = await execute_trade_if_valid(signal_data)
        
        if trade_result:
            log(f"✅ Pattern trade executed: {symbol} {trade_direction} based on {pattern} (predicted: {predicted_move:+.1f}%)")
            
            # Track this as a pattern-based trade
            trade_data = {
                "score_history": [12.0],
                "trade_type": trade_type,
                "entry_price": trade_result.get('entry_price'),
                "direction": trade_direction.lower(),
                "cycles": 0,
                "exited": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "qty": trade_result.get('qty'),
                "sl_order_id": trade_result.get('sl_order_id'),
                "tp1_target": trade_result.get('tp1_price'),
                "sl_price": trade_result.get('sl_price'),
                "original_sl": trade_result.get('sl_price'),
                "pattern_based": True,
                "pattern_type": pattern,
                "predicted_move": predicted_move,
                "similarity_score": similarity_score,
                "similarity_score": None  # Will be set by caller
            }

            await execute_pattern_based_trade(symbol, predicted_direction, predicted_move, current_pattern, similarity_score)
            
            # Track the trade
            from monitor import track_active_trade
            track_active_trade(symbol, trade_data)
            
            return True
        else:
            log(f"❌ Pattern trade execution failed: {symbol}")
            return False
            
    except Exception as e:
        log(f"❌ Error executing pattern trade for {symbol}: {e}", level="ERROR")
        return False

# LEGACY COMPATIBILITY - Keep original function for backward compatibility
async def send_pattern_match_alert(symbol, pattern_name, pattern_data, match_score):
    """Legacy function - kept for compatibility"""
    message = (
        f"🧬 <b>Pattern Match Detected</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Pattern:</b> {pattern_name}\n"
        f"<b>Similarity:</b> {match_score * 100:.1f}%\n"
        f"<b>Historical Direction:</b> {pattern_data.get('direction', 'unknown')}\n"
        f"<b>Average Move:</b> {pattern_data.get('avg_move', 0):.2f}%\n"
        f"<b>Win Rate:</b> {pattern_data.get('win_rate', 0):.1f}%"
    )
    
    await send_telegram_message(message)
    log(f"🧬 Pattern match alert sent for {symbol}: {pattern_name}")

async def execute_pattern_trade(symbol, pattern_data, pattern_name):
    """Legacy function - kept for compatibility"""
    log(f"🔍 Pattern trade opportunity for {symbol}: {pattern_name}")
    write_log(f"PATTERN TRADE: {symbol} | {pattern_name} | Historical Success: {pattern_data.get('win_rate', 0):.1f}%")
    return False

# DIAGNOSTIC FUNCTIONS

def get_pattern_stats():
    """Get current pattern matching statistics"""
    return {
        "total_scans": pattern_stats["scans"],
        "total_matches": pattern_stats["matches"], 
        "total_trades": pattern_stats["trades"],
        "match_rate": pattern_stats["matches"] / pattern_stats["scans"] if pattern_stats["scans"] > 0 else 0,
        "trade_rate": pattern_stats["trades"] / pattern_stats["matches"] if pattern_stats["matches"] > 0 else 0
    }

def get_database_info():
    """Get information about the pattern database"""
    patterns_db = load_pattern_memory()
    
    if not patterns_db:
        return {"status": "empty", "count": 0}
    
    # Analyze the database
    pattern_types = defaultdict(int)
    directions = defaultdict(int)
    recent_count = 0
    
    now = datetime.now()
    
    for record in patterns_db:
        pattern_type = record.get('pattern')
        if pattern_type:
            pattern_types[pattern_type] += 1
        
        direction = record.get('direction')
        if direction:
            directions[direction] += 1
        
        # Check if recent (last 24 hours)
        try:
            timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', ''))
            if (now - timestamp).total_seconds() < 86400:  # 24 hours
                recent_count += 1
        except:
            continue
    
    return {
        "status": "active",
        "count": len(patterns_db),
        "pattern_types": dict(pattern_types),
        "directions": dict(directions),
        "recent_patterns": recent_count
    }

# ENHANCED DIAGNOSTIC LOGGING

async def pattern_match_scan_debug(symbols):
    """Debug version with extensive logging"""
    global recent_pattern_matches
    
    log("🔍 PATTERN MATCHER DEBUG SCAN STARTING")
    pattern_stats["scans"] += 1
    
    # Load data with debug info
    patterns_db = load_pattern_memory()
    recent_pattern_matches = load_cooldown_memory()
    
    log(f"📊 Debug Info:")
    log(f"   Patterns in database: {len(patterns_db)}")
    log(f"   Symbols to scan: {len(symbols)}")
    log(f"   Active cooldowns: {len(recent_pattern_matches)}")
    
    if not patterns_db:
        log("❌ CRITICAL: No pattern database - matcher cannot work without historical data!")
        return
    
    # Show pattern types available
    pattern_types = set(record.get('pattern') for record in patterns_db if record.get('pattern'))
    log(f"   Available pattern types: {list(pattern_types)[:10]}...")
    
    patterns_detected = 0
    patterns_checked = 0
    matches_found = 0
    
    for symbol in symbols:
        try:
            from websocket_candles import live_candles
            
            if symbol not in live_candles or not live_candles[symbol].get("5"):
                continue
            
            patterns_checked += 1
            candles = list(live_candles[symbol]["5"])
            if len(candles) < 30:
                continue
                
            # Detect pattern with debug
            current_pattern = detect_pattern(candles)
            if current_pattern:
                patterns_detected += 1
                log(f"🎯 DETECTED: {symbol} has pattern '{current_pattern}'")
                
                # Check if pattern exists in database
                pattern_records = [r for r in patterns_db if r.get('pattern') == current_pattern]
                if pattern_records:
                    log(f"   ✅ Found {len(pattern_records)} historical examples of '{current_pattern}'")
                    
                    # Quick win rate check
                    profitable = sum(1 for r in pattern_records if r.get('move_pct', 0) > 1.0)
                    win_rate = profitable / len(pattern_records)
                    log(f"   📊 Historical win rate: {win_rate:.1%}")
                    
                    if win_rate >= 0.5:
                        matches_found += 1
                        log(f"   ✅ POTENTIAL MATCH: {symbol} - {current_pattern}")
                    else:
                        log(f"   ❌ Poor win rate, skipping: {symbol} - {current_pattern}")
                else:
                    log(f"   ❌ No historical data for: {current_pattern}")
            
        except Exception as e:
            log(f"❌ Debug scan error for {symbol}: {e}", level="ERROR")
    
    log(f"🏁 DEBUG SCAN COMPLETE:")
    log(f"   Symbols checked: {patterns_checked}")
    log(f"   Patterns detected: {patterns_detected}")
    log(f"   Potential matches: {matches_found}")

# Add alias for main.py compatibility
pattern_match_scan_enhanced = pattern_match_scan
