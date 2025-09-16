# pattern_discovery.py - FIXED VERSION

import asyncio
from collections import defaultdict
from score import score_symbol
from pattern_detector import detect_pattern
from volume import get_average_volume
from logger import log, write_log
from error_handler import send_telegram_message, send_error_to_telegram
from websocket_candles import live_candles

import json
import os
from datetime import datetime

PATTERN_LOG_PATH = "pattern_memory.json"
MIN_MOVE_PCT = 2.0  # Detect patterns when ±2% or more move happens
TIMEFRAMES = ['1', '3', '5']  # Only short-term patterns for discovery
MAX_CANDLES = 20

# Load + Save Pattern Memory
def save_patterns(data):
    try:
        with open(PATTERN_LOG_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"❌ Failed to save pattern data: {e}", level="ERROR")

def load_patterns():
    if os.path.exists(PATTERN_LOG_PATH):
        try:
            with open(PATTERN_LOG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            log(f"❌ Failed to load pattern memory: {e}")
    return []

# Main pattern discovery scan
async def pattern_discovery_scan(symbols):
    pattern_db = load_patterns()

    for symbol in symbols:
        await asyncio.sleep(1.5)  # Prevent overload

        if symbol not in live_candles:
            continue

        try:
            candles = list(live_candles[symbol]['1'])  # Use 1m candles for trigger detection
            if len(candles) < MAX_CANDLES:
                continue

            recent = candles[-MAX_CANDLES:]
            open_price = float(recent[0]['open'])
            high = max(float(c['high']) for c in recent)
            low = min(float(c['low']) for c in recent)

            move_up = ((high - open_price) / open_price) * 100
            move_down = ((open_price - low) / open_price) * 100

            is_valid = move_up >= MIN_MOVE_PCT or move_down >= MIN_MOVE_PCT
            direction = "pump" if move_up >= move_down else "dump"

            if not is_valid:
                continue

            candles_by_tf = {
                tf: list(live_candles[symbol][str(tf)]) for tf in TIMEFRAMES
                if len(live_candles[symbol][str(tf)]) >= 30
            }

            # FIXED: Properly unpack all 5 values returned by score_symbol
            score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(symbol, candles_by_tf)
            
            last_pattern = detect_pattern(candles[-2:])
            volume_now = float(candles[-1]['volume'])
            avg_vol = get_average_volume(candles, window=15)

            pattern_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "direction": direction,
                "move_pct": round(move_up if direction == "pump" else move_down, 2),
                "trade_type": trade_type,
                "pattern": last_pattern,
                "volume_spike": volume_now > avg_vol * 1.5,
                "tf_scores": tf_scores,
                "score": score,
                "indicator_scores": indicator_scores,  # Include indicator scores
                "used_indicators": used_indicators,    # Include used indicators
                "context": {
                    "rsi": tf_scores.get('rsi', None),
                    "macd": tf_scores.get('macd', None),
                    "supertrend": tf_scores.get('supertrend', None)
                }
            }

            pattern_db.append(pattern_record)
            save_patterns(pattern_db)

            log(f"🔍 Pattern found on {symbol} ({direction}) | Move: {pattern_record['move_pct']}%")
            write_log(f"PATTERN FOUND: {symbol} | {pattern_record}")
            write_log("PATTERN DISCOVERY: cycle running...")

        except Exception as e:
            log(f"❌ Pattern scan failed for {symbol}: {e}", level="ERROR")
            # Log the full error for debugging
            import traceback
            log(f"Full error trace: {traceback.format_exc()}", level="ERROR")
