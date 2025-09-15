import os
import csv
import json
from datetime import datetime

LOG_PATH = "/mnt/data/bot_logs"
TRADE_LOG_PATH = "/mnt/data/trade_logs"
LOG_FILE = os.path.join(LOG_PATH, "trading_bot_activity.log")
TRADE_LOG_CSV = os.path.join(TRADE_LOG_PATH, "trade_setups.csv")

# Ensure directories exist
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(TRADE_LOG_PATH, exist_ok=True)

def write_log(message, level="INFO"):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] [{level.upper()}] {message}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception as e:
        print(f"Logging error: {e}")

def log_trade_to_file(symbol, direction, entry, sl, tp1, tp2, result, score, trade_type, confidence,
                      tf_scores=None, indicator_scores=None, used_indicators=None,
                      pattern_detected=None, whale_signal=None, volume_spike=None, sl_strategy=None,
                      missed_upside=None, pullback_after=None):
    """
    Log a structured trade result to CSV for later analysis.
    Includes scoring breakdowns, indicators, pattern/volume/whale flags, SL strategy,
    and missed upside/pullback values post-exit.
    """
    file_exists = os.path.isfile(TRADE_LOG_CSV)
    try:
        with open(TRADE_LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "symbol", "direction", "entry_price", "sl", "tp1", "tp2",
                "result", "score", "trade_type", "confidence",
                "tf_scores", "indicator_scores", "used_indicators",
                "pattern_detected", "whale_signal", "volume_spike", "sl_strategy",
                "missed_upside", "pullback_after"
            ], quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "result": result,
                "score": score,
                "trade_type": trade_type,
                "confidence": confidence,
                "tf_scores": json.dumps(tf_scores or {}, ensure_ascii=False),
                "indicator_scores": json.dumps(indicator_scores or {}, ensure_ascii=False),
                "used_indicators": json.dumps(used_indicators or [], ensure_ascii=False),
                "pattern_detected": pattern_detected,
                "whale_signal": whale_signal,
                "volume_spike": volume_spike,
                "sl_strategy": sl_strategy,
                "missed_upside": missed_upside,
                "pullback_after": pullback_after
            })
    except Exception as e:
        write_log(f"❌ Failed to log trade: {e}", level="ERROR")

def update_trade_result(symbol, result_value):
    """
    Update the result field of the most recent open trade for a given symbol.
    """
    try:
        if not os.path.exists(TRADE_LOG_CSV):
            return

        rows = []
        with open(TRADE_LOG_CSV, mode="r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        updated = False
        for row in reversed(rows):
            if row["symbol"] == symbol and row["result"] == "open":
                row["result"] = result_value
                updated = True
                break

        if updated:
            with open(TRADE_LOG_CSV, mode="w", newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(rows)
            write_log(f"✅ Trade result updated: {symbol} → {result_value}")
        else:
            write_log(f"⚠️ No open trade found for {symbol} to update", level="WARNING")

    except Exception as e:
        write_log(f"❌ Failed to update result for {symbol}: {e}", level="ERROR")

