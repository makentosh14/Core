"""
websocket_candles.py — Phase 3 audit rewrite

Phase 3 fixes applied:

  #1 (showstopper): deduplicate intra-bar WebSocket updates by timestamp.
      Bybit V5 kline sends a stream of updates every ~1-5s while a bar is
      forming. Old code appended every message, so the deque held mostly
      duplicates of the latest bar. Indicators were computing on garbage.
      New code: if the last entry has the same timestamp, REPLACE it
      (intra-bar update); otherwise APPEND (new bar). The deque now holds
      ~100 distinct bars.

  #4 — exponential backoff on reconnect (base 1s, cap 60s); resets to base
       on first successful message.

  #5 — Telegram alerts are throttled per (category, interval) to one every
       5 minutes. Stops the reconnect-storm alert flood.

  #6, #7 — seed runs on connect when the deque is sparse OR when its
            newest bar is stale. With #1 fixed, "sparse" is meaningful again.

  #11 — stream_candles uses gather(*tasks, return_exceptions=True) so one
        stream dying doesn't take the other six down.

Also exposes a `candles_are_fresh()` helper used by main.py's scan path to
refuse to score symbols whose data has gone stale (Phase 3 Fix #3).
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Dict, Optional

import websockets

from scanner import symbol_category_map
from logger import log
from error_handler import send_error_to_telegram
from bybit_api import signed_request


live_candles = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
SUPPORTED_INTERVALS = ['1', '3', '5', '15', '30', '60', '240']

# Map interval label → seconds in one bar. Used by candles_are_fresh().
INTERVAL_SECONDS: Dict[str, int] = {
    "1": 60, "3": 180, "5": 300, "15": 900, "30": 1800, "60": 3600, "240": 14400,
}

# Reconnect backoff bounds.
_BASE_RECONNECT_DELAY = 1.0
_MAX_RECONNECT_DELAY = 60.0

# Telegram alert throttle table; key = (category, interval, kind).
_ALERT_THROTTLE_SEC = 300  # one alert per key per 5 minutes
_last_alert_at: Dict[str, float] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_or_replace_candle(d: deque, candle: dict) -> None:
    """Phase 3 Fix #1: intra-bar updates have the same `timestamp` as the
    last entry. Replace the last entry rather than appending so the deque
    holds distinct bars only."""
    if d and d[-1].get("timestamp") == candle.get("timestamp"):
        d[-1] = candle
    else:
        d.append(candle)


def candles_are_fresh(candles, interval: str, max_age_multiplier: float = 3.0) -> bool:
    """Phase 3 Fix #3 helper. True iff the newest candle is younger than
    `max_age_multiplier × one_bar_duration`. Used by the scan path to refuse
    to score symbols whose feed has gone stale.

    Default multiplier of 3 means: a 1m candle has up to ~3 minutes of grace
    before we call it stale. Generous enough to ride out brief WS hiccups,
    tight enough to catch a dead stream within minutes.
    """
    if not candles:
        return False
    try:
        last = candles[-1]
        last_ts_ms = int(last.get("timestamp", 0))
        if last_ts_ms <= 0:
            return False
        last_ts_sec = last_ts_ms / 1000.0
        age_sec = time.time() - last_ts_sec
        max_age = INTERVAL_SECONDS.get(str(interval), 60) * max_age_multiplier
        return 0 <= age_sec < max_age
    except Exception:
        return False


async def _throttled_alert(key: str, message: str) -> None:
    """Phase 3 Fix #5. Drop the message if we sent one with this key recently."""
    now = time.time()
    last = _last_alert_at.get(key, 0)
    if now - last < _ALERT_THROTTLE_SEC:
        return
    _last_alert_at[key] = now
    try:
        await send_error_to_telegram(message)
    except Exception as e:
        log(f"⚠️ throttled alert send failed: {e}", level="WARN")


def _normalize_candle(k: dict) -> Optional[dict]:
    """Validate and shape a single Bybit V5 kline payload entry."""
    try:
        return {
            "timestamp": int(k["start"]),
            "open":   k["open"],
            "high":   k["high"],
            "low":    k["low"],
            "close":  k["close"],
            "volume": k["volume"],
            # Carry the confirm flag through in case downstream wants to
            # distinguish unconfirmed (forming) bars from closed ones.
            "confirm": bool(k.get("confirm", False)),
        }
    except (KeyError, TypeError, ValueError) as e:
        log(f"⚠️ malformed kline payload: {e} | raw={k}", level="WARN")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# REST history fetch (unchanged semantics)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_candles_rest(symbol: str, interval: str = '1', limit: int = 200,
                              category: str = 'linear'):
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    safe_limit = max(1, min(int(limit), 200))
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(safe_limit),
    }

    resp = await signed_request("GET", "/v5/market/kline", params)
    if resp.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline error for {symbol} ({interval}m): {resp.get('retMsg')}")

    raw = resp.get("result", {}).get("list", []) or []

    # Bybit returns newest-first; reverse to oldest-first.
    candles = []
    for c in reversed(raw):
        candles.append({
            "timestamp": int(c[0]),
            "open":  str(c[1]),
            "high":  str(c[2]),
            "low":   str(c[3]),
            "close": str(c[4]),
            "volume": str(c[5]),
        })
    return candles


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket handler — one task per (category, interval) pair
# ─────────────────────────────────────────────────────────────────────────────

async def handle_stream(url: str, symbols, category: str, interval) -> None:
    # Bind interval into a local string immediately.
    _interval = str(interval)
    backoff = _BASE_RECONNECT_DELAY

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                args = [f"kline.{_interval}.{symbol}" for symbol in symbols
                        if symbol_category_map.get(symbol) == category]
                if not args:
                    return

                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                log(f"📡 Subscribed to {len(args)} {category.upper()} @ {_interval}m")

                # Phase 3 Fix #6/#7: seed on connect when the deque is sparse
                # OR its newest bar is stale. Now meaningful because Fix #1
                # ensures the deque actually contains distinct bars.
                seed_symbols = [s for s in symbols
                                if symbol_category_map.get(s, 'linear') == category]
                for sym in seed_symbols:
                    try:
                        d = live_candles[sym][_interval]
                        need_seed = len(d) < 50 or not candles_are_fresh(
                            d, _interval, max_age_multiplier=2.0
                        )
                        if need_seed:
                            hist = await fetch_candles_rest(
                                sym, interval=_interval, limit=100, category=category
                            )
                            if hist:
                                d.clear()
                                for c in hist:
                                    d.append(c)
                                log(f"🌱 {sym} [{_interval}m] seeded: {len(hist)} candles")
                    except Exception as seed_err:
                        log(f"⚠️ Seed failed {sym} {_interval}m: {seed_err}", level="WARN")

                # Receive loop — successful operation resets the backoff.
                first_message_received = False
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        if not first_message_received:
                            backoff = _BASE_RECONNECT_DELAY  # reset on first good message
                            first_message_received = True

                        data = json.loads(message)
                        topic = data.get("topic", "")
                        if not topic or "data" not in data:
                            continue

                        parts = topic.split(".")
                        if len(parts) < 3:
                            continue
                        interval_from_topic = parts[1]
                        symbol = parts[-1]

                        # Only process messages matching our subscribed interval
                        if interval_from_topic != _interval:
                            log(f"⚠️ Interval mismatch: got {interval_from_topic} on "
                                f"{_interval} stream, skipping", level="WARN")
                            continue

                        candles_payload = data["data"]
                        if isinstance(candles_payload, dict):
                            candles_payload = [candles_payload]

                        if not isinstance(candles_payload, list):
                            continue

                        deque_target = live_candles[symbol][interval_from_topic]
                        for k in candles_payload:
                            candle = _normalize_candle(k)
                            if candle is None:
                                continue
                            _append_or_replace_candle(deque_target, candle)

                        # Quieter log — full bar count line was spammy at 1m.
                        # Uncomment for debug:
                        # log(f"📈 {symbol} [{category}] @{interval_from_topic} "
                        #     f"updated | bars: {len(deque_target)}")

                    except asyncio.TimeoutError:
                        await _throttled_alert(
                            f"recv_timeout|{category}|{_interval}",
                            f"⚠️ No data for {category} {_interval}m in 30s — reconnecting",
                        )
                        log(f"⚠️ No data for {category} {_interval}m in 30s — reconnecting", level="WARN")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        await _throttled_alert(
                            f"conn_closed|{category}|{_interval}",
                            f"❌ WS closed unexpectedly for {category} {_interval}m: {e}",
                        )
                        log(f"❌ WS closed unexpectedly for {category} {_interval}m: {e}", level="ERROR")
                        break
                    except Exception as e:
                        await _throttled_alert(
                            f"recv_error|{category}|{_interval}",
                            f"❌ WS stream error in {category} {_interval}m: {e}",
                        )
                        log(f"❌ WS stream error in {category} {_interval}m: {e}", level="ERROR")
                        break

        except Exception as e:
            await _throttled_alert(
                f"connect_failed|{category}|{_interval}",
                f"❌ Connection failed for {category} {_interval}m: {e}",
            )
            log(f"❌ Connection failed for {category} {_interval}m: {e}", level="ERROR")

        # Phase 3 Fix #4: exponential backoff with cap.
        log(f"🔁 Reconnecting {category} {_interval}m in {backoff:.1f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, _MAX_RECONNECT_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: spawn one handler per interval
# ─────────────────────────────────────────────────────────────────────────────

async def stream_candles(symbols) -> None:
    futures_url = "wss://stream.bybit.com/v5/public/linear"
    tasks = [
        asyncio.create_task(handle_stream(futures_url, symbols, "linear", interval))
        for interval in SUPPORTED_INTERVALS
    ]
    # Phase 3 Fix #11: one stream dying must not take the others with it.
    await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "live_candles",
    "SUPPORTED_INTERVALS",
    "INTERVAL_SECONDS",
    "stream_candles",
    "handle_stream",
    "fetch_candles_rest",
    "candles_are_fresh",
    "_append_or_replace_candle",  # exported for tests
]
