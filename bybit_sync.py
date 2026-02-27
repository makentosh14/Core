# bybit_sync.py
# Sync bot's active trades with Bybit's actual open positions (linear perps)
import json, os
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from bybit_api import signed_request
from logger import log

# If your bot uses a different file name, change this:
PERSIST_PATH = "monitor_active_trades.json"

# -------------- Low-level IO -----------------

def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log(f"⚠️ SYNC: failed reading {path}: {e}", level="WARN")
        return {}

def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# -------------- Bybit fetchers -----------------

async def _fetch_positions_linear(settle_coin: str = "USDT", symbol: str | None = None):
    """
    Bybit v5 positions (linear). You must provide either symbol or settleCoin.
    - settle_coin: e.g., "USDT" for USDT-settled perps
    - symbol: set this if you want a single symbol (then settleCoin is not required)
    Returns a list of position dicts (may be empty).
    """
    params = {"category": "linear"}
    if symbol:
        params["symbol"] = symbol
    else:
        params["settleCoin"] = settle_coin  # <-- REQUIRED when no symbol

    out = []
    cursor = None
    while True:
        if cursor:
            params["cursor"] = cursor
        resp = await signed_request("GET", "/v5/position/list", params)
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit position error: {resp.get('retMsg')}")
        result = resp.get("result", {}) or {}
        out.extend(result.get("list", []) or [])
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
    return out

async def _fetch_open_stop_orders(symbols: list[str], settle_coin: str = "USDT") -> dict:
    """
    Fetch open conditional/stop orders and index by symbol.
    When no symbol is given, include settleCoin for unified accounts.
    """
    params = {"category": "linear", "openOnly": "1", "settleCoin": settle_coin}
    resp = await signed_request("GET", "/v5/order/realtime", params)
    if resp.get("retCode") != 0:
        log(f"⚠️ SYNC: order/realtime error: {resp.get('retMsg')}", level="WARN")
        return {}
    open_orders = resp.get("result", {}).get("list", []) or []
    by_symbol = {}
    want = set(symbols or [])
    for o in open_orders:
        sym = o.get("symbol")
        if want and sym not in want:
            continue
        stop_type = (o.get("stopOrderType") or "").lower()
        trig = o.get("triggerPrice")
        if stop_type == "stoploss" and trig:
            try:
                by_symbol[sym] = {"sl": float(trig), "sl_order_id": o.get("orderId")}
            except Exception:
                pass
    return by_symbol


# -------------- Mapping -----------------

def _bybit_position_to_trade(p: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Map Bybit position to bot's active-trade schema.
    We keep bot fields you likely use in monitor/execute:
      symbol, trade_type, direction, entry_price, qty, sl, sl_order_id, tp1_hit, exited, timestamp
    """
    symbol = p.get("symbol")
    side = (p.get("side") or "").lower()            # 'buy' / 'sell'
    direction = "Long" if side == "buy" else "Short"
    try:
        qty = float(p.get("size") or 0)
    except Exception:
        qty = 0.0
    try:
        entry = float(p.get("avgPrice") or 0)
    except Exception:
        entry = 0.0

    trade = {
        "symbol": symbol,
        "trade_type": p.get("tradeType") or "CoreIntraday",
        "direction": direction,
        "entry_price": entry,
        "qty": qty,
        "sl": None,
        "sl_order_id": None,
        "tp1": None,
        "tp1_hit": False,
        "trailing_pct": None,
        "exited": False,
        "timestamp": _now_str(),
        "source": "bybit_sync"
    }
    return symbol, trade

# -------------- Public API -----------------

async def sync_bot_with_bybit(send_telegram: bool = True) -> Dict[str, Any]:
    """
    Reconcile bot state with Bybit open positions:

    1) Pull open positions (linear) and map to bot schema.
    2) Pull open stop orders; attach SL + sl_order_id where found.
    3) Merge with disk JSON:
       - Add/update symbols that exist on Bybit
       - Mark 'exited' for symbols present in JSON but missing on Bybit
    4) Save JSON atomically.
    5) Update in-memory monitor.active_trades (if importable).
    6) Optionally notify via Telegram.

    Returns the updated JSON dict.
    """
    try:
        # 1) Positions from exchange
        positions = await _fetch_positions_linear(settle_coin="USDT")
        bybit_symbols = [p.get("symbol") for p in positions if p.get("symbol")]
        pos_map: Dict[str, Dict[str, Any]] = {}
        for p in positions:
            sym, trade = _bybit_position_to_trade(p)
            if sym:
                pos_map[sym] = trade

        # 2) Attach SL info best-effort
        try:
            sl_map = await _fetch_open_stop_orders(bybit_symbols, settle_coin="USDT") if bybit_symbols else {}
            for sym, slinfo in sl_map.items():
                if sym in pos_map:
                    pos_map[sym]["sl"] = slinfo.get("sl")
                    pos_map[sym]["sl_order_id"] = slinfo.get("sl_order_id")
        except Exception as e:
            log(f"⚠️ SYNC: stop-orders fetch failed: {e}", level="WARN")

        # 3) Merge with disk JSON
        disk = _read_json(PERSIST_PATH)
        updated: Dict[str, Any] = {}

        # add/update from Bybit
        for sym, t in pos_map.items():
            prev = disk.get(sym, {})
            # keep any extra bot fields, overwrite core ones with fresh exchange truth
            merged = {**prev, **t}
            updated[sym] = merged

        # mark exited for symbols on disk but missing on exchange
        changes_exit = []
        for sym, t in (disk.items() if isinstance(disk, dict) else []):
            if sym not in updated:
                t = dict(t) if isinstance(t, dict) else {}
                if not t.get("exited"):
                    t["exited"] = True
                    t["exit_reason"] = "bybit_sync_missing_on_exchange"
                    t["exit_time"] = _now_str()
                    changes_exit.append(sym)
                updated[sym] = t

        # 4) Persist
        _write_json_atomic(PERSIST_PATH, updated)

        # 5) Update in-memory monitor.active_trades
        try:
            from monitor import active_trades  # dict in your monitor module
            active_trades.clear()
            for s, tr in updated.items():
                if not tr.get("exited"):
                    active_trades[s] = tr
            log(f"🧠 SYNC: in-memory active_trades set to {len(active_trades)} open")
        except Exception as e:
            log(f"ℹ️ SYNC: monitor.active_trades not refreshed ({e})", level="INFO")

        # 6) Telegram notify (optional)
        if send_telegram:
            try:
                from telegram_bot import send_telegram_message
                opened = list(pos_map.keys())
                if changes_exit or opened:
                    msg_lines = ["🔁 <b>Bybit Sync</b>"]
                    if opened:
                        msg_lines.append("Open on exchange:")
                        msg_lines += [f"• {s}" for s in opened]
                    if changes_exit:
                        msg_lines.append("Closed on exchange (marked exited in bot):")
                        msg_lines += [f"• {s}" for s in changes_exit]
                    await send_telegram_message("\n".join(msg_lines))
            except Exception as e:
                log(f"⚠️ SYNC: telegram notify failed: {e}", level="WARN")

        log(f"✅ SYNC: reconciled {len(pos_map)} open position(s), {len(updated)} total records")
        return updated

    except Exception as e:
        log(f"❌ SYNC: failed: {e}", level="ERROR")
        return {}
