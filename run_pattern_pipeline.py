#!/usr/bin/env python3
"""
run_pattern_pipeline.py
=======================
Standalone runner for the full pattern discovery pipeline.
Fetches data from Bybit, mines rules, exports JSON + report.

Place this file in your bot directory (same folder as main.py, features.py, etc.)

Run:
    python run_pattern_pipeline.py
    python run_pattern_pipeline.py --top 50 --days 90 --min-sample 100
    python run_pattern_pipeline.py --symbols BTCUSDT,ETHUSDT --days 180

Output files:
    rules_export.json    -> Copy to Pine Script manually
    report.md            -> Human-readable analysis
    pipeline_debug.log   -> Full debug log
"""

import asyncio
import aiohttp
import argparse
import json
import sys
import os
import time
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any

# ── Make sure your bot's logger is used if available ──────────
try:
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")

# ── Import our pipeline modules ────────────────────────────────
from features import compute_features
from label_events import label_events_multi_horizon, HORIZON_MINUTES, THRESHOLDS
from miner import RuleMiner, build_samples
from export_rules import export_rules, generate_report

# ────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit here or use CLI args)
# ────────────────────────────────────────────────────────────────
BYBIT_API        = "https://api.bybit.com"
RATE_DELAY       = 0.12        # seconds between Bybit requests

DEFAULT_TOP_N    = 50          # top N symbols by 24h turnover
DEFAULT_DAYS     = 90          # history to fetch per symbol
DEFAULT_CANDLE_LIMIT = 1000    # candles per REST request (Bybit max=1000)
DEFAULT_TF       = "15"        # event detection timeframe
DEFAULT_MIN_SAMPLE = 100       # minimum triggers for a rule to be valid
DEFAULT_MIN_PRECISION = 0.52   # minimum precision (win rate)

BASE_TF     = "15"
CONFIRM_TF1 = "60"
CONFIRM_TF2 = "240"

OUTPUT_JSON   = "rules_export.json"
OUTPUT_REPORT = "report.md"
OUTPUT_DEBUG  = "pipeline_debug.log"
# ────────────────────────────────────────────────────────────────


# ── BYBIT DATA FETCHERS ────────────────────────────────────────

async def fetch_top_symbols(session: aiohttp.ClientSession, top_n: int = 50) -> List[str]:
    """Fetch top USDT perpetual symbols by 24h turnover."""
    url = f"{BYBIT_API}/v5/market/tickers"
    params = {"category": "linear"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
        tickers = data.get("result", {}).get("list", [])
        usdt = [t for t in tickers if str(t.get("symbol", "")).endswith("USDT")]
        usdt.sort(key=lambda t: float(t.get("turnover24h") or 0), reverse=True)
        symbols = [t["symbol"] for t in usdt[:top_n]]
        log(f"📡 Fetched {len(symbols)} top symbols")
        return symbols
    except Exception as e:
        log(f"❌ fetch_top_symbols failed: {e}", level="ERROR")
        return []


async def fetch_candles_paged(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int,
    limit_per_req: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch up to `days` days of candles for a symbol/interval via paged REST.
    Bybit returns newest first; we reverse to oldest first.
    """
    url   = f"{BYBIT_API}/v5/market/kline"
    start = int((time.time() - days * 86400) * 1000)
    end   = int(time.time() * 1000)

    all_candles: List[Dict] = []
    cur_end = end
    max_pages = 20  # safety limit

    for _ in range(max_pages):
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": interval,
            "end":      str(cur_end),
            "limit":    str(limit_per_req),
        }
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=20)) as resp:
                data = await resp.json()

            if data.get("retCode") != 0:
                break

            klines = data.get("result", {}).get("list", [])
            if not klines:
                break

            for k in klines:
                ts = int(k[0])
                if ts < start:
                    continue
                all_candles.append({
                    "ts":     ts,
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]),
                })

            oldest_ts = int(klines[-1][0])
            if oldest_ts <= start:
                break
            cur_end = oldest_ts - 1
            await asyncio.sleep(RATE_DELAY)

        except asyncio.TimeoutError:
            log(f"⚠️ Timeout fetching {symbol} {interval}m", level="WARN")
            break
        except Exception as e:
            log(f"⚠️ Error fetching {symbol} {interval}m: {e}", level="WARN")
            break

    # Deduplicate + sort oldest first
    seen = set()
    unique = []
    for c in all_candles:
        if c["ts"] not in seen:
            seen.add(c["ts"])
            unique.append(c)
    unique.sort(key=lambda x: x["ts"])
    return unique


# ── TELEGRAM NOTIFICATION (uses your bot's sender if available) ──

async def notify_telegram(msg: str):
    try:
        from error_handler import send_telegram_message
        await send_telegram_message(msg)
    except Exception:
        try:
            from telegram_bot import send_telegram_message
            await send_telegram_message(msg)
        except Exception:
            log(f"[TELEGRAM] {msg}")


# ── DEBUG LOG FILE ─────────────────────────────────────────────

def write_debug(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(OUTPUT_DEBUG, "a") as f:
        f.write(line)


# ── MAIN PIPELINE ──────────────────────────────────────────────

async def run_pipeline(
    symbols:      List[str] = None,
    top_n:        int  = DEFAULT_TOP_N,
    days:         int  = DEFAULT_DAYS,
    tf:           str  = DEFAULT_TF,
    min_sample:   int  = DEFAULT_MIN_SAMPLE,
    min_precision: float = DEFAULT_MIN_PRECISION,
    window:       int  = 30,
    send_telegram: bool = True,
):
    started = datetime.now(timezone.utc)
    log("=" * 60)
    log("🚀 Starting Pattern Discovery Pipeline")
    log(f"   Days={days} | TF={tf}m | MinSample={min_sample} | MinPrecision={min_precision}")
    log("=" * 60)

    # Clear debug log
    with open(OUTPUT_DEBUG, "w") as f:
        f.write(f"Pipeline started {started.isoformat()}\n")

    # Step 1: Get symbols
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        if not symbols:
            symbols = await fetch_top_symbols(session, top_n)
        if not symbols:
            log("❌ No symbols to scan. Aborting.", level="ERROR")
            return

        log(f"\n📊 Step 1: Fetching candles for {len(symbols)} symbols ({days} days)...")

        candles_by_symbol: Dict[str, List[Dict]] = {}
        events_by_symbol:  Dict[str, List[Dict]] = {}
        failed = []

        for i, sym in enumerate(symbols):
            pct = (i + 1) / len(symbols) * 100
            print(f"  [{i+1:3d}/{len(symbols)}] ({pct:5.1f}%) {sym}...", end="", flush=True)

            try:
                candles = await fetch_candles_paged(session, sym, tf, days)
                if len(candles) < 200:
                    print(f" ⚠️ only {len(candles)} candles, skip")
                    failed.append(sym)
                    continue

                candles_by_symbol[sym] = candles
                events = label_events_multi_horizon(
                    candles,
                    tf=tf,
                    horizons=HORIZON_MINUTES,
                    thresholds=THRESHOLDS,
                    cooldown_bars=5
                )
                events_by_symbol[sym] = events

                n_pump = sum(1 for e in events if e["direction"] == "pump")
                n_dump = sum(1 for e in events if e["direction"] == "dump")
                print(f" ✅ {len(candles)} bars | pump={n_pump} dump={n_dump}")
                write_debug(f"{sym}: {len(candles)} candles, {n_pump} pumps, {n_dump} dumps")

            except Exception as e:
                print(f" ❌ {e}")
                write_debug(f"{sym}: ERROR {e}\n{traceback.format_exc()}")
                failed.append(sym)

            await asyncio.sleep(RATE_DELAY)

    log(f"\n✅ Data collected: {len(candles_by_symbol)} symbols OK, {len(failed)} failed")

    # Step 2: Build samples
    log(f"\n📊 Step 2: Building feature samples (window={window} bars)...")
    samples = build_samples(candles_by_symbol, events_by_symbol, window_to_use=window)

    n_pumps = sum(1 for s in samples if s["direction"] == "pump")
    n_dumps = sum(1 for s in samples if s["direction"] == "dump")
    n_neg   = len(samples) - n_pumps - n_dumps

    log(f"   Total samples: {len(samples):,} | pumps: {n_pumps:,} | dumps: {n_dumps:,} | non-events: {n_neg:,}")
    write_debug(f"Samples: total={len(samples)}, pumps={n_pumps}, dumps={n_dumps}, neg={n_neg}")

    if len(samples) < 500:
        log("⚠️ Very few samples. Try more symbols or longer days.", level="WARN")

    # Step 3: Mine rules
    log(f"\n📊 Step 3: Mining rules (min_sample={min_sample}, min_precision={min_precision})...")

    miner = RuleMiner(
        min_sample_size  = min_sample,
        min_precision    = min_precision,
        max_fpr          = 0.20,
        max_conditions   = 6,
        top_k            = 20,
    )

    bucket_results = {}

    for direction in ["pump", "dump"]:
        for bucket in [5.0, 10.0, 20.0]:
            key = f"{direction}_{int(bucket)}"
            log(f"\n  Mining {direction.upper()} >= {bucket}%...")
            try:
                rules = miner.mine(samples, direction=direction, bucket=bucket, n_wf_splits=5)
                bucket_results[key] = rules
                log(f"  → {len(rules)} rules found")
                write_debug(f"{key}: {len(rules)} rules")
            except Exception as e:
                log(f"  ❌ Mining failed for {key}: {e}", level="ERROR")
                write_debug(f"{key}: MINING ERROR {e}\n{traceback.format_exc()}")
                bucket_results[key] = []

    # Step 4: Export
    log(f"\n📊 Step 4: Exporting rules...")

    # Primary: pump5 = long, dump5 = short (can be changed)
    long_rules  = bucket_results.get("pump_5", [])[:10]
    short_rules = bucket_results.get("dump_5", [])[:10]

    metadata = {
        "symbols_scanned": len(candles_by_symbol),
        "symbols_failed":  len(failed),
        "days_back":       days,
        "tf":              tf,
        "window":          window,
        "min_sample":      min_sample,
        "min_precision":   min_precision,
        "total_samples":   len(samples),
        "n_pumps":         n_pumps,
        "n_dumps":         n_dumps,
    }
    for key, rules in bucket_results.items():
        metadata[f"{key}_rules_found"] = len(rules)

    export = export_rules(
        long_rules  = long_rules,
        short_rules = short_rules,
        output_path = OUTPUT_JSON,
        base_tf     = BASE_TF,
        confirm_tf1 = CONFIRM_TF1,
        confirm_tf2 = CONFIRM_TF2,
        metadata    = metadata,
    )

    # Also save all bucket results separately
    all_rules_path = "all_rules.json"
    with open(all_rules_path, "w") as f:
        json.dump(bucket_results, f, indent=2)
    log(f"   All bucket rules saved -> {all_rules_path}")

    # Step 5: Report
    log(f"\n📊 Step 5: Generating report...")
    report_text = generate_report(export, samples, OUTPUT_REPORT)

    # ── PRINT SUMMARY TO CONSOLE ──────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    summary_lines = [
        "",
        "=" * 60,
        "✅ PIPELINE COMPLETE",
        "=" * 60,
        f"  Runtime:          {elapsed:.0f}s",
        f"  Symbols scanned:  {len(candles_by_symbol)} / {len(symbols)}",
        f"  Total samples:    {len(samples):,}",
        f"  Pump events (5%): {n_pumps:,}",
        f"  Dump events (5%): {n_dumps:,}",
        "",
        "  Rules discovered:",
    ]
    for key, rules in bucket_results.items():
        summary_lines.append(f"    {key:<15} → {len(rules)} rules")
    summary_lines += [
        "",
        f"  Output files:",
        f"    {OUTPUT_JSON}    ← Pine Script rules",
        f"    {all_rules_path}        ← All bucket rules",
        f"    {OUTPUT_REPORT}          ← Analysis report",
        f"    {OUTPUT_DEBUG}   ← Debug log",
        "=" * 60,
    ]

    for line in summary_lines:
        print(line)
    write_debug("\n".join(summary_lines))

    # ── TOP RULES PREVIEW ─────────────────────────────────────
    if long_rules:
        print("\n🟢 TOP LONG RULES (pump ≥ 5%):")
        for r in long_rules[:3]:
            stats = r.get("stats", {})
            wf    = r.get("wf_validation", {})
            print(f"  {r['name']}")
            print(f"    precision={stats.get('precision', 0):.3f} | "
                  f"wf_mean={wf.get('wf_precision_mean', 0):.3f} ± {wf.get('wf_precision_std', 0):.3f} | "
                  f"recall={stats.get('recall', 0):.3f} | "
                  f"fpr={stats.get('fpr', 0):.3f} | "
                  f"n={stats.get('sample_size', 0)} | "
                  f"avg_move={stats.get('avg_move', 0):.2f}%")
            print(f"    conditions: {len(r['conditions'])}")
            for c in r["conditions"]:
                print(f"      • {c['feature']} {c['operator']} {c.get('threshold', '')}")

    if short_rules:
        print("\n🔴 TOP SHORT RULES (dump ≥ 5%):")
        for r in short_rules[:3]:
            stats = r.get("stats", {})
            wf    = r.get("wf_validation", {})
            print(f"  {r['name']}")
            print(f"    precision={stats.get('precision', 0):.3f} | "
                  f"wf_mean={wf.get('wf_precision_mean', 0):.3f} ± {wf.get('wf_precision_std', 0):.3f} | "
                  f"recall={stats.get('recall', 0):.3f} | "
                  f"fpr={stats.get('fpr', 0):.3f} | "
                  f"n={stats.get('sample_size', 0)} | "
                  f"avg_move={stats.get('avg_move', 0):.2f}%")
            print(f"    conditions: {len(r['conditions'])}")
            for c in r["conditions"]:
                print(f"      • {c['feature']} {c['operator']} {c.get('threshold', '')}")

    # ── TELEGRAM SUMMARY ──────────────────────────────────────
    if send_telegram:
        tg_msg = (
            f"✅ *Pattern Pipeline Complete*\n"
            f"Symbols: {len(candles_by_symbol)} | Samples: {len(samples):,}\n"
            f"Long rules: {len(long_rules)} | Short rules: {len(short_rules)}\n"
        )
        for key, rules in bucket_results.items():
            tg_msg += f"`{key}`: {len(rules)} rules\n"
        tg_msg += f"\nFiles: `{OUTPUT_JSON}`, `{OUTPUT_REPORT}`"
        await notify_telegram(tg_msg)

    return export, bucket_results


# ── CLI ENTRY POINT ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pattern Discovery Pipeline")
    parser.add_argument("--symbols",     type=str,   default=None,
                        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--top",         type=int,   default=DEFAULT_TOP_N,
                        help=f"Top N symbols by turnover (default {DEFAULT_TOP_N})")
    parser.add_argument("--days",        type=int,   default=DEFAULT_DAYS,
                        help=f"Days of history (default {DEFAULT_DAYS})")
    parser.add_argument("--tf",          type=str,   default=DEFAULT_TF,
                        help=f"Entry timeframe in minutes (default {DEFAULT_TF})")
    parser.add_argument("--min-sample",  type=int,   default=DEFAULT_MIN_SAMPLE,
                        help=f"Min triggers for a rule (default {DEFAULT_MIN_SAMPLE})")
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION,
                        help=f"Min precision (default {DEFAULT_MIN_PRECISION})")
    parser.add_argument("--window",      type=int,   default=30,
                        help="Feature window size in bars (default 30)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Disable Telegram notifications")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    symbol_list = None
    if args.symbols:
        symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    asyncio.run(run_pipeline(
        symbols       = symbol_list,
        top_n         = args.top,
        days          = args.days,
        tf            = args.tf,
        min_sample    = args.min_sample,
        min_precision = args.min_precision,
        window        = args.window,
        send_telegram = not args.no_telegram,
    ))
