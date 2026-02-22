#!/usr/bin/env python3
"""
export_rules.py
===============
Takes mined rule dicts and exports them to the Pine-compatible JSON schema.
Also generates report.md.

Usage:
    python export_rules.py
    or import and call export_rules(long_rules, short_rules, output_path)
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


# ── CONDITION TYPE MAPPING ────────────────────────────────────
# Maps (feature, operator) -> Pine condition type and parameters

def _condition_to_pine(cond: Dict) -> Optional[Dict]:
    """
    Convert a miner condition dict to Pine export schema condition.
    Returns None if not mappable (Pine will skip it).
    """
    feat = cond["feature"]
    op   = cond["operator"]
    thr  = cond.get("threshold", 0.5)

    # BB squeeze
    if feat == "bb_squeeze" and op == "is_true":
        return {"type": "bb_squeeze", "bb_squeezed": True}
    if feat == "bb_width_pct" and op == "lt":
        return {"type": "bb_squeeze", "bb_width_pct_lt": thr}

    # ATR contraction
    if feat in ("atr_contracting", "atr_ratio_lt_09") and op == "is_true":
        return {"type": "atr_contract", "atr_ratio_lt": 0.9}
    if feat == "atr_ratio" and op == "lt":
        return {"type": "atr_contract", "atr_ratio_lt": thr}

    # Trend alignment - EMA
    if feat == "ema50_gt_ema200" and op == "is_true":
        return {"type": "trend_align", "ema50_gt_ema200": True}
    if feat == "ema50_gt_ema200" and op == "is_false":
        return {"type": "trend_align", "ema50_lt_ema200": True}
    if feat == "close_gt_ema50" and op == "is_true":
        return {"type": "trend_align", "close_gt_ema50": True}
    if feat == "close_gt_ema200" and op == "is_true":
        return {"type": "trend_align", "close_gt_ema200": True}
    if feat == "ema_golden_cross" and op == "is_true":
        return {"type": "trend_align", "golden_cross": True}
    if feat == "ema_death_cross" and op == "is_true":
        return {"type": "trend_align", "death_cross": True}

    # Supertrend
    if feat == "supertrend_bull" and op == "is_true":
        return {"type": "trend_align", "supertrend_bull": True}
    if feat == "supertrend_bear" and op == "is_true":
        return {"type": "trend_align", "supertrend_bear": True}
    if feat == "supertrend_flipped_bull" and op == "is_true":
        return {"type": "trend_align", "supertrend_flipped_bull": True}
    if feat == "supertrend_flipped_bear" and op == "is_true":
        return {"type": "trend_align", "supertrend_flipped_bear": True}

    # Volume spike
    if feat == "volume_spike_18" and op == "is_true":
        return {"type": "volume_spike", "vol_sma_mult_gt": 1.8}
    if feat == "volume_spike_25" and op == "is_true":
        return {"type": "volume_spike", "vol_sma_mult_gt": 2.5}
    if feat == "volume_spike_ratio" and op == "gt":
        return {"type": "volume_spike", "vol_sma_mult_gt": thr}
    if feat == "volume_contracting" and op == "is_true":
        return {"type": "volume_contraction", "vol_contracting": True}

    # Liquidity sweeps
    if feat == "liquidity_sweep_high" and op == "is_true":
        return {"type": "liquidity_sweep_high", "swing_lookback": 20}
    if feat == "liquidity_sweep_low" and op == "is_true":
        return {"type": "liquidity_sweep_low", "swing_lookback": 20}

    # RSI / Momentum
    if feat == "rsi" and op == "gt":
        return {"type": "momentum_confirm", "rsi_gt": thr}
    if feat == "rsi" and op == "lt":
        return {"type": "momentum_confirm", "rsi_lt": thr}
    if feat == "rsi_oversold" and op == "is_true":
        return {"type": "momentum_confirm", "rsi_lt": 30}
    if feat == "rsi_overbought" and op == "is_true":
        return {"type": "momentum_confirm", "rsi_gt": 70}

    # MACD
    if feat == "macd_hist" and op == "gt":
        return {"type": "momentum_confirm", "macd_hist_gt": thr}
    if feat == "macd_hist" and op == "lt":
        return {"type": "momentum_confirm", "macd_hist_lt": thr}
    if feat == "macd_cross_up" and op == "is_true":
        return {"type": "momentum_confirm", "macd_cross_up": True}
    if feat == "macd_cross_down" and op == "is_true":
        return {"type": "momentum_confirm", "macd_cross_down": True}

    # Breakout
    if feat == "breakout_up" and op == "is_true":
        return {"type": "breakout", "direction": "up", "lookback": 20}
    if feat == "breakout_down" and op == "is_true":
        return {"type": "breakout", "direction": "down", "lookback": 20}

    # Range compression
    if feat in ("range_compressed_vs_prior",) and op == "is_true":
        return {"type": "range_compression", "compressed": True}
    if feat == "range_compression" and op == "lt":
        return {"type": "range_compression", "ratio_lt": thr}

    # OBV
    if feat == "obv_rising" and op == "is_true":
        return {"type": "obv_trend", "rising": True}
    if feat == "obv_falling" and op == "is_true":
        return {"type": "obv_trend", "falling": True}

    # Candle patterns
    if feat == "candle_bull_engulf" and op == "is_true":
        return {"type": "candle_pattern", "pattern": "bull_engulf"}
    if feat == "candle_bear_engulf" and op == "is_true":
        return {"type": "candle_pattern", "pattern": "bear_engulf"}
    if feat == "candle_bull_pinbar" and op == "is_true":
        return {"type": "candle_pattern", "pattern": "bull_pinbar"}
    if feat == "candle_bear_pinbar" and op == "is_true":
        return {"type": "candle_pattern", "pattern": "bear_pinbar"}

    # Fallback: export raw
    return {"type": "raw", "feature": feat, "operator": op, "threshold": thr}


def _rule_to_pine_format(rule: Dict,
                          direction: str,
                          base_tf: str = "15",
                          confirm_tf1: str = "60",
                          confirm_tf2: str = "240") -> Dict:
    """
    Convert a miner rule dict to Pine export schema rule.
    Assigns timeframes heuristically:
    - Conditions involving trend/higher context -> confirm_tf1 or tf2
    - Entry conditions -> base_tf
    """
    trend_features = {
        "ema50_gt_ema200", "close_gt_ema50", "close_gt_ema200",
        "ema_golden_cross", "ema_death_cross", "supertrend_bull",
        "supertrend_bear", "supertrend_flipped_bull", "supertrend_flipped_bear",
    }

    pine_conditions = []
    for cond in rule.get("conditions", []):
        pine_cond = _condition_to_pine(cond)
        if pine_cond is None:
            continue
        feat = cond["feature"]
        # Assign TF
        if feat in trend_features:
            pine_cond["tf"] = confirm_tf2
        elif feat.startswith("ema") or feat.startswith("supertrend"):
            pine_cond["tf"] = confirm_tf1
        else:
            pine_cond["tf"] = base_tf
        pine_conditions.append(pine_cond)

    stats = rule.get("stats", {})
    wf    = rule.get("wf_validation", {})

    return {
        "name":      rule["name"],
        "direction": direction,
        "weight":    rule.get("weight", 1.0),
        "conditions": pine_conditions,
        "min_score":  round(rule.get("weight", 1.0) * len(pine_conditions) * 0.6, 2),
        "meta": {
            "precision":          stats.get("precision"),
            "recall":             stats.get("recall"),
            "fpr":                stats.get("fpr"),
            "avg_move_pct":       stats.get("avg_move"),
            "median_move_pct":    stats.get("median_move"),
            "sample_size":        stats.get("sample_size"),
            "wf_precision_mean":  wf.get("wf_precision_mean"),
            "wf_precision_std":   wf.get("wf_precision_std"),
        }
    }


def export_rules(
    long_rules:  List[Dict],
    short_rules: List[Dict],
    output_path: str = "rules_export.json",
    base_tf:     str = "15",
    confirm_tf1: str = "60",
    confirm_tf2: str = "240",
    version:     str = "1.0",
    metadata:    Optional[Dict] = None
) -> Dict:
    """
    Export long and short rules to Pine-compatible JSON.
    Returns the exported dict.
    """
    pine_long  = [_rule_to_pine_format(r, "long",  base_tf, confirm_tf1, confirm_tf2) for r in long_rules]
    pine_short = [_rule_to_pine_format(r, "short", base_tf, confirm_tf1, confirm_tf2) for r in short_rules]

    export = {
        "version":     version,
        "generated":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_tf":     base_tf,
        "confirm_tf1": confirm_tf1,
        "confirm_tf2": confirm_tf2,
        "long_rules":  pine_long,
        "short_rules": pine_short,
        "metadata":    metadata or {},
    }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[Export] Wrote {len(pine_long)} long + {len(pine_short)} short rules -> {output_path}")
    return export


def generate_report(
    export: Dict,
    all_samples: List[Dict],
    output_path: str = "report.md"
) -> str:
    """
    Generate a Markdown report summarizing the exported rules and stats.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_long  = len(export.get("long_rules",  []))
    n_short = len(export.get("short_rules", []))

    # Dataset stats
    n_total  = len(all_samples)
    n_pumps  = sum(1 for s in all_samples if s.get("direction") == "pump")
    n_dumps  = sum(1 for s in all_samples if s.get("direction") == "dump")
    n_neg    = n_total - n_pumps - n_dumps

    lines = [
        f"# Pattern Mining Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Base TF:** {export['base_tf']}m | "
        f"**Confirm TF1:** {export['confirm_tf1']}m | "
        f"**Confirm TF2:** {export['confirm_tf2']}m",
        f"",
        f"## Dataset Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total samples | {n_total:,} |",
        f"| Pump events   | {n_pumps:,} |",
        f"| Dump events   | {n_dumps:,} |",
        f"| Non-events    | {n_neg:,} |",
        f"| Base rate pump | {n_pumps/n_total*100:.2f}% |" if n_total else "| Base rate | N/A |",
        f"",
        f"## Exported Rules",
        f"",
        f"- **Long rules:** {n_long}",
        f"- **Short rules:** {n_short}",
        f"",
    ]

    for section, rules in [("LONG", export.get("long_rules", [])),
                            ("SHORT", export.get("short_rules", []))]:
        lines.append(f"## {section} Rules Detail")
        lines.append("")
        for rule in rules:
            meta = rule.get("meta", {})
            lines += [
                f"### {rule['name']}",
                f"",
                f"**Weight:** {rule['weight']} | **Min Score:** {rule['min_score']}",
                f"",
                f"**Stats:**",
                f"| Metric | Train | WF Mean |",
                f"|--------|-------|---------|",
                f"| Precision | {meta.get('precision', 'N/A'):.3f} | "
                f"{meta.get('wf_precision_mean', 'N/A'):.3f} |" if meta.get('precision') is not None else
                "| Precision | N/A | N/A |",
                f"| Recall | {meta.get('recall', 0):.3f} | {meta.get('wf_precision_mean', 0):.3f} |",
                f"| FPR | {meta.get('fpr', 0):.3f} | - |",
                f"| Avg Move % | {meta.get('avg_move_pct', 0):.2f}% | - |",
                f"| Median Move % | {meta.get('median_move_pct', 0):.2f}% | - |",
                f"| Sample Size | {meta.get('sample_size', 0)} | - |",
                f"",
                f"**Conditions ({len(rule['conditions'])}):**",
                f"",
            ]
            for i, cond in enumerate(rule["conditions"], 1):
                ctype = cond.get("type", "?")
                tf    = cond.get("tf", "?")
                # Pretty print remaining keys
                details = {k: v for k, v in cond.items() if k not in ("type", "tf")}
                lines.append(f"{i}. `[TF:{tf}m]` **{ctype}** — {json.dumps(details)}")
            lines.append("")

    lines += [
        "## Usage",
        "",
        "1. Copy `rules_export.json` to your TradingView Pine Script.",
        "2. Manually implement each condition in Pine v6 using the type/parameter schema.",
        "3. Sum weights of matching rules; fire signal when `total_score >= min_score`.",
        "",
        "---",
        f"*Auto-generated by export_rules.py*",
    ]

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"[Report] Wrote report -> {output_path}")
    return report_text


# ── MAIN (full pipeline runner) ────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import aiohttp
    from label_events import label_events_multi_horizon
    from miner import RuleMiner, build_samples

    # ── CONFIG ────────────────────────────────────────────────
    SYMBOLS_TO_SCAN = 30      # top N by volume
    DAYS_BACK       = 90
    TF              = "15"    # entry timeframe for events
    BASE_TF         = "15"
    CONFIRM_TF1     = "60"
    CONFIRM_TF2     = "240"
    OUTPUT_JSON     = "rules_export.json"
    OUTPUT_REPORT   = "report.md"
    MIN_SAMPLE      = 50      # lower for testing; use 200 in production
    # ──────────────────────────────────────────────────────────

    BYBIT_API = "https://api.bybit.com"

    async def fetch_symbols(session: aiohttp.ClientSession, top_n: int = 30) -> List[str]:
        url = f"{BYBIT_API}/v5/market/tickers"
        params = {"category": "linear"}
        async with session.get(url, params=params) as resp:
            data = await resp.json()
        tickers = data.get("result", {}).get("list", [])
        usdt = [t for t in tickers if t.get("symbol", "").endswith("USDT")]
        usdt.sort(key=lambda t: float(t.get("turnover24h") or 0), reverse=True)
        return [t["symbol"] for t in usdt[:top_n]]

    async def fetch_candles(session: aiohttp.ClientSession,
                            symbol: str, interval: str, limit: int = 1000) -> List[Dict]:
        url = f"{BYBIT_API}/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        async with session.get(url, params=params) as resp:
            data = await resp.json()
        raw = data.get("result", {}).get("list", [])
        # Bybit returns [ts, open, high, low, close, volume, turnover], newest first
        candles = []
        for row in reversed(raw):
            candles.append({
                "ts":     int(row[0]),
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })
        return candles

    async def main():
        print("[Main] Starting full pipeline...")

        async with aiohttp.ClientSession() as session:
            symbols = await fetch_symbols(session, SYMBOLS_TO_SCAN)
            print(f"[Main] Scanning {len(symbols)} symbols: {symbols[:5]}...")

            candles_by_symbol: Dict[str, List[Dict]] = {}
            events_by_symbol:  Dict[str, List[Dict]] = {}

            for sym in symbols:
                try:
                    candles = await fetch_candles(session, sym, TF, limit=1000)
                    if len(candles) < 100:
                        continue
                    candles_by_symbol[sym] = candles
                    events = label_events_multi_horizon(candles, tf=TF)
                    events_by_symbol[sym]  = events
                    print(f"  {sym}: {len(candles)} candles, {len(events)} events")
                    await asyncio.sleep(0.12)  # rate limit
                except Exception as e:
                    print(f"  {sym}: ERROR {e}")

        # Build samples
        print("[Main] Building samples...")
        samples = build_samples(candles_by_symbol, events_by_symbol, window_to_use=30)
        print(f"[Main] Total samples: {len(samples)}")

        # Mine
        miner = RuleMiner(min_sample_size=MIN_SAMPLE, min_precision=0.55)

        pump5_rules  = miner.mine(samples, direction="pump", bucket=5.0)
        dump5_rules  = miner.mine(samples, direction="dump", bucket=5.0)
        pump10_rules = miner.mine(samples, direction="pump", bucket=10.0)
        dump10_rules = miner.mine(samples, direction="dump", bucket=10.0)

        # Use pump5 as long_rules, dump5 as short_rules for primary export
        long_rules  = pump5_rules[:10]
        short_rules = dump5_rules[:10]

        # Export
        export = export_rules(
            long_rules  = long_rules,
            short_rules = short_rules,
            output_path = OUTPUT_JSON,
            base_tf     = BASE_TF,
            confirm_tf1 = CONFIRM_TF1,
            confirm_tf2 = CONFIRM_TF2,
            metadata    = {
                "symbols_scanned": len(symbols),
                "pump5_rules_found":  len(pump5_rules),
                "dump5_rules_found":  len(dump5_rules),
                "pump10_rules_found": len(pump10_rules),
                "dump10_rules_found": len(dump10_rules),
                "total_samples":      len(samples),
            }
        )

        # Report
        generate_report(export, samples, OUTPUT_REPORT)
        print("[Main] Done.")

    asyncio.run(main())
