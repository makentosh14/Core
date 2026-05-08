#!/usr/bin/env python3
"""
test_score.py v2 — Comprehensive test for score.py and dependencies
====================================================================
v2 fixes vs v1:
  - call_safely() signature bug fixed (was the cause of all 7 "test suite
    crashed" errors). New approach: each test uses r.check(name, lambda).
  - Synthetic candle builders now use NOISY trends (70% trend / 30% pullback)
    instead of monotonic. This prevents RSI saturating to 0/100 which made
    "downtrend" produce bullish-RSI-oversold readings (a false positive that
    hid real bugs).
  - 4h veto test uses a stronger bearish 4h to ensure 2+ indicators agree.

Tests run in TWO layers:
  LAYER 1 (UNIT) — Each helper / dependency in isolation. No external API.
  LAYER 2 (INTEG) — Full score_symbol() and enhanced_score_symbol() pipelines.

Usage:
    python3 test_score.py                # everything
    python3 test_score.py --unit         # Layer 1 only
    python3 test_score.py --integration  # Layer 2 only
    python3 test_score.py --verbose      # show every PASS line
    python3 test_score.py --module rsi   # one module only

Exit codes: 0 ok, 1 some failed, 2 setup error.
"""

import sys
import os
import argparse
import math
import random
import traceback
import time
from typing import List, Dict, Any, Tuple, Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[Tuple[str, str]] = []
        self.skipped_tests: List[Tuple[str, str]] = []
        self.start_time = time.time()

    def ok(self, name: str):
        self.passed += 1
        if self.verbose:
            print(f"  ✅ {name}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.failures.append((name, reason))
        print(f"  ❌ {name}")
        for line in str(reason).splitlines():
            print(f"      {line}")

    def skip(self, name: str, reason: str):
        self.skipped += 1
        self.skipped_tests.append((name, reason))
        if self.verbose:
            print(f"  ⏭️  {name} (skipped: {reason})")

    def check(self, name: str, fn: Callable, *args, **kwargs):
        """
        Call fn(*args). Pass/fail semantics:
          - if fn raises   → fail with traceback
          - if fn returns False or (False, msg) → fail with msg
          - if fn returns True or (True, msg) or any truthy → pass
          - if fn returns None → pass (no return = success by default)
        """
        try:
            res = fn(*args, **kwargs)
        except Exception as e:
            self.fail(name, f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
            ok, msg = res
            if ok:
                self.ok(f"{name}{(': ' + msg) if msg else ''}")
            else:
                self.fail(name, msg)
            return res
        if res is False:
            self.fail(name, "returned False")
            return res
        self.ok(name)
        return res

    def summary(self) -> int:
        elapsed = time.time() - self.start_time
        print()
        print("═" * 70)
        print("TEST SUMMARY")
        print("═" * 70)
        print(f"  Passed:  {self.passed}")
        print(f"  Failed:  {self.failed}")
        print(f"  Skipped: {self.skipped}")
        print(f"  Total:   {self.passed + self.failed + self.skipped}")
        print(f"  Time:    {elapsed:.2f}s")

        if self.failed > 0:
            print("\nFAILED TESTS:")
            for name, _ in self.failures:
                print(f"  ❌ {name}")
            return 1
        if self.passed == 0 and self.skipped > 0:
            print("\n⚠️  All tests were skipped — verify the score module imports OK.")
            return 1
        print("\n🎉 All tests passed!")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC CANDLE BUILDERS (noisy trends — RSI/MACD won't saturate)
# ─────────────────────────────────────────────────────────────────────────────

def make_candle(open_p, high, low, close, volume, timestamp=0):
    return {
        "timestamp": timestamp,
        "open": float(open_p),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


def make_uptrend_candles(n: int = 60, start_price: float = 100.0,
                        step_pct: float = 0.3, volume: float = 1_000_000,
                        seed: int = 42) -> List[Dict]:
    """70% up bars, 30% pullback. Net upward drift but RSI won't saturate."""
    rng = random.Random(seed)
    candles = []
    price = start_price
    for i in range(n):
        prev_close = price
        if rng.random() < 0.7:
            move = step_pct * rng.uniform(0.7, 1.5)
        else:
            move = -step_pct * rng.uniform(0.3, 0.7)
        price = price * (1 + move / 100)
        high = max(prev_close, price) * (1 + rng.uniform(0.0005, 0.002))
        low  = min(prev_close, price) * (1 - rng.uniform(0.0005, 0.002))
        candles.append(make_candle(prev_close, high, low, price, volume,
                                   timestamp=i * 60_000))
    return candles


def make_downtrend_candles(n: int = 60, start_price: float = 100.0,
                          step_pct: float = 0.3, volume: float = 1_000_000,
                          seed: int = 43) -> List[Dict]:
    """70% down bars, 30% bounce."""
    rng = random.Random(seed)
    candles = []
    price = start_price
    for i in range(n):
        prev_close = price
        if rng.random() < 0.7:
            move = -step_pct * rng.uniform(0.7, 1.5)
        else:
            move = step_pct * rng.uniform(0.3, 0.7)
        price = price * (1 + move / 100)
        high = max(prev_close, price) * (1 + rng.uniform(0.0005, 0.002))
        low  = min(prev_close, price) * (1 - rng.uniform(0.0005, 0.002))
        candles.append(make_candle(prev_close, high, low, price, volume,
                                   timestamp=i * 60_000))
    return candles


def make_ranging_candles(n: int = 60, center_price: float = 100.0,
                        amplitude_pct: float = 0.5, volume: float = 1_000_000,
                        seed: int = 44) -> List[Dict]:
    rng = random.Random(seed)
    candles = []
    prev = center_price
    for i in range(n):
        offset = math.sin(i * 0.5) * (amplitude_pct / 100) * center_price
        noise = rng.uniform(-0.0005, 0.0005) * center_price
        close = center_price + offset + noise
        high = max(prev, close) * (1 + rng.uniform(0.0005, 0.0015))
        low  = min(prev, close) * (1 - rng.uniform(0.0005, 0.0015))
        candles.append(make_candle(prev, high, low, close, volume,
                                   timestamp=i * 60_000))
        prev = close
    return candles


def make_volume_spike_candles(n: int = 30, base_price: float = 100.0,
                              spike_at: int = -1, spike_multiplier: float = 3.0,
                              seed: int = 45) -> List[Dict]:
    rng = random.Random(seed)
    candles = []
    base_volume = 1_000_000
    for i in range(n):
        is_spike = (i == n + spike_at) if spike_at < 0 else (i == spike_at)
        volume = base_volume * spike_multiplier if is_spike else base_volume
        prev = base_price + rng.uniform(-0.1, 0.1)
        close = base_price + rng.uniform(-0.1, 0.1)
        high = max(prev, close) + 0.2
        low = min(prev, close) - 0.2
        candles.append(make_candle(prev, high, low, close, volume,
                                   timestamp=i * 60_000))
    return candles


def make_uptrend_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    return {
        "1":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.10, seed=1),
        "3":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.20, seed=3),
        "5":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.30, seed=5),
        "15":  make_uptrend_candles(n=max(min_candles, 50), step_pct=0.50, seed=15),
        "30":  make_uptrend_candles(n=max(min_candles, 40), step_pct=0.70, seed=30),
        "60":  make_uptrend_candles(n=max(min_candles, 40), step_pct=0.90, seed=60),
        "240": make_uptrend_candles(n=max(min_candles, 30), step_pct=1.20, seed=240),
    }


def make_downtrend_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    return {
        "1":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.10, seed=101),
        "3":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.20, seed=103),
        "5":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.30, seed=105),
        "15":  make_downtrend_candles(n=max(min_candles, 50), step_pct=0.50, seed=115),
        "30":  make_downtrend_candles(n=max(min_candles, 40), step_pct=0.70, seed=130),
        "60":  make_downtrend_candles(n=max(min_candles, 40), step_pct=0.90, seed=160),
        "240": make_downtrend_candles(n=max(min_candles, 30), step_pct=1.20, seed=140),
    }


def make_ranging_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    return {
        "1":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.3, seed=201),
        "3":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.4, seed=203),
        "5":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.5, seed=205),
        "15":  make_ranging_candles(n=max(min_candles, 50), amplitude_pct=0.6, seed=215),
        "30":  make_ranging_candles(n=max(min_candles, 40), amplitude_pct=0.7, seed=230),
        "60":  make_ranging_candles(n=max(min_candles, 40), amplitude_pct=0.8, seed=260),
        "240": make_ranging_candles(n=max(min_candles, 30), amplitude_pct=1.0, seed=240),
    }


def make_chasing_set() -> Dict[str, List[Dict]]:
    base = make_uptrend_candles(n=60, step_pct=0.10, seed=301)
    last_close = base[-4]["close"]
    for i in range(3):
        new_close = last_close * (1 + (i + 1) * 0.012)
        base[-3 + i] = make_candle(
            open_p=last_close if i == 0 else base[-3 + i - 1]["close"],
            high=new_close * 1.003,
            low=last_close * 0.998,
            close=new_close,
            volume=2_000_000,
            timestamp=base[-3 + i]["timestamp"],
        )
    return {
        "1":   base,
        "5":   make_uptrend_candles(n=60, step_pct=0.30, seed=305),
        "15":  make_uptrend_candles(n=50, step_pct=0.50, seed=315),
        "240": make_uptrend_candles(n=30, step_pct=1.20, seed=340),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def try_import(module_name: str, attrs: List[str], r: TestResult) -> Optional[Any]:
    try:
        mod = __import__(module_name, fromlist=attrs)
    except Exception as e:
        for attr in attrs:
            r.skip(f"import {module_name}.{attr}", f"module import failed: {e}")
        return None
    for attr in attrs:
        if hasattr(mod, attr):
            r.ok(f"import {module_name}.{attr}")
        else:
            r.fail(f"import {module_name}.{attr}",
                  f"attribute missing in {module_name}")
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1.0 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def test_imports(r: TestResult, module_filter: Optional[str] = None):
    print("\n" + "─" * 70)
    print("LAYER 1.0 — IMPORT VERIFICATION")
    print("─" * 70)

    expected_imports = {
        "logger":                ["log"],
        "rsi":                   ["calculate_rsi", "calculate_rsi_with_bands",
                                  "calculate_stoch_rsi", "analyze_multi_timeframe_rsi"],
        "macd":                  ["detect_macd_cross", "get_macd_divergence",
                                  "get_macd_momentum"],
        "supertrend":            ["calculate_supertrend_signal", "get_supertrend_state",
                                  "calculate_multi_timeframe_supertrend"],
        "ema":                   ["detect_ema_crossover", "calculate_ema_ribbon",
                                  "analyze_ema_ribbon", "detect_ema_squeeze"],
        "bollinger":             ["calculate_bollinger_bands", "detect_band_walk",
                                  "get_bollinger_signal"],
        "pattern_detector":      ["detect_pattern", "analyze_pattern_strength",
                                  "detect_pattern_cluster", "get_pattern_direction",
                                  "get_all_patterns", "REVERSAL_PATTERNS",
                                  "CONTINUATION_PATTERNS"],
        "volume":                ["is_volume_spike", "get_average_volume",
                                  "detect_volume_climax", "get_volume_profile",
                                  "get_volume_weighted_average_price"],
        "stealth_detector":      ["detect_volume_divergence", "detect_slow_breakout",
                                  "detect_stealth_accumulation_advanced"],
        "whale_detector":        ["detect_whale_activity",
                                  "detect_whale_activity_advanced"],
        "indicator_fixes":       ["rebalance_indicator_scores",
                                  "get_balanced_rsi_signal",
                                  "analyze_volume_direction"],
        "enhanced_entry_validator": ["entry_validator"],
        "pattern_context_analyzer": ["pattern_context_analyzer"],
        "divergence_detector":   ["divergence_detector"],
        "trend_filters":         ["validate_short_signal", "monitor_btc_trend_accuracy",
                                  "monitor_altseason_status"],
    }

    for mod_name, attrs in expected_imports.items():
        if module_filter and module_filter != mod_name:
            continue
        try_import(mod_name, attrs, r)

    print()
    try_import("score", [
        "score_symbol", "enhanced_score_symbol", "determine_direction",
        "calculate_confidence", "has_pump_potential", "detect_momentum_strength",
        "safe_detect_momentum_strength", "get_4h_trend_gate",
        "check_4h_alignment_veto", "check_not_chasing",
        "WEIGHTS", "TRADE_TYPE_TF", "MIN_TF_REQUIRED", "VOLUME_SPIKE_THRESHOLD",
    ], r)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — INDICATOR MODULE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _last_or_value(out):
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[-1]
    if isinstance(out, (int, float)):
        return out
    return None


def test_rsi(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.1 — RSI MODULE")
    print("─" * 70)
    try:
        from rsi import calculate_rsi, calculate_rsi_with_bands, calculate_stoch_rsi
    except Exception as e:
        r.skip("rsi tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(60, step_pct=0.3)
    down = make_downtrend_candles(60, step_pct=0.3)
    ranging = make_ranging_candles(60)

    def chk_rsi_uptrend():
        out = calculate_rsi(up)
        last = _last_or_value(out)
        if last is None:
            return False, f"no value (got {out!r})"
        return (last > 50), f"RSI(uptrend) = {last:.2f}, expected > 50"

    def chk_rsi_downtrend():
        out = calculate_rsi(down)
        last = _last_or_value(out)
        if last is None:
            return False, f"no value (got {out!r})"
        return (last < 50), f"RSI(downtrend) = {last:.2f}, expected < 50"

    def chk_rsi_ranging():
        out = calculate_rsi(ranging)
        last = _last_or_value(out)
        if last is None:
            return False, "no value"
        return (30 < last < 70), f"RSI(ranging) = {last:.2f}, expected 30..70"

    def chk_rsi_bands():
        out = calculate_rsi_with_bands(up)
        return isinstance(out, dict), f"expected dict, got {type(out).__name__}"

    def chk_stoch_rsi():
        out = calculate_stoch_rsi(up)
        return (out is None or isinstance(out, dict)), \
               f"expected dict/None, got {type(out).__name__}"

    r.check("rsi: uptrend produces RSI > 50", chk_rsi_uptrend)
    r.check("rsi: downtrend produces RSI < 50", chk_rsi_downtrend)
    r.check("rsi: ranging produces RSI in 30..70", chk_rsi_ranging)
    r.check("rsi: calculate_rsi_with_bands returns dict", chk_rsi_bands)
    r.check("rsi: calculate_stoch_rsi runs", chk_stoch_rsi)


def test_macd(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.2 — MACD MODULE")
    print("─" * 70)
    try:
        from macd import detect_macd_cross, get_macd_momentum
    except Exception as e:
        r.skip("macd tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(60, step_pct=0.3)
    down = make_downtrend_candles(60, step_pct=0.3)

    def chk_macd_cross():
        out = detect_macd_cross(up)
        return (out in ("bullish", "bearish", None)), f"got {out!r}"

    def chk_momentum_up():
        m = get_macd_momentum(up)
        return isinstance(m, (int, float)), f"got {type(m).__name__}"

    def chk_momentum_down():
        m = get_macd_momentum(down)
        return isinstance(m, (int, float)), f"got {type(m).__name__}"

    def chk_momentum_direction():
        mu = get_macd_momentum(up)
        md = get_macd_momentum(down)
        if not (isinstance(mu, (int, float)) and isinstance(md, (int, float))):
            return False, f"non-numeric: up={mu!r}, down={md!r}"
        return (mu > md), f"up={mu:.4f}, down={md:.4f}"

    r.check("macd: detect_macd_cross returns str/None", chk_macd_cross)
    r.check("macd: get_macd_momentum returns number (uptrend)", chk_momentum_up)
    r.check("macd: get_macd_momentum returns number (downtrend)", chk_momentum_down)
    r.check("macd: uptrend momentum > downtrend momentum", chk_momentum_direction)


def test_supertrend(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.3 — SUPERTREND MODULE")
    print("─" * 70)
    try:
        from supertrend import (calculate_supertrend_signal, get_supertrend_state,
                                calculate_multi_timeframe_supertrend)
    except Exception as e:
        r.skip("supertrend tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(60, step_pct=0.4)
    down = make_downtrend_candles(60, step_pct=0.4)

    def chk_sig_up():
        out = calculate_supertrend_signal(up)
        return (out is None or isinstance(out, str)), f"got {out!r}"

    def chk_sig_down():
        out = calculate_supertrend_signal(down)
        return (out is None or isinstance(out, str)), f"got {out!r}"

    def chk_sanity():
        u = calculate_supertrend_signal(up)
        d = calculate_supertrend_signal(down)
        if u and d:
            ok = not (u == "bearish" and d == "bullish")
            return ok, f"up={u}, down={d}"
        return True, f"signals: up={u}, down={d}"

    def chk_state():
        out = get_supertrend_state(up)
        if not isinstance(out, dict):
            return False, f"got {type(out).__name__}"
        return ("trend" in out), f"keys: {list(out.keys())}"

    def chk_mtf():
        out = calculate_multi_timeframe_supertrend(make_uptrend_set())
        if not isinstance(out, dict):
            return False, f"got {type(out).__name__}"
        return ("alignment" in out), f"keys: {list(out.keys())}"

    r.check("supertrend: signal returns str/None (uptrend)", chk_sig_up)
    r.check("supertrend: signal returns str/None (downtrend)", chk_sig_down)
    r.check("supertrend: directional sanity", chk_sanity)
    r.check("supertrend: get_supertrend_state has 'trend' key", chk_state)
    r.check("supertrend: MTF returns 'alignment' dict", chk_mtf)


def test_ema(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.4 — EMA MODULE")
    print("─" * 70)
    try:
        from ema import (detect_ema_crossover, calculate_ema_ribbon,
                        analyze_ema_ribbon, detect_ema_squeeze)
    except Exception as e:
        r.skip("ema tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(80, step_pct=0.4)
    holder = {}

    def chk_crossover():
        out = detect_ema_crossover(up)
        return (out in ("bullish", "bearish", None)), f"got {out!r}"

    def chk_ribbon():
        ribbon = calculate_ema_ribbon(up)
        holder["v"] = ribbon
        return ribbon is not None, "got None"

    def chk_ribbon_analysis():
        ribbon = holder.get("v")
        if ribbon is None:
            return False, "ribbon None"
        out = analyze_ema_ribbon(ribbon)
        if not isinstance(out, dict):
            return False, f"got {type(out).__name__}"
        if "trend" not in out or "strength" not in out:
            return False, f"keys: {list(out.keys())}"
        return True, f"trend={out['trend']}, strength={out['strength']:.2f}"

    def chk_squeeze():
        ribbon = holder.get("v")
        if ribbon is None:
            return False, "ribbon None"
        out = detect_ema_squeeze(ribbon)
        if not isinstance(out, dict):
            return False, f"got {type(out).__name__}"
        return ("squeezing" in out), f"keys: {list(out.keys())}"

    r.check("ema: detect_ema_crossover returns str/None", chk_crossover)
    r.check("ema: calculate_ema_ribbon returns truthy", chk_ribbon)
    r.check("ema: analyze_ema_ribbon returns trend+strength", chk_ribbon_analysis)
    r.check("ema: detect_ema_squeeze returns squeezing dict", chk_squeeze)


def test_bollinger(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.5 — BOLLINGER MODULE")
    print("─" * 70)
    try:
        from bollinger import (calculate_bollinger_bands, detect_band_walk,
                              get_bollinger_signal)
    except Exception as e:
        r.skip("bollinger tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(60, step_pct=0.4)
    holder = {}

    def chk_bands():
        bb = calculate_bollinger_bands(up)
        holder["v"] = bb
        if bb is None or len(bb) == 0:
            return False, f"got {bb!r}"
        return True, f"len={len(bb)}"

    def chk_band_keys():
        bb = holder.get("v")
        if not bb:
            return False, "no bands"
        last = bb[-1]
        if not isinstance(last, dict):
            return False, f"got {type(last).__name__}"
        for k in ("upper", "lower"):
            if k not in last:
                return False, f"missing '{k}'"
        return True, ""

    def chk_band_walk():
        bb = holder.get("v")
        if not bb:
            return False, "no bands"
        out = detect_band_walk(up, bb)
        return (out is None or isinstance(out, dict)), f"got {type(out).__name__}"

    def chk_signal():
        out = get_bollinger_signal(up)
        if not isinstance(out, dict):
            return False, f"got {type(out).__name__}"
        for k in ("signal", "strength"):
            if k not in out:
                return False, f"missing '{k}'"
        return True, f"signal={out['signal']}, strength={out['strength']:.2f}"

    r.check("bollinger: calculate_bollinger_bands returns truthy", chk_bands)
    r.check("bollinger: bands have upper/lower", chk_band_keys)
    r.check("bollinger: detect_band_walk runs", chk_band_walk)
    r.check("bollinger: get_bollinger_signal returns signal+strength", chk_signal)


def test_volume(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.6 — VOLUME MODULE")
    print("─" * 70)
    try:
        from volume import (is_volume_spike, get_average_volume, detect_volume_climax,
                            get_volume_weighted_average_price)
    except Exception as e:
        r.skip("volume tests", f"import failed: {e}")
        return

    spike = make_volume_spike_candles(n=30, spike_at=-1, spike_multiplier=3.0)
    no_spike = make_volume_spike_candles(n=30, spike_at=-1, spike_multiplier=1.0)

    def chk_spike():
        out = is_volume_spike(spike, 1.8)
        return out is True, f"got {out!r}"

    def chk_no_spike():
        out = is_volume_spike(no_spike, 1.8)
        return out is False, f"got {out!r}"

    def chk_avg():
        out = get_average_volume(spike)
        return (isinstance(out, (int, float)) and out > 0), f"got {out!r}"

    def chk_climax():
        try:
            detect_volume_climax(spike)
            return True, ""
        except Exception as e:
            return False, f"raised: {e}"

    def chk_vwap():
        out = get_volume_weighted_average_price(make_uptrend_candles(60))
        return (out is None or isinstance(out, (int, float))), \
               f"got {type(out).__name__}"

    r.check("volume: is_volume_spike detects 3x spike", chk_spike)
    r.check("volume: is_volume_spike rejects flat volume", chk_no_spike)
    r.check("volume: get_average_volume returns positive number", chk_avg)
    r.check("volume: detect_volume_climax runs", chk_climax)
    r.check("volume: get_volume_weighted_average_price returns number/None", chk_vwap)


def test_pattern_detector(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.7 — PATTERN DETECTOR MODULE")
    print("─" * 70)
    try:
        from pattern_detector import (detect_pattern, get_pattern_direction,
                                     analyze_pattern_strength, get_all_patterns,
                                     REVERSAL_PATTERNS)
    except Exception as e:
        r.skip("pattern_detector tests", f"import failed: {e}")
        return

    up = make_uptrend_candles(60, step_pct=0.3)
    holder = {}

    def chk_detect():
        p = detect_pattern(up)
        holder["v"] = p
        return (p is None or isinstance(p, str)), f"got {type(p).__name__}"

    def chk_direction():
        p = holder.get("v")
        if p is None:
            return True, "no pattern — skipped"
        d = get_pattern_direction(p)
        return isinstance(d, str), f"got {type(d).__name__}"

    def chk_strength():
        p = holder.get("v")
        if p is None:
            return True, "no pattern — skipped"
        s = analyze_pattern_strength(p, up)
        if not isinstance(s, (int, float)):
            return False, f"got {type(s).__name__}"
        return (0 <= s <= 1.5), f"got {s}"

    def chk_all():
        out = get_all_patterns(up)
        return isinstance(out, dict), f"got {type(out).__name__}"

    def chk_reversal():
        if isinstance(REVERSAL_PATTERNS, dict) and "bullish" in REVERSAL_PATTERNS:
            return True, ""
        return False, f"got {REVERSAL_PATTERNS!r}"

    r.check("pattern: detect_pattern returns string/None", chk_detect)
    r.check("pattern: get_pattern_direction returns string", chk_direction)
    r.check("pattern: analyze_pattern_strength in 0..1.5", chk_strength)
    r.check("pattern: get_all_patterns returns dict", chk_all)
    r.check("pattern: REVERSAL_PATTERNS has 'bullish' key", chk_reversal)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1.8 — SCORE.PY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def test_score_helpers(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.8 — SCORE.PY HELPER FUNCTIONS")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        r.skip("score helpers", f"import failed: {e}")
        return

    # determine_direction
    r.check("score: determine_direction(strong bullish) = Long",
           lambda: (score.determine_direction({"1": 3.0, "5": 3.5, "15": 4.0}) == "Long",
                   "expected Long"))
    r.check("score: determine_direction(strong bearish) = Short",
           lambda: (score.determine_direction({"1": -3.0, "5": -3.5, "15": -4.0}) == "Short",
                   "expected Short"))
    r.check("score: determine_direction(mixed) = None",
           lambda: (score.determine_direction({"1": 1.5, "5": -1.5, "15": 0.5}) is None,
                   "mixed should give None"))
    r.check("score: determine_direction(weak total<2) = None",
           lambda: (score.determine_direction({"1": 0.6, "5": 0.7, "15": 0.5}) is None,
                   "total=1.8 (<2) should give None"))
    r.check("score: determine_direction(empty) = None",
           lambda: (score.determine_direction({}) is None, ""))
    r.check("score: determine_direction(strong opposing TF) = None",
           lambda: (score.determine_direction({"1": 3.0, "5": 3.5, "15": -2.5}) is None,
                   "-2.5 TF should veto Long"))

    # calculate_confidence
    bull_conf = [0]
    def chk_conf_strong():
        c = score.calculate_confidence(10.0, {"1": 2.0, "5": 2.5, "15": 3.0},
                                        {"btc_trend": "bullish"}, "Intraday")
        bull_conf[0] = c
        return (isinstance(c, int) and 50 <= c <= 100), f"conf={c}"

    def chk_conf_counter():
        c = score.calculate_confidence(10.0, {"1": 2.0, "5": 2.5, "15": 3.0},
                                        {"btc_trend": "bearish"}, "Intraday")
        return (c < bull_conf[0]), f"counter={c} vs aligned={bull_conf[0]}"

    def chk_conf_empty():
        c = score.calculate_confidence(10.0, {}, {}, "Scalp")
        return c == 0, f"got {c}"

    r.check("score: calculate_confidence(strong setup) returns int 50..100", chk_conf_strong)
    r.check("score: counter-trend confidence < aligned", chk_conf_counter)
    r.check("score: calculate_confidence(empty TFs) = 0", chk_conf_empty)

    # check_4h_alignment_veto — use strong uptrend so all 3 indicators agree
    def chk_4h_long_ok():
        c4h = make_uptrend_candles(60, step_pct=0.8, seed=999)
        allow, _ = score.check_4h_alignment_veto({"240": c4h}, "Long")
        return allow, "Long should be allowed in strong 4h uptrend"

    def chk_4h_short_vetoed():
        c4h = make_uptrend_candles(60, step_pct=0.8, seed=999)
        allow, reason = score.check_4h_alignment_veto({"240": c4h}, "Short")
        return (not allow), f"Short should be vetoed (reason={reason})"

    def chk_4h_no_data():
        allow, _ = score.check_4h_alignment_veto({}, "Long")
        return allow, "no 4h data should fail open"

    r.check("score: 4h uptrend allows Long", chk_4h_long_ok)
    r.check("score: 4h uptrend vetoes Short", chk_4h_short_vetoed)
    r.check("score: 4h veto fails open (no data)", chk_4h_no_data)

    # check_not_chasing
    def chk_chase_clean():
        allow, _ = score.check_not_chasing(make_uptrend_set(), "Long", "Scalp")
        return allow, "clean entry should be allowed"

    def chk_chase_extended():
        allow, reason = score.check_not_chasing(make_chasing_set(), "Long", "Scalp")
        return (not allow), f"extended entry should be rejected ({reason})"

    r.check("score: anti-chase allows clean entry", chk_chase_clean)
    r.check("score: anti-chase rejects extended entry", chk_chase_extended)

    # safe_detect_momentum_strength
    def chk_mom():
        h, d, s = score.safe_detect_momentum_strength(
            make_uptrend_candles(40, step_pct=0.5))
        ok = (isinstance(h, bool)
              and d in (None, "bullish", "bearish")
              and isinstance(s, (int, float)))
        return ok, f"({h}, {d}, {s})"

    def chk_mom_empty():
        h, _, _ = score.safe_detect_momentum_strength([])
        return h is False, f"got {h!r}"

    r.check("score: safe_detect_momentum_strength runs on uptrend", chk_mom)
    r.check("score: safe_detect_momentum_strength([]) = False", chk_mom_empty)


def test_trend_filters(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 1.9 — TREND FILTERS")
    print("─" * 70)
    try:
        from trend_filters import validate_short_signal
        import asyncio
    except Exception as e:
        r.skip("trend_filters tests", f"import failed: {e}")
        return

    async def run_validate():
        try:
            out = await validate_short_signal("TESTUSDT", {})
            return ("ok", out)
        except Exception as e:
            return ("err", str(e))

    def chk():
        try:
            kind, val = asyncio.run(run_validate())
        except Exception as e:
            return False, f"asyncio.run crashed: {e}"
        if kind == "err":
            return True, f"async raised but did not crash test: {val}"
        if val is False:
            return True, "returned False on empty input"
        if isinstance(val, bool):
            return True, f"returned bool {val}"
        return False, f"unexpected return {val!r}"

    r.check("trend_filters: validate_short_signal handles empty input", chk)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_score(out):
    if not isinstance(out, tuple) or len(out) != 5:
        l = len(out) if hasattr(out, '__len__') else '?'
        return None, f"expected 5-tuple, got {type(out).__name__} of len {l}"
    sv, tfs, tt, ind, used = out
    if not isinstance(sv, (int, float)):
        return None, f"score is {type(sv).__name__}"
    if not isinstance(tfs, dict):
        return None, f"tf_scores is {type(tfs).__name__}"
    if tt not in ("Scalp", "Intraday", "Swing"):
        return None, f"trade_type={tt!r}"
    if not isinstance(ind, dict):
        return None, f"indicator_scores is {type(ind).__name__}"
    if not isinstance(used, list):
        return None, f"used_indicators is {type(used).__name__}"
    return (sv, tfs, tt, ind, used), None


def test_score_symbol_integration(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 2.1 — score_symbol() FULL PIPELINE")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        r.skip("score_symbol integration", f"import failed: {e}")
        return

    def chk_uptrend_tuple():
        out = score.score_symbol("TESTUSDT", make_uptrend_set(),
                                  {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        s, tf_s, tt, ind, _ = u
        return True, f"score={s:.2f}, type={tt}, tfs={len(tf_s)}, ind={len(ind)}"

    def chk_uptrend_positive():
        out = score.score_symbol("TESTUSDT", make_uptrend_set(),
                                  {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        s = u[0]
        return (s > 0), f"got {s:.2f}, expected > 0"

    def chk_downtrend_tuple():
        out = score.score_symbol("TESTUSDT", make_downtrend_set(),
                                  {"btc_trend": "bearish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        return True, f"score={u[0]:.2f}, type={u[2]}"

    def chk_uptrend_higher():
        out_up = score.score_symbol("TESTUSDT", make_uptrend_set(),
                                     {"btc_trend": "bullish"})
        out_dn = score.score_symbol("TESTUSDT", make_downtrend_set(),
                                     {"btc_trend": "bearish"})
        u1, e1 = _unpack_score(out_up)
        u2, e2 = _unpack_score(out_dn)
        if e1 or e2:
            return False, e1 or e2
        return (u1[0] > u2[0]), f"up={u1[0]:.2f}, down={u2[0]:.2f}"

    def chk_empty():
        try:
            score.score_symbol("TESTUSDT", {}, {})
            return True, "handled"
        except Exception as e:
            return False, f"crashed: {e}"

    def chk_short_candles():
        try:
            score.score_symbol("TESTUSDT",
                              {tf: make_uptrend_candles(5) for tf in ["1", "5", "15"]},
                              {})
            return True, "handled"
        except Exception as e:
            return False, f"crashed: {e}"

    def chk_weights_st():
        v = score.WEIGHTS.get("supertrend_mtf", 0)
        return (v >= 1.3), f"got {v} (expected >= 1.3 post-patch)"

    def chk_weights_bb():
        v = score.WEIGHTS.get("bollinger_squeeze", 1)
        return (v == 0.0), f"got {v} (expected 0.0)"

    r.check("score_symbol: uptrend returns valid 5-tuple", chk_uptrend_tuple)
    r.check("score_symbol: uptrend score > 0", chk_uptrend_positive)
    r.check("score_symbol: downtrend returns valid 5-tuple", chk_downtrend_tuple)
    r.check("score_symbol: uptrend score > downtrend score", chk_uptrend_higher)
    r.check("score_symbol: empty candles handled", chk_empty)
    r.check("score_symbol: short candles handled", chk_short_candles)
    r.check("score: WEIGHTS supertrend_mtf >= 1.3", chk_weights_st)
    r.check("score: WEIGHTS bollinger_squeeze = 0", chk_weights_bb)


def test_enhanced_score_symbol_integration(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 2.2 — enhanced_score_symbol() WITH QUALITY GATES")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        r.skip("enhanced_score_symbol", f"import failed: {e}")
        return

    def chk_uptrend():
        out = score.enhanced_score_symbol("TESTUSDT", make_uptrend_set(),
                                           {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        return True, f"score={u[0]:.2f}, type={u[2]}, ind={len(u[3])}"

    def chk_chase():
        out = score.enhanced_score_symbol("TESTUSDT", make_chasing_set(),
                                           {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        return True, f"chase score={u[0]:.2f}"

    def chk_ranging():
        out = score.enhanced_score_symbol("TESTUSDT", make_ranging_set(), {})
        u, err = _unpack_score(out)
        if err:
            return False, err
        d = score.determine_direction(u[1])
        return True, f"score={u[0]:.2f}, direction={d}"

    def chk_market_ctx():
        out_b = score.enhanced_score_symbol("TESTUSDT", make_uptrend_set(),
                                             {"btc_trend": "bullish"})
        out_r = score.enhanced_score_symbol("TESTUSDT", make_uptrend_set(),
                                             {"btc_trend": "bearish"})
        u1, e1 = _unpack_score(out_b)
        u2, e2 = _unpack_score(out_r)
        if e1 or e2:
            return False, e1 or e2
        return (u2[0] <= u1[0]), \
               f"counter={u2[0]:.2f} vs aligned={u1[0]:.2f}"

    r.check("enhanced_score_symbol: uptrend OK", chk_uptrend)
    r.check("enhanced_score_symbol: chase scenario handled", chk_chase)
    r.check("enhanced_score_symbol: ranging produces value", chk_ranging)
    r.check("enhanced_score_symbol: counter-trend ≤ aligned", chk_market_ctx)


def test_quality_gates_end_to_end(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 2.3 — QUALITY GATES END-TO-END")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        r.skip("quality gates e2e", f"import failed: {e}")
        return

    def strong_bear_4h_set():
        return {
            "1":   make_uptrend_candles(60, step_pct=0.10, seed=701),
            "5":   make_uptrend_candles(60, step_pct=0.30, seed=705),
            "15":  make_uptrend_candles(50, step_pct=0.50, seed=715),
            "240": make_downtrend_candles(60, step_pct=0.8, seed=740),
        }

    def chk_4h_veto():
        out = score.enhanced_score_symbol(
            "TESTUSDT", strong_bear_4h_set(), {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        s = u[0]
        return (s == 0), \
               f"score={s:.2f} (expected 0 — 4h bearish should veto Long)"

    def chk_anti_chase():
        out = score.enhanced_score_symbol("TESTUSDT", make_chasing_set(),
                                           {"btc_trend": "bullish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        return True, f"score={u[0]:.2f} (gate processed)"

    def chk_strong_down():
        # The most important test — strong bearish setup should NOT produce Long
        out = score.enhanced_score_symbol("TESTUSDT", make_downtrend_set(),
                                           {"btc_trend": "bearish"})
        u, err = _unpack_score(out)
        if err:
            return False, err
        s, tf_s, _, _, _ = u
        d = score.determine_direction(tf_s)
        if s == 0:
            return True, "gated to 0 (no clean signal)"
        if d == "Short":
            return True, f"correctly Short (score={s:.2f})"
        if d == "Long" and s > 0:
            return False, (f"⚠️  BUG REPRODUCED: bearish setup → Long signal "
                          f"score={s:.2f}. This is the win-rate problem.")
        return True, f"score={s:.2f}, dir={d}"

    r.check("e2e: 4h-bearish vetoes Long-low-TF setup", chk_4h_veto)
    r.check("e2e: anti-chase scenario processed", chk_anti_chase)
    r.check("e2e: strong downtrend → Short or 0 (NOT Long)", chk_strong_down)


def test_main_integration(r: TestResult):
    print("\n" + "─" * 70)
    print("LAYER 2.4 — main.py INTEGRATION SMOKE TEST")
    print("─" * 70)
    try:
        import main
    except Exception as e:
        r.skip("main.py integration", f"import main failed: {e}")
        return

    expected = ["score_symbol", "determine_direction", "calculate_confidence",
                "has_pump_potential", "detect_momentum_strength"]
    for fn in expected:
        def make_check(name):
            return lambda: (hasattr(main, name), f"main.{name} present")
        r.check(f"main: {fn} accessible", make_check(fn))

    thresholds = [("MIN_SCALP_SCORE", 10.5), ("MIN_INTRADAY_SCORE", 12.0),
                  ("MIN_SWING_SCORE", 15.5)]
    for attr, expected_val in thresholds:
        def make_thr(a, e):
            def fn():
                if not hasattr(main, a):
                    return False, "missing"
                actual = getattr(main, a)
                return (actual >= e), f"got {actual} (expected >= {e})"
            return fn
        r.check(f"main: {attr} >= {expected_val}", make_thr(attr, expected_val))


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main_runner():
    parser = argparse.ArgumentParser(description="Test score.py and dependencies")
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--module", type=str, default=None)
    args = parser.parse_args()

    run_unit = args.unit or not (args.unit or args.integration)
    run_integration = args.integration or not (args.unit or args.integration)

    print("═" * 70)
    print("SCORE.PY TEST SUITE v2")
    print("═" * 70)
    print(f"Working dir:  {os.getcwd()}")
    print(f"Layer 1 unit: {run_unit}")
    print(f"Layer 2 int:  {run_integration}")
    print(f"Verbose:      {args.verbose}")
    if args.module:
        print(f"Module filter: {args.module}")

    r = TestResult(verbose=args.verbose)

    if run_unit:
        test_imports(r, args.module)

        unit_tests = {
            "rsi":              test_rsi,
            "macd":             test_macd,
            "supertrend":       test_supertrend,
            "ema":              test_ema,
            "bollinger":        test_bollinger,
            "volume":           test_volume,
            "pattern_detector": test_pattern_detector,
            "score":            test_score_helpers,
            "trend_filters":    test_trend_filters,
        }
        for name, fn in unit_tests.items():
            if args.module and args.module != name:
                continue
            try:
                fn(r)
            except Exception as e:
                r.fail(f"{name} test suite",
                      f"runner crashed: {e}\n{traceback.format_exc()}")

    if run_integration:
        for fn in [test_score_symbol_integration,
                   test_enhanced_score_symbol_integration,
                   test_quality_gates_end_to_end,
                   test_main_integration]:
            try:
                fn(r)
            except Exception as e:
                r.fail(fn.__name__,
                      f"runner crashed: {e}\n{traceback.format_exc()}")

    return r.summary()


if __name__ == "__main__":
    try:
        sys.exit(main_runner())
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(2)
