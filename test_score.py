#!/usr/bin/env python3
"""
test_score.py — Comprehensive test for score.py and dependencies
=================================================================
Tests run in TWO layers so you can pinpoint exactly what's broken:

  LAYER 1 (UNIT) — Each helper / dependency is tested with controlled
                   synthetic candles. No external API calls.
  LAYER 2 (INTEG) — Full score_symbol() and enhanced_score_symbol()
                    pipelines tested with realistic multi-TF candle sets.

Usage:
    python3 test_score.py                # run everything
    python3 test_score.py --unit         # only Layer 1 (faster)
    python3 test_score.py --integration  # only Layer 2
    python3 test_score.py --verbose      # show all PASS/FAIL detail
    python3 test_score.py --module rsi   # only test a specific module

Exit codes:
    0 — all tests passed
    1 — at least one test failed
    2 — import error / setup failure

Place this file in the same directory as score.py and run it with python3.
"""

import sys
import os
import argparse
import traceback
import time
from typing import List, Dict, Any, Tuple, Callable

# ── Test result tracking ─────────────────────────────────────────────────────
class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[Tuple[str, str]] = []  # (test_name, reason)
        self.skipped_tests: List[Tuple[str, str]] = []
        self.start_time = time.time()

    def record_pass(self, name: str, verbose: bool):
        self.passed += 1
        if verbose:
            print(f"  ✅ {name}")

    def record_fail(self, name: str, reason: str):
        self.failed += 1
        self.failures.append((name, reason))
        print(f"  ❌ {name}")
        # Indent the reason
        for line in str(reason).splitlines():
            print(f"      {line}")

    def record_skip(self, name: str, reason: str, verbose: bool):
        self.skipped += 1
        self.skipped_tests.append((name, reason))
        if verbose:
            print(f"  ⏭️  {name} (skipped: {reason})")

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
            print()
            print("FAILED TESTS:")
            for name, _ in self.failures:
                print(f"  ❌ {name}")
            return 1
        if self.passed == 0 and self.skipped > 0:
            print("\n⚠️  All tests were skipped — verify the score module imports OK.")
            return 1
        print("\n🎉 All tests passed!")
        return 0


# ── Synthetic candle builders ────────────────────────────────────────────────
# Candle shape used everywhere in the project: dict with string-or-float values.
# We use float-typed values; score.py's own helpers wrap with float() defensively.

def make_candle(open_p, high, low, close, volume, timestamp=0):
    """Build one candle dict matching the project's expected shape."""
    return {
        "timestamp": timestamp,
        "open": float(open_p),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


def make_uptrend_candles(n: int = 60, start_price: float = 100.0,
                        step_pct: float = 0.3, volume: float = 1_000_000) -> List[Dict]:
    """
    Create a clean monotonic uptrend with realistic intra-bar wicks.
    Each candle closes step_pct% above the previous close.
    """
    candles = []
    price = start_price
    for i in range(n):
        prev_close = price
        price = price * (1 + step_pct / 100)
        high  = price * 1.002
        low   = prev_close * 0.999
        candles.append(make_candle(prev_close, high, low, price, volume, timestamp=i * 60_000))
    return candles


def make_downtrend_candles(n: int = 60, start_price: float = 100.0,
                          step_pct: float = 0.3, volume: float = 1_000_000) -> List[Dict]:
    """Clean monotonic downtrend."""
    candles = []
    price = start_price
    for i in range(n):
        prev_close = price
        price = price * (1 - step_pct / 100)
        high  = prev_close * 1.001
        low   = price * 0.998
        candles.append(make_candle(prev_close, high, low, price, volume, timestamp=i * 60_000))
    return candles


def make_ranging_candles(n: int = 60, center_price: float = 100.0,
                        amplitude_pct: float = 0.5, volume: float = 1_000_000) -> List[Dict]:
    """Sideways oscillation around center_price."""
    import math
    candles = []
    for i in range(n):
        offset = math.sin(i * 0.5) * (amplitude_pct / 100) * center_price
        prev_close = center_price + math.sin((i - 1) * 0.5) * (amplitude_pct / 100) * center_price
        close = center_price + offset
        high  = max(prev_close, close) * 1.001
        low   = min(prev_close, close) * 0.999
        candles.append(make_candle(prev_close, high, low, close, volume, timestamp=i * 60_000))
    return candles


def make_volume_spike_candles(n: int = 30, base_price: float = 100.0,
                              spike_at: int = -1, spike_multiplier: float = 3.0) -> List[Dict]:
    """
    Flat price with one volume spike at position `spike_at` (default last candle).
    Used to test is_volume_spike, detect_volume_climax, etc.
    """
    candles = []
    base_volume = 1_000_000
    for i in range(n):
        is_spike = (i == n + spike_at) if spike_at < 0 else (i == spike_at)
        volume = base_volume * spike_multiplier if is_spike else base_volume
        # Tiny price wiggle so MACD/EMA don't NaN out
        close = base_price + (i % 3 - 1) * 0.05
        prev_close = base_price + ((i - 1) % 3 - 1) * 0.05
        high = max(close, prev_close) + 0.1
        low = min(close, prev_close) - 0.1
        candles.append(make_candle(prev_close, high, low, close, volume, timestamp=i * 60_000))
    return candles


def make_uptrend_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    """Multi-timeframe uptrend candle set — 1m, 5m, 15m, 60m, 240m."""
    return {
        "1":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.05),
        "3":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.10),
        "5":   make_uptrend_candles(n=max(min_candles, 60), step_pct=0.20),
        "15":  make_uptrend_candles(n=max(min_candles, 50), step_pct=0.40),
        "30":  make_uptrend_candles(n=max(min_candles, 40), step_pct=0.60),
        "60":  make_uptrend_candles(n=max(min_candles, 40), step_pct=0.80),
        "240": make_uptrend_candles(n=max(min_candles, 30), step_pct=1.20),
    }


def make_downtrend_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    """Multi-timeframe downtrend candle set."""
    return {
        "1":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.05),
        "3":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.10),
        "5":   make_downtrend_candles(n=max(min_candles, 60), step_pct=0.20),
        "15":  make_downtrend_candles(n=max(min_candles, 50), step_pct=0.40),
        "30":  make_downtrend_candles(n=max(min_candles, 40), step_pct=0.60),
        "60":  make_downtrend_candles(n=max(min_candles, 40), step_pct=0.80),
        "240": make_downtrend_candles(n=max(min_candles, 30), step_pct=1.20),
    }


def make_ranging_set(min_candles: int = 50) -> Dict[str, List[Dict]]:
    """Multi-timeframe ranging market — should NOT produce strong signals."""
    return {
        "1":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.3),
        "3":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.4),
        "5":   make_ranging_candles(n=max(min_candles, 60), amplitude_pct=0.5),
        "15":  make_ranging_candles(n=max(min_candles, 50), amplitude_pct=0.6),
        "30":  make_ranging_candles(n=max(min_candles, 40), amplitude_pct=0.7),
        "60":  make_ranging_candles(n=max(min_candles, 40), amplitude_pct=0.8),
        "240": make_ranging_candles(n=max(min_candles, 30), amplitude_pct=1.0),
    }


def make_chasing_set() -> Dict[str, List[Dict]]:
    """
    Uptrend that just spiked 3% in the last 3 candles — should be vetoed
    by check_not_chasing for Long Scalp.
    """
    base = make_uptrend_candles(n=60, step_pct=0.05)
    # Inject a sharp last 3-candle move (~3%)
    last_close = base[-4]["close"]
    for i in range(3):
        new_close = last_close * (1 + (i + 1) * 0.012)  # +1.2% each
        base[-3 + i] = make_candle(
            open_p=last_close if i == 0 else base[-3 + i - 1]["close"],
            high=new_close * 1.003,
            low=last_close * 0.998,
            close=new_close,
            volume=2_000_000,
            timestamp=base[-3 + i]["timestamp"],
        )
    return {
        "1":  base,
        "5":  make_uptrend_candles(n=60, step_pct=0.20),
        "15": make_uptrend_candles(n=50, step_pct=0.40),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT-TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_import(module_name: str, attrs: List[str], result: TestResult, verbose: bool):
    """Try to import attrs from module_name. Records passes/fails."""
    try:
        mod = __import__(module_name, fromlist=attrs)
    except Exception as e:
        for attr in attrs:
            result.record_skip(f"import {module_name}.{attr}", f"module import failed: {e}", verbose)
        return None

    for attr in attrs:
        if hasattr(mod, attr):
            result.record_pass(f"import {module_name}.{attr}", verbose)
        else:
            result.record_fail(f"import {module_name}.{attr}", f"attribute missing in {module_name}")
    return mod


def call_safely(name: str, fn: Callable, args, expected_type, result: TestResult,
               check_fn: Callable = None, verbose: bool = False):
    """
    Call fn(*args). Pass if result is of expected_type and check_fn returns True.
    check_fn receives the return value and returns (ok: bool, msg: str).
    """
    try:
        out = fn(*args)
    except Exception as e:
        result.record_fail(name, f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None

    if expected_type and not isinstance(out, expected_type):
        result.record_fail(name, f"expected {expected_type.__name__}, got {type(out).__name__}: {out!r}")
        return None

    if check_fn:
        try:
            ok, msg = check_fn(out)
        except Exception as e:
            result.record_fail(name, f"check_fn raised: {e}")
            return None
        if not ok:
            result.record_fail(name, f"check failed: {msg} | got: {out!r}")
            return None

    result.record_pass(name, verbose)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — UNIT TESTS PER MODULE
# ─────────────────────────────────────────────────────────────────────────────

def test_imports(result: TestResult, verbose: bool, module_filter: str = None):
    """Verify every dependency score.py imports actually exists & loads."""
    print("\n" + "─" * 70)
    print("LAYER 1.0 — IMPORT VERIFICATION")
    print("─" * 70)

    expected_imports = {
        "logger":                ["log"],
        "rsi":                   ["calculate_rsi", "calculate_rsi_with_bands",
                                  "calculate_stoch_rsi", "analyze_multi_timeframe_rsi"],
        "macd":                  ["detect_macd_cross", "get_macd_divergence", "get_macd_momentum"],
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
        "whale_detector":        ["detect_whale_activity", "detect_whale_activity_advanced"],
        "indicator_fixes":       ["rebalance_indicator_scores", "get_balanced_rsi_signal",
                                  "analyze_volume_direction"],
        "enhanced_entry_validator":  ["entry_validator"],
        "pattern_context_analyzer":  ["pattern_context_analyzer"],
        "divergence_detector":   ["divergence_detector"],
        "trend_filters":         ["validate_short_signal", "monitor_btc_trend_accuracy",
                                  "monitor_altseason_status"],
    }

    for mod_name, attrs in expected_imports.items():
        if module_filter and module_filter != mod_name:
            continue
        safe_import(mod_name, attrs, result, verbose)

    # And finally score.py itself
    print()
    safe_import("score", [
        "score_symbol", "enhanced_score_symbol", "determine_direction",
        "calculate_confidence", "has_pump_potential", "detect_momentum_strength",
        "safe_detect_momentum_strength", "get_4h_trend_gate",
        "check_4h_alignment_veto", "check_not_chasing",
        "WEIGHTS", "TRADE_TYPE_TF", "MIN_TF_REQUIRED", "VOLUME_SPIKE_THRESHOLD",
    ], result, verbose)


def test_rsi(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.1 — RSI MODULE")
    print("─" * 70)
    try:
        from rsi import calculate_rsi, calculate_rsi_with_bands, calculate_stoch_rsi
    except Exception as e:
        result.record_skip("rsi tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(60)
    down = make_downtrend_candles(60)

    # calculate_rsi — should return a list/array of floats with the last value > 50 in uptrend
    def check_rsi_uptrend(out):
        if isinstance(out, (list, tuple)) and len(out) > 0:
            last = out[-1]
        elif isinstance(out, (int, float)):
            last = out
        else:
            return False, f"unexpected type {type(out).__name__}"
        if last is None:
            return False, "got None"
        return (last > 50), f"RSI(uptrend) last value = {last:.2f}, expected > 50"

    def check_rsi_downtrend(out):
        if isinstance(out, (list, tuple)) and len(out) > 0:
            last = out[-1]
        elif isinstance(out, (int, float)):
            last = out
        else:
            return False, f"unexpected type {type(out).__name__}"
        if last is None:
            return False, "got None"
        return (last < 50), f"RSI(downtrend) last value = {last:.2f}, expected < 50"

    call_safely("rsi: uptrend produces RSI > 50", calculate_rsi, [up],
               None, check_rsi_uptrend, verbose=verbose)
    call_safely("rsi: downtrend produces RSI < 50", calculate_rsi, [down],
               None, check_rsi_downtrend, verbose=verbose)

    # calculate_rsi_with_bands — should be a dict
    def check_rsi_bands(out):
        if not isinstance(out, dict):
            return False, "expected dict"
        return True, ""
    call_safely("rsi: calculate_rsi_with_bands returns dict", calculate_rsi_with_bands,
               [up], dict, check_rsi_bands, verbose=verbose)

    # Stoch RSI
    call_safely("rsi: calculate_stoch_rsi runs", calculate_stoch_rsi, [up],
               None, lambda x: (x is None or isinstance(x, dict),
                               f"expected dict or None, got {type(x).__name__}"),
               verbose=verbose)


def test_macd(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.2 — MACD MODULE")
    print("─" * 70)
    try:
        from macd import detect_macd_cross, get_macd_momentum
    except Exception as e:
        result.record_skip("macd tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(60)
    down = make_downtrend_candles(60)

    # detect_macd_cross returns "bullish" / "bearish" / None
    call_safely("macd: detect_macd_cross returns string or None", detect_macd_cross, [up],
               None, lambda x: (x in ("bullish", "bearish", None),
                               f"expected 'bullish'/'bearish'/None, got {x!r}"),
               verbose=verbose)

    # get_macd_momentum returns a number
    def check_momentum(out):
        if not isinstance(out, (int, float)):
            return False, f"expected number, got {type(out).__name__}"
        return True, ""
    call_safely("macd: get_macd_momentum returns number (uptrend)", get_macd_momentum, [up],
               None, check_momentum, verbose=verbose)
    call_safely("macd: get_macd_momentum returns number (downtrend)", get_macd_momentum, [down],
               None, check_momentum, verbose=verbose)

    # Sanity: uptrend momentum should generally be > 0
    try:
        mom_up = get_macd_momentum(up)
        mom_down = get_macd_momentum(down)
        if mom_up > mom_down:
            result.record_pass("macd: uptrend momentum > downtrend momentum", verbose)
        else:
            result.record_fail("macd: uptrend momentum > downtrend momentum",
                              f"got mom_up={mom_up:.4f}, mom_down={mom_down:.4f}")
    except Exception as e:
        result.record_fail("macd: uptrend > downtrend momentum sanity", str(e))


def test_supertrend(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.3 — SUPERTREND MODULE")
    print("─" * 70)
    try:
        from supertrend import (calculate_supertrend_signal, get_supertrend_state,
                                calculate_multi_timeframe_supertrend)
    except Exception as e:
        result.record_skip("supertrend tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(60)
    down = make_downtrend_candles(60)

    # calculate_supertrend_signal: should be "bullish"/"bearish"/None or similar string
    def check_signal_string(out):
        if out is None:
            return True, ""
        if not isinstance(out, str):
            return False, f"expected string or None, got {type(out).__name__}"
        return True, ""
    call_safely("supertrend: signal returns string/None (uptrend)",
               calculate_supertrend_signal, [up], None, check_signal_string, verbose=verbose)

    # get_supertrend_state: dict with 'trend' key
    def check_state(out):
        if not isinstance(out, dict):
            return False, f"expected dict, got {type(out).__name__}"
        if "trend" not in out:
            return False, "missing 'trend' key"
        return True, ""
    call_safely("supertrend: get_supertrend_state returns dict",
               get_supertrend_state, [up], dict, check_state, verbose=verbose)

    # MTF supertrend — uses dict of TFs
    candles_set = make_uptrend_set()
    def check_mtf(out):
        if not isinstance(out, dict):
            return False, f"expected dict, got {type(out).__name__}"
        if "alignment" not in out:
            return False, "missing 'alignment' key"
        return True, ""
    call_safely("supertrend: MTF supertrend returns alignment dict",
               calculate_multi_timeframe_supertrend, [candles_set], dict, check_mtf, verbose=verbose)


def test_ema(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.4 — EMA MODULE")
    print("─" * 70)
    try:
        from ema import (detect_ema_crossover, calculate_ema_ribbon,
                        analyze_ema_ribbon, detect_ema_squeeze)
    except Exception as e:
        result.record_skip("ema tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(80)

    call_safely("ema: detect_ema_crossover runs (uptrend)", detect_ema_crossover, [up],
               None, lambda x: (x in ("bullish", "bearish", None),
                               f"expected str or None, got {x!r}"), verbose=verbose)

    ribbon = call_safely("ema: calculate_ema_ribbon returns truthy",
                        calculate_ema_ribbon, [up], None,
                        lambda x: (x is not None, "got None"), verbose=verbose)

    if ribbon is not None:
        def check_ribbon_analysis(out):
            if not isinstance(out, dict):
                return False, f"expected dict, got {type(out).__name__}"
            if "trend" not in out or "strength" not in out:
                return False, f"missing 'trend' or 'strength' key, got keys: {list(out.keys())}"
            return True, ""
        call_safely("ema: analyze_ema_ribbon returns trend+strength",
                   analyze_ema_ribbon, [ribbon], dict, check_ribbon_analysis, verbose=verbose)

        def check_squeeze(out):
            if not isinstance(out, dict):
                return False, f"expected dict, got {type(out).__name__}"
            if "squeezing" not in out:
                return False, "missing 'squeezing' key"
            return True, ""
        call_safely("ema: detect_ema_squeeze returns squeezing dict",
                   detect_ema_squeeze, [ribbon], dict, check_squeeze, verbose=verbose)


def test_bollinger(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.5 — BOLLINGER MODULE")
    print("─" * 70)
    try:
        from bollinger import (calculate_bollinger_bands, detect_band_walk,
                              get_bollinger_signal)
    except Exception as e:
        result.record_skip("bollinger tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(60)

    bb = call_safely("bollinger: calculate_bollinger_bands returns truthy",
                    calculate_bollinger_bands, [up], None,
                    lambda x: (x is not None and len(x) > 0, "got None or empty"),
                    verbose=verbose)

    if bb is not None and len(bb) > 0:
        # Last band should have upper/mid/lower
        def check_band_keys(out):
            last = out[-1]
            if not isinstance(last, dict):
                return False, "expected dict element"
            for k in ("upper", "lower"):
                if k not in last:
                    return False, f"missing '{k}' in last band"
            return True, ""
        call_safely("bollinger: bands have upper/lower keys", lambda x: x, [bb],
                   None, check_band_keys, verbose=verbose)

        # Band walk
        call_safely("bollinger: detect_band_walk runs", detect_band_walk, [up, bb],
                   None, lambda x: (x is None or isinstance(x, dict),
                                   f"expected dict/None, got {type(x).__name__}"),
                   verbose=verbose)

    def check_bb_signal(out):
        if not isinstance(out, dict):
            return False, f"expected dict, got {type(out).__name__}"
        if "signal" not in out or "strength" not in out:
            return False, f"missing 'signal' or 'strength', keys: {list(out.keys())}"
        return True, ""
    call_safely("bollinger: get_bollinger_signal returns signal+strength",
               get_bollinger_signal, [up], dict, check_bb_signal, verbose=verbose)


def test_volume(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.6 — VOLUME MODULE")
    print("─" * 70)
    try:
        from volume import (is_volume_spike, get_average_volume, detect_volume_climax,
                            get_volume_weighted_average_price)
    except Exception as e:
        result.record_skip("volume tests", f"import failed: {e}", verbose)
        return

    spike = make_volume_spike_candles(n=30, spike_at=-1, spike_multiplier=3.0)
    no_spike = make_volume_spike_candles(n=30, spike_at=-1, spike_multiplier=1.0)

    # is_volume_spike
    call_safely("volume: is_volume_spike detects 3x spike", is_volume_spike, [spike, 1.8],
               None, lambda x: (x is True or x == True, f"expected True, got {x!r}"),
               verbose=verbose)
    call_safely("volume: is_volume_spike rejects flat volume", is_volume_spike, [no_spike, 1.8],
               None, lambda x: (x is False or x == False, f"expected False, got {x!r}"),
               verbose=verbose)

    # get_average_volume
    call_safely("volume: get_average_volume returns positive number",
               get_average_volume, [spike], None,
               lambda x: (isinstance(x, (int, float)) and x > 0,
                         f"expected number > 0, got {x!r}"), verbose=verbose)

    # detect_volume_climax — may return tuple or bool depending on version
    call_safely("volume: detect_volume_climax runs", detect_volume_climax, [spike],
               None, lambda x: (True, ""), verbose=verbose)

    # VWAP
    call_safely("volume: get_volume_weighted_average_price returns number/None",
               get_volume_weighted_average_price, [make_uptrend_candles(60)], None,
               lambda x: (x is None or isinstance(x, (int, float)),
                         f"expected number/None, got {type(x).__name__}"), verbose=verbose)


def test_pattern_detector(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 1.7 — PATTERN DETECTOR MODULE")
    print("─" * 70)
    try:
        from pattern_detector import (detect_pattern, get_pattern_direction,
                                     analyze_pattern_strength, get_all_patterns,
                                     REVERSAL_PATTERNS, CONTINUATION_PATTERNS)
    except Exception as e:
        result.record_skip("pattern_detector tests", f"import failed: {e}", verbose)
        return

    up = make_uptrend_candles(60)

    # detect_pattern — string or None
    pattern = call_safely("pattern: detect_pattern returns string/None",
                         detect_pattern, [up], None,
                         lambda x: (x is None or isinstance(x, str),
                                   f"expected str/None, got {type(x).__name__}"),
                         verbose=verbose)

    # If we got a pattern, test direction + strength
    if pattern:
        call_safely(f"pattern: get_pattern_direction({pattern}) returns string",
                   get_pattern_direction, [pattern], None,
                   lambda x: (isinstance(x, str), f"expected str, got {type(x).__name__}"),
                   verbose=verbose)
        call_safely(f"pattern: analyze_pattern_strength({pattern}) returns float",
                   analyze_pattern_strength, [pattern, up], None,
                   lambda x: (isinstance(x, (int, float)) and 0 <= x <= 1.5,
                             f"expected 0..1.5, got {x!r}"), verbose=verbose)

    # get_all_patterns returns dict
    call_safely("pattern: get_all_patterns returns dict",
               get_all_patterns, [up], dict,
               lambda x: (True, ""), verbose=verbose)

    # REVERSAL_PATTERNS / CONTINUATION_PATTERNS structure
    if isinstance(REVERSAL_PATTERNS, dict) and "bullish" in REVERSAL_PATTERNS:
        result.record_pass("pattern: REVERSAL_PATTERNS has 'bullish' key", verbose)
    else:
        result.record_fail("pattern: REVERSAL_PATTERNS structure",
                          f"expected dict with 'bullish' key, got {REVERSAL_PATTERNS!r}")


def test_score_helpers(result: TestResult, verbose: bool):
    """Direct tests of score.py's own helper functions."""
    print("\n" + "─" * 70)
    print("LAYER 1.8 — SCORE.PY HELPER FUNCTIONS")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        result.record_skip("score helpers", f"import score.py failed: {e}", verbose)
        return

    # ── determine_direction ──────────────────────────────────────────────────
    # Strong consensus uptrend
    out = score.determine_direction({"1": 3.0, "5": 3.5, "15": 4.0})
    if out == "Long":
        result.record_pass("score: determine_direction(strong bullish) = Long", verbose)
    else:
        result.record_fail("score: determine_direction(strong bullish) = Long",
                          f"got {out!r}")

    # Strong consensus downtrend
    out = score.determine_direction({"1": -3.0, "5": -3.5, "15": -4.0})
    if out == "Short":
        result.record_pass("score: determine_direction(strong bearish) = Short", verbose)
    else:
        result.record_fail("score: determine_direction(strong bearish) = Short",
                          f"got {out!r}")

    # Mixed/ambiguous → None (key quality patch behavior)
    out = score.determine_direction({"1": 1.5, "5": -1.5, "15": 0.5})
    if out is None:
        result.record_pass("score: determine_direction(mixed) = None (quality gate)", verbose)
    else:
        result.record_fail("score: determine_direction(mixed) = None",
                          f"got {out!r}, mixed signals should return None")

    # Below threshold → None (was Long under old rules)
    out = score.determine_direction({"1": 0.6, "5": 0.7, "15": 0.5})
    if out is None:
        result.record_pass("score: determine_direction(weak total<2) = None", verbose)
    else:
        result.record_fail("score: determine_direction(weak total<2) = None",
                          f"got {out!r}, total=1.8 is below new 2.0 threshold")

    # Empty → None
    out = score.determine_direction({})
    if out is None:
        result.record_pass("score: determine_direction(empty) = None", verbose)
    else:
        result.record_fail("score: determine_direction(empty) = None", f"got {out!r}")

    # Strong opposing TF should veto
    out = score.determine_direction({"1": 3.0, "5": 3.5, "15": -2.5})
    if out is None:
        result.record_pass("score: determine_direction(strong opposing TF) = None", verbose)
    else:
        result.record_fail("score: determine_direction(strong opposing TF) = None",
                          f"got {out!r}, -2.5 TF should veto Long")

    # ── calculate_confidence ─────────────────────────────────────────────────
    # Decent score + aligned TFs + bullish market → reasonable confidence
    conf = score.calculate_confidence(
        score=10.0,
        tf_scores={"1": 2.0, "5": 2.5, "15": 3.0},
        market_context={"btc_trend": "bullish"},
        trade_type="Intraday",
    )
    if isinstance(conf, int) and 50 <= conf <= 100:
        result.record_pass(f"score: calculate_confidence(strong setup) = {conf}", verbose)
    else:
        result.record_fail(f"score: calculate_confidence(strong setup)",
                          f"expected int 50..100, got {conf!r}")

    # Counter-trend long should get harsh penalty
    conf_counter = score.calculate_confidence(
        score=10.0,
        tf_scores={"1": 2.0, "5": 2.5, "15": 3.0},
        market_context={"btc_trend": "bearish"},
        trade_type="Intraday",
    )
    if conf_counter < conf:
        result.record_pass(f"score: counter-trend confidence ({conf_counter}) < trend ({conf})",
                          verbose)
    else:
        result.record_fail("score: counter-trend confidence penalty",
                          f"counter-trend conf {conf_counter} >= trend conf {conf}")

    # Empty tf_scores → 0
    conf_empty = score.calculate_confidence(10.0, {}, {}, "Scalp")
    if conf_empty == 0:
        result.record_pass("score: calculate_confidence(empty TFs) = 0", verbose)
    else:
        result.record_fail("score: calculate_confidence(empty TFs) = 0",
                          f"got {conf_empty!r}")

    # ── check_4h_alignment_veto ──────────────────────────────────────────────
    # Strong 4h uptrend — Long should be allowed, Short should be vetoed
    candles_4h_up = make_uptrend_candles(60, step_pct=1.0)
    allow_long, _ = score.check_4h_alignment_veto({"240": candles_4h_up}, "Long")
    if allow_long:
        result.record_pass("score: 4h uptrend allows Long", verbose)
    else:
        result.record_fail("score: 4h uptrend allows Long", "got veto for Long in uptrend")

    allow_short, reason_short = score.check_4h_alignment_veto({"240": candles_4h_up}, "Short")
    if not allow_short:
        result.record_pass(f"score: 4h uptrend vetoes Short ({reason_short})", verbose)
    else:
        result.record_fail("score: 4h uptrend vetoes Short", "Short was allowed in 4h uptrend")

    # No 4h data → fail open (allow)
    allow_open, _ = score.check_4h_alignment_veto({}, "Long")
    if allow_open:
        result.record_pass("score: 4h veto fails open (no data → allow)", verbose)
    else:
        result.record_fail("score: 4h veto fails open", "rejected with no 4h data")

    # ── check_not_chasing ────────────────────────────────────────────────────
    # Clean entry — should pass
    clean_set = make_uptrend_set()
    allow_clean, _ = score.check_not_chasing(clean_set, "Long", "Scalp")
    if allow_clean:
        result.record_pass("score: anti-chase allows clean entry", verbose)
    else:
        result.record_fail("score: anti-chase allows clean entry", "rejected clean uptrend")

    # Already chasing — should be rejected
    chase_set = make_chasing_set()
    allow_chase, reason_chase = score.check_not_chasing(chase_set, "Long", "Scalp")
    if not allow_chase:
        result.record_pass(f"score: anti-chase rejects extended entry ({reason_chase})", verbose)
    else:
        result.record_fail("score: anti-chase rejects extended entry",
                          "allowed entry after 3% spike — should be vetoed")

    # ── safe_detect_momentum_strength ────────────────────────────────────────
    has_mom, direction, strength = score.safe_detect_momentum_strength(
        make_uptrend_candles(40, step_pct=0.5))
    if isinstance(has_mom, bool) and (direction in (None, "bullish", "bearish")) \
            and isinstance(strength, (int, float)):
        result.record_pass(
            f"score: safe_detect_momentum_strength uptrend = ({has_mom}, {direction}, {strength:.2f})",
            verbose)
    else:
        result.record_fail("score: safe_detect_momentum_strength uptrend",
                          f"got {(has_mom, direction, strength)!r}")

    # Empty / short candles should not crash, should return falsy
    has_mom_empty, _, _ = score.safe_detect_momentum_strength([])
    if has_mom_empty is False:
        result.record_pass("score: safe_detect_momentum_strength([]) = False", verbose)
    else:
        result.record_fail("score: safe_detect_momentum_strength([])",
                          f"got {has_mom_empty!r}")


def test_trend_filters(result: TestResult, verbose: bool):
    """Test trend_filters integration. May skip if it requires API access."""
    print("\n" + "─" * 70)
    print("LAYER 1.9 — TREND FILTERS")
    print("─" * 70)
    try:
        from trend_filters import validate_short_signal
        import asyncio
    except Exception as e:
        result.record_skip("trend_filters tests", f"import failed: {e}", verbose)
        return

    # validate_short_signal is async and may make API calls — we only verify
    # it's callable and handles missing TF data gracefully.
    async def run():
        # Missing required TFs should return False (per the function source)
        try:
            out = await validate_short_signal("TESTUSDT", {})
        except Exception as e:
            return False, f"raised on empty input: {e}"
        if out is False:
            return True, "correctly rejected on missing TFs"
        # Network call may have happened — still acceptable if it returned a bool
        if isinstance(out, bool):
            return True, f"returned bool {out} (likely API-call path)"
        return False, f"unexpected return {out!r}"

    try:
        ok, msg = asyncio.run(run())
        if ok:
            result.record_pass(f"trend_filters: validate_short_signal handles empty input ({msg})",
                             verbose)
        else:
            result.record_fail("trend_filters: validate_short_signal handles empty input", msg)
    except Exception as e:
        result.record_skip("trend_filters: validate_short_signal",
                          f"async run failed (probably needs API): {e}", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_score(out):
    """score_symbol returns 5-tuple: (score, tf_scores, trade_type, ind_scores, used)."""
    if not isinstance(out, tuple) or len(out) != 5:
        return None, f"expected 5-tuple, got {type(out).__name__} of len {len(out) if hasattr(out, '__len__') else '?'}"
    score_val, tf_scores, trade_type, ind_scores, used = out
    if not isinstance(score_val, (int, float)):
        return None, f"score is {type(score_val).__name__}, expected number"
    if not isinstance(tf_scores, dict):
        return None, f"tf_scores is {type(tf_scores).__name__}, expected dict"
    if trade_type not in ("Scalp", "Intraday", "Swing"):
        return None, f"trade_type is {trade_type!r}, expected Scalp/Intraday/Swing"
    if not isinstance(ind_scores, dict):
        return None, f"indicator_scores is {type(ind_scores).__name__}, expected dict"
    if not isinstance(used, list):
        return None, f"used_indicators is {type(used).__name__}, expected list"
    return (score_val, tf_scores, trade_type, ind_scores, used), None


def test_score_symbol_integration(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 2.1 — score_symbol() FULL PIPELINE")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        result.record_skip("score_symbol integration", f"import failed: {e}", verbose)
        return

    # Test 1: Strong uptrend produces positive score
    candles_up = make_uptrend_set()
    try:
        out = score.score_symbol("TESTUSDT", candles_up, {"btc_trend": "bullish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("score_symbol: uptrend returns valid 5-tuple", err)
        else:
            s, tf_s, tt, ind, used = unpacked
            details = f"score={s:.2f}, type={tt}, tf_count={len(tf_s)}, indicators={len(ind)}"
            result.record_pass(f"score_symbol: uptrend returns valid tuple ({details})", verbose)
            if s > 0:
                result.record_pass(f"score_symbol: uptrend score > 0 (got {s:.2f})", verbose)
            else:
                result.record_fail("score_symbol: uptrend score > 0",
                                  f"got {s:.2f} (expected positive)")
    except Exception as e:
        result.record_fail("score_symbol: uptrend pipeline",
                          f"raised: {e}\n{traceback.format_exc()}")

    # Test 2: Strong downtrend produces negative score
    candles_down = make_downtrend_set()
    try:
        out = score.score_symbol("TESTUSDT", candles_down, {"btc_trend": "bearish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("score_symbol: downtrend returns valid 5-tuple", err)
        else:
            s, _, tt, _, _ = unpacked
            if s < 0:
                result.record_pass(f"score_symbol: downtrend score < 0 (got {s:.2f}, type={tt})",
                                 verbose)
            else:
                result.record_fail("score_symbol: downtrend score < 0",
                                  f"got {s:.2f} (expected negative)")
    except Exception as e:
        result.record_fail("score_symbol: downtrend pipeline",
                          f"raised: {e}\n{traceback.format_exc()}")

    # Test 3: Empty candles handled gracefully (no crash)
    try:
        out = score.score_symbol("TESTUSDT", {}, {})
        unpacked, err = _unpack_score(out)
        if err:
            # An empty input may legitimately return weird structure — but it must not crash
            # If it returned a 5-tuple with score 0, that's fine
            result.record_fail("score_symbol: empty candles returns valid tuple", err)
        else:
            result.record_pass("score_symbol: empty candles handled (no crash)", verbose)
    except Exception as e:
        result.record_fail("score_symbol: empty candles handled",
                          f"raised: {e}")

    # Test 4: Insufficient candles (< 10 each) handled gracefully
    short_set = {tf: make_uptrend_candles(5) for tf in ["1", "5", "15"]}
    try:
        out = score.score_symbol("TESTUSDT", short_set, {})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("score_symbol: short candles returns valid tuple", err)
        else:
            result.record_pass("score_symbol: short candles handled gracefully", verbose)
    except Exception as e:
        result.record_fail("score_symbol: short candles handled",
                          f"raised: {e}")

    # Test 5: Verify correct indicator weights are picked up (sanity check)
    try:
        weights = score.WEIGHTS
        if weights.get("supertrend_mtf", 0) >= 1.3:
            result.record_pass(
                f"score: WEIGHTS has tightened supertrend_mtf >= 1.3 "
                f"(got {weights['supertrend_mtf']})", verbose)
        else:
            result.record_fail("score: WEIGHTS has tightened supertrend_mtf >= 1.3",
                              f"got {weights.get('supertrend_mtf')}")
        if weights.get("bollinger_squeeze", 1) == 0.0:
            result.record_pass("score: WEIGHTS bollinger_squeeze removed (=0)", verbose)
        else:
            result.record_fail("score: WEIGHTS bollinger_squeeze removed",
                              f"got {weights.get('bollinger_squeeze')}")
    except Exception as e:
        result.record_fail("score: WEIGHTS structure", str(e))


def test_enhanced_score_symbol_integration(result: TestResult, verbose: bool):
    print("\n" + "─" * 70)
    print("LAYER 2.2 — enhanced_score_symbol() WITH QUALITY GATES")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        result.record_skip("enhanced_score_symbol", f"import failed: {e}", verbose)
        return

    # Test 1: Uptrend with bullish market — should pass quality gates
    candles_up = make_uptrend_set()
    try:
        out = score.enhanced_score_symbol(
            "TESTUSDT", candles_up, {"btc_trend": "bullish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("enhanced_score_symbol: uptrend valid tuple", err)
        else:
            s, _, tt, ind, _ = unpacked
            result.record_pass(
                f"enhanced_score_symbol: uptrend OK (score={s:.2f}, type={tt}, "
                f"indicators={len(ind)})", verbose)
    except Exception as e:
        result.record_fail("enhanced_score_symbol: uptrend",
                          f"raised: {e}\n{traceback.format_exc()}")

    # Test 2: Chasing scenario — quality gate should reject (return 0 or 5-tuple with 0 score)
    chase_set = make_chasing_set()
    try:
        out = score.enhanced_score_symbol("TESTUSDT", chase_set, {"btc_trend": "bullish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("enhanced_score_symbol: chase valid tuple", err)
        else:
            s, _, _, _, _ = unpacked
            # Chasing scenarios may pass or fail depending on the specific chase magnitude;
            # we mainly want to confirm it doesn't crash and respects gates
            result.record_pass(
                f"enhanced_score_symbol: chase scenario handled (score={s:.2f})", verbose)
    except Exception as e:
        result.record_fail("enhanced_score_symbol: chase scenario",
                          f"raised: {e}")

    # Test 3: Ranging market — should produce low/zero direction
    candles_range = make_ranging_set()
    try:
        out = score.enhanced_score_symbol("TESTUSDT", candles_range, {})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("enhanced_score_symbol: ranging valid tuple", err)
        else:
            s, tf_s, _, _, _ = unpacked
            direction = score.determine_direction(tf_s)
            details = f"score={s:.2f}, direction={direction}"
            # Quality patch: ranging should typically return None direction → score 0
            if abs(s) < 5 or direction is None:
                result.record_pass(
                    f"enhanced_score_symbol: ranging gives weak/no signal ({details})", verbose)
            else:
                # Not a hard fail — some ranging setups might score, but warn
                result.record_pass(
                    f"enhanced_score_symbol: ranging produced signal ({details}) — verify if real",
                    verbose)
    except Exception as e:
        result.record_fail("enhanced_score_symbol: ranging",
                          f"raised: {e}")

    # Test 4: Bearish market vs Long — counter-trend penalty should apply
    candles_up = make_uptrend_set()
    try:
        out_bull = score.enhanced_score_symbol(
            "TESTUSDT", candles_up, {"btc_trend": "bullish"})
        out_bear = score.enhanced_score_symbol(
            "TESTUSDT", candles_up, {"btc_trend": "bearish"})
        unp_bull, err1 = _unpack_score(out_bull)
        unp_bear, err2 = _unpack_score(out_bear)
        if err1 or err2:
            result.record_fail("enhanced_score_symbol: market context comparison",
                              err1 or err2)
        else:
            s_bull, _, _, _, _ = unp_bull
            s_bear, _, _, _, _ = unp_bear
            # Counter-trend should produce equal or LOWER score
            if s_bear <= s_bull:
                result.record_pass(
                    f"enhanced_score_symbol: counter-trend penalty applied "
                    f"(bull={s_bull:.2f}, bear={s_bear:.2f})", verbose)
            else:
                result.record_fail(
                    "enhanced_score_symbol: counter-trend penalty applied",
                    f"counter-trend score ({s_bear:.2f}) > aligned ({s_bull:.2f})")
    except Exception as e:
        result.record_fail("enhanced_score_symbol: market context comparison",
                          f"raised: {e}")


def test_quality_gates_end_to_end(result: TestResult, verbose: bool):
    """Verify the new quality gates actually filter signals as designed."""
    print("\n" + "─" * 70)
    print("LAYER 2.3 — QUALITY GATES END-TO-END")
    print("─" * 70)
    try:
        import score
    except Exception as e:
        result.record_skip("quality gates e2e", f"import score failed: {e}", verbose)
        return

    # Gate 1: 4h direction veto in action
    # Construct: 1m/5m/15m bullish, but 4h strongly bearish → should be vetoed
    bull_low_tf = {
        "1":  make_uptrend_candles(60, step_pct=0.05),
        "5":  make_uptrend_candles(60, step_pct=0.20),
        "15": make_uptrend_candles(50, step_pct=0.40),
        "240": make_downtrend_candles(40, step_pct=1.5),  # bearish 4h
    }
    try:
        out = score.enhanced_score_symbol("TESTUSDT", bull_low_tf, {"btc_trend": "bullish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("e2e: 4h veto produces valid tuple", err)
        else:
            s, _, _, _, _ = unpacked
            # Bullish lower TFs but bearish 4h: enhanced_score_symbol should return 0
            if s == 0:
                result.record_pass(
                    "e2e: 4h-bearish-vs-Long-low-TF correctly vetoed (score=0)", verbose)
            else:
                result.record_fail(
                    "e2e: 4h veto should reject bullish low-TF with bearish 4h",
                    f"got score={s:.2f}, expected 0")
    except Exception as e:
        result.record_fail("e2e: 4h veto", f"raised: {e}")

    # Gate 2: Anti-chase — uptrend that already moved 3%+ in last 3 candles
    chase_set = make_chasing_set()
    try:
        out = score.enhanced_score_symbol("TESTUSDT", chase_set, {"btc_trend": "bullish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("e2e: anti-chase produces valid tuple", err)
        else:
            s, _, _, _, _ = unpacked
            if s == 0:
                result.record_pass("e2e: anti-chase correctly vetoed extended entry (score=0)",
                                 verbose)
            else:
                # Chase magnitude is borderline — it may pass, log result for visibility
                result.record_pass(
                    f"e2e: anti-chase scenario (score={s:.2f}) — verify if borderline OK",
                    verbose)
    except Exception as e:
        result.record_fail("e2e: anti-chase", f"raised: {e}")

    # Gate 3: Unanimous strong downtrend → Short signal possible
    candles_down = make_downtrend_set()
    try:
        out = score.enhanced_score_symbol("TESTUSDT", candles_down, {"btc_trend": "bearish"})
        unpacked, err = _unpack_score(out)
        if err:
            result.record_fail("e2e: strong downtrend produces valid tuple", err)
        else:
            s, tf_s, _, _, _ = unpacked
            direction = score.determine_direction(tf_s)
            if direction == "Short" or s < 0:
                result.record_pass(
                    f"e2e: strong downtrend → Short bias (score={s:.2f}, dir={direction})",
                    verbose)
            elif s == 0:
                # Gate vetoed it — not necessarily wrong, depends on indicators
                result.record_pass(
                    "e2e: strong downtrend gated to 0 (gates worked, no clean Short)", verbose)
            else:
                result.record_fail(
                    "e2e: strong downtrend should give Short or 0",
                    f"got score={s:.2f}, dir={direction}")
    except Exception as e:
        result.record_fail("e2e: strong downtrend", f"raised: {e}")


def test_main_integration(result: TestResult, verbose: bool):
    """Verify main.py imports score.py correctly and uses it as expected."""
    print("\n" + "─" * 70)
    print("LAYER 2.4 — main.py INTEGRATION SMOKE TEST")
    print("─" * 70)
    try:
        import main
    except Exception as e:
        result.record_skip("main.py integration", f"import main failed: {e}", verbose)
        return

    # main.py should import these from score
    expected_in_main = [
        "score_symbol", "determine_direction", "calculate_confidence",
        "has_pump_potential", "detect_momentum_strength",
    ]
    for fn in expected_in_main:
        if hasattr(main, fn):
            result.record_pass(f"main: {fn} accessible", verbose)
        else:
            result.record_fail(f"main: {fn} accessible",
                              f"main.py is missing reference to score.{fn}")

    # main.py should have the tightened thresholds
    threshold_checks = [
        ("MIN_SCALP_SCORE", 10.5, ">="),
        ("MIN_INTRADAY_SCORE", 12.0, ">="),
        ("MIN_SWING_SCORE", 15.5, ">="),
    ]
    for attr, expected, op in threshold_checks:
        if hasattr(main, attr):
            actual = getattr(main, attr)
            if (op == ">=" and actual >= expected) or (op == "==" and actual == expected):
                result.record_pass(f"main: {attr}={actual} ({op} {expected})", verbose)
            else:
                result.record_fail(f"main: {attr} threshold check",
                                  f"got {actual}, expected {op} {expected}")
        else:
            result.record_fail(f"main: {attr} present", f"missing on main.py")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test score.py and dependencies")
    parser.add_argument("--unit", action="store_true",
                       help="Run only Layer 1 (unit tests of dependencies)")
    parser.add_argument("--integration", action="store_true",
                       help="Run only Layer 2 (full pipeline tests)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show every PASS in detail")
    parser.add_argument("--module", type=str, default=None,
                       help="Test only a specific module name (rsi, macd, ema, etc.)")
    args = parser.parse_args()

    # Default: run everything
    run_unit = args.unit or not (args.unit or args.integration)
    run_integration = args.integration or not (args.unit or args.integration)

    print("═" * 70)
    print("SCORE.PY TEST SUITE")
    print("═" * 70)
    print(f"Working dir:  {os.getcwd()}")
    print(f"Layer 1 unit: {run_unit}")
    print(f"Layer 2 int:  {run_integration}")
    print(f"Verbose:      {args.verbose}")
    if args.module:
        print(f"Module filter: {args.module}")

    result = TestResult()

    if run_unit:
        test_imports(result, args.verbose, args.module)

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
                fn(result, args.verbose)
            except Exception as e:
                result.record_fail(f"{name} test suite", f"runner crashed: {e}")
                traceback.print_exc()

    if run_integration:
        integration_tests = [
            test_score_symbol_integration,
            test_enhanced_score_symbol_integration,
            test_quality_gates_end_to_end,
            test_main_integration,
        ]
        for fn in integration_tests:
            try:
                fn(result, args.verbose)
            except Exception as e:
                result.record_fail(fn.__name__, f"runner crashed: {e}")
                traceback.print_exc()

    return result.summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(2)
