# ai_memory.py - Enhanced AI Memory System with Better Performance

import json
import os
import time
import asyncio
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from logger import log, write_log
import gzip
import shutil

# File paths
MEMORY_FILE = "ai_memory.json"
MEMORY_BACKUP = "ai_memory_backup.json"
MEMORY_FILE_GZ = "ai_memory.json.gz"  # Compressed storage for large datasets

# Global memory database with enhanced structure
memory_db = {
    "profiles": {},           # Score profile performance
    "patterns": {},          # Pattern-based memory
    "market_conditions": {}, # Market condition memory
    "time_based": {},       # Time-based performance
    "symbol_specific": {},  # Symbol-specific patterns
    "strategy_stats": {},   # Strategy performance stats
}

# Performance cache
_performance_cache = {}
_cache_ttl = 300  # 5 minutes cache TTL
_cache_timestamps = {}

# Thread lock for concurrent access
_memory_lock = threading.Lock()

# Configuration
MAX_PROFILES_PER_TYPE = 1000  # Maximum profiles to keep in memory
MIN_SAMPLES_FOR_CONFIDENCE = 5  # Minimum samples before trusting statistics
PROFILE_EXPIRY_DAYS = 30  # Days before expiring unused profiles
COMPRESSION_THRESHOLD = 10 * 1024 * 1024  # 10MB - compress if file larger

class EnhancedMemoryDB:
    """Enhanced memory database with better performance and features"""
    
    def __init__(self):
        self.profiles = defaultdict(lambda: {"win": 0, "loss": 0, "breakeven": 0, "total": 0, "pnl": []})
        self.recent_trades = deque(maxlen=1000)  # Keep last 1000 trades for analysis
        self.performance_by_hour = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        self.performance_by_day = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        self.symbol_stats = defaultdict(lambda: {"total": 0, "wins": 0, "avg_pnl": 0})
        
    def add_trade(self, trade_data: Dict):
        """Add a trade to recent history for pattern analysis"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.recent_trades.append(trade_data)
        
        # Update hourly stats
        hour = datetime.now().hour
        self.performance_by_hour[hour]["trades"] += 1
        if trade_data.get('result') == 'win':
            self.performance_by_hour[hour]["wins"] += 1
        self.performance_by_hour[hour]["total_pnl"] += trade_data.get('pnl', 0)
        
        # Update daily stats
        day_of_week = datetime.now().weekday()
        self.performance_by_day[day_of_week]["trades"] += 1
        if trade_data.get('result') == 'win':
            self.performance_by_day[day_of_week]["wins"] += 1
        self.performance_by_day[day_of_week]["total_pnl"] += trade_data.get('pnl', 0)

# Initialize enhanced memory
enhanced_memory = EnhancedMemoryDB()

def clean_key(tf_scores: Dict[str, float], precision: int = 1) -> str:
    """
    Normalize scores to specified decimal places to group similar profiles
    
    Args:
        tf_scores: Timeframe scores dictionary
        precision: Decimal places for rounding
        
    Returns:
        str: Normalized key for grouping
    """
    if not tf_scores:
        return "empty"
        
    # Sort keys for consistent ordering
    sorted_items = sorted(tf_scores.items())
    
    # Round values and create key
    normalized = {k: round(v, precision) for k, v in sorted_items}
    
    # Create a hash for very long keys to save memory
    key_str = json.dumps(normalized, sort_keys=True)
    if len(key_str) > 100:
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    return key_str

def log_trade_result(symbol: str, tf_scores: Dict[str, float], result: str, 
                    pnl: float = 0, trade_type: str = None, 
                    market_conditions: Dict = None):
    """
    Enhanced trade result logging with multiple dimensions
    
    Args:
        symbol: Trading symbol
        tf_scores: Timeframe scores
        result: "win", "loss", or "breakeven"
        pnl: Profit/loss percentage
        trade_type: Type of trade (Scalp/Intraday/Swing)
        market_conditions: Current market conditions
    """
    with _memory_lock:
        try:
            # Log to main profile database
            profile_key = clean_key(tf_scores)
            
            if profile_key not in memory_db["profiles"]:
                memory_db["profiles"][profile_key] = {
                    "win": 0, "loss": 0, "breakeven": 0, "total": 0,
                    "pnl": [], "last_seen": None, "symbols": set()
                }
            
            profile = memory_db["profiles"][profile_key]
            profile[result] += 1
            profile["total"] += 1
            profile["pnl"].append(pnl)
            profile["last_seen"] = datetime.now().isoformat()
            
            # Convert set to list for JSON serialization
            if isinstance(profile["symbols"], set):
                profile["symbols"].add(symbol)
                profile["symbols"] = list(profile["symbols"])
            else:
                if symbol not in profile["symbols"]:
                    profile["symbols"].append(symbol)
            
            # Keep only last 100 PnL values to save memory
            if len(profile["pnl"]) > 100:
                profile["pnl"] = profile["pnl"][-100:]
            
            # Log to enhanced memory
            enhanced_memory.add_trade({
                "symbol": symbol,
                "result": result,
                "pnl": pnl,
                "trade_type": trade_type,
                "profile_key": profile_key,
                "market_conditions": market_conditions
            })
            
            # Update symbol-specific stats
            update_symbol_stats(symbol, result, pnl)
            
            # Update strategy stats if available
            if trade_type:
                update_strategy_stats(trade_type, result, pnl)
            
            # Cleanup old profiles if needed
            if len(memory_db["profiles"]) > MAX_PROFILES_PER_TYPE:
                cleanup_old_profiles()
            
            # Save to disk (async to avoid blocking)
            asyncio.create_task(save_memory_async())
            
        except Exception as e:
            log(f"❌ Error logging trade result: {e}", level="ERROR")

def update_symbol_stats(symbol: str, result: str, pnl: float):
    """Update symbol-specific statistics"""
    if symbol not in memory_db["symbol_specific"]:
        memory_db["symbol_specific"][symbol] = {
            "total": 0, "wins": 0, "losses": 0, 
            "total_pnl": 0, "best_trade": 0, "worst_trade": 0,
            "avg_win": 0, "avg_loss": 0, "win_rate": 0
        }
    
    stats = memory_db["symbol_specific"][symbol]
    stats["total"] += 1
    
    if result == "win":
        stats["wins"] += 1
        stats["avg_win"] = ((stats["avg_win"] * (stats["wins"] - 1)) + pnl) / stats["wins"]
    elif result == "loss":
        stats["losses"] += 1
        stats["avg_loss"] = ((stats["avg_loss"] * (stats["losses"] - 1)) + pnl) / stats["losses"]
    
    stats["total_pnl"] += pnl
    stats["best_trade"] = max(stats["best_trade"], pnl)
    stats["worst_trade"] = min(stats["worst_trade"], pnl)
    stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0

def update_strategy_stats(strategy: str, result: str, pnl: float):
    """Update strategy-specific statistics"""
    if strategy not in memory_db["strategy_stats"]:
        memory_db["strategy_stats"][strategy] = {
            "total": 0, "wins": 0, "losses": 0,
            "total_pnl": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "pnl_history": []
        }
    
    stats = memory_db["strategy_stats"][strategy]
    stats["total"] += 1
    
    if result == "win":
        stats["wins"] += 1
    elif result == "loss":
        stats["losses"] += 1
    
    stats["total_pnl"] += pnl
    stats["pnl_history"].append(pnl)
    
    # Keep only last 200 PnL values
    if len(stats["pnl_history"]) > 200:
        stats["pnl_history"] = stats["pnl_history"][-200:]
    
    # Calculate Sharpe ratio and max drawdown
    if len(stats["pnl_history"]) >= 20:
        returns = np.array(stats["pnl_history"])
        stats["sharpe_ratio"] = calculate_sharpe_ratio(returns)
        stats["max_drawdown"] = calculate_max_drawdown(returns)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0) -> float:
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns"""
    cumulative = (1 + returns / 100).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown)) * 100

def get_profile_confidence(tf_scores: Dict[str, float], 
                          market_conditions: Dict = None) -> Tuple[float, Dict]:
    """
    Enhanced confidence calculation with multiple factors
    
    Returns:
        Tuple of (confidence_score, detailed_metrics)
    """
    profile_key = clean_key(tf_scores)
    
    # Check cache first
    cache_key = f"{profile_key}_{str(market_conditions)}"
    if cache_key in _performance_cache and is_cache_valid(cache_key):
        return _performance_cache[cache_key]
    
    # Base confidence from profile performance
    base_confidence = 0.5
    metrics = {
        "samples": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "sharpe_ratio": 0,
        "time_performance": 0,
        "market_alignment": 0
    }
    
    # Get profile stats
    if profile_key in memory_db["profiles"]:
        stats = memory_db["profiles"][profile_key]
        
        if stats["total"] >= MIN_SAMPLES_FOR_CONFIDENCE:
            metrics["samples"] = stats["total"]
            metrics["win_rate"] = stats["win"] / stats["total"]
            
            # Calculate profit factor
            if stats["pnl"]:
                wins = [p for p in stats["pnl"] if p > 0]
                losses = [abs(p) for p in stats["pnl"] if p < 0]
                
                if wins and losses:
                    avg_win = np.mean(wins)
                    avg_loss = np.mean(losses)
                    metrics["profit_factor"] = avg_win / avg_loss if avg_loss > 0 else 0
                
                # Calculate Sharpe ratio
                if len(stats["pnl"]) >= 20:
                    metrics["sharpe_ratio"] = calculate_sharpe_ratio(np.array(stats["pnl"]))
            
            # Base confidence from win rate and profit factor
            base_confidence = metrics["win_rate"] * 0.4 + \
                            min(metrics["profit_factor"] / 3, 0.3) + \
                            min(metrics["sharpe_ratio"] / 3, 0.3)
    
    # Adjust for time-based performance
    current_hour = datetime.now().hour
    if current_hour in enhanced_memory.performance_by_hour:
        hour_stats = enhanced_memory.performance_by_hour[current_hour]
        if hour_stats["trades"] >= 10:
            hour_win_rate = hour_stats["wins"] / hour_stats["trades"]
            metrics["time_performance"] = hour_win_rate
            base_confidence *= (0.8 + hour_win_rate * 0.4)  # 0.8 to 1.2x multiplier
    
    # Adjust for market conditions
    if market_conditions:
        market_key = json.dumps(market_conditions, sort_keys=True)
        if market_key in memory_db["market_conditions"]:
            market_stats = memory_db["market_conditions"][market_key]
            if market_stats["total"] >= 10:
                market_win_rate = market_stats["wins"] / market_stats["total"]
                metrics["market_alignment"] = market_win_rate
                base_confidence *= (0.8 + market_win_rate * 0.4)
    
    # Cap confidence between 0 and 1
    final_confidence = max(0.1, min(0.95, base_confidence))
    
    # Cache the result
    _performance_cache[cache_key] = (final_confidence, metrics)
    _cache_timestamps[cache_key] = time.time()
    
    return final_confidence, metrics

def get_best_performers(top_n: int = 10, min_trades: int = 20) -> List[Dict]:
    """Get top performing profiles"""
    performers = []
    
    for profile_key, stats in memory_db["profiles"].items():
        if stats["total"] >= min_trades:
            win_rate = stats["win"] / stats["total"]
            avg_pnl = np.mean(stats["pnl"]) if stats["pnl"] else 0
            
            performers.append({
                "profile": profile_key,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "total_trades": stats["total"],
                "profit_factor": calculate_profit_factor(stats),
                "symbols": stats.get("symbols", [])
            })
    
    # Sort by a composite score
    performers.sort(key=lambda x: x["win_rate"] * 0.5 + min(x["profit_factor"] / 4, 0.5), 
                   reverse=True)
    
    return performers[:top_n]

def calculate_profit_factor(stats: Dict) -> float:
    """Calculate profit factor from stats"""
    if not stats.get("pnl"):
        return 0
    
    wins = [p for p in stats["pnl"] if p > 0]
    losses = [abs(p) for p in stats["pnl"] if p < 0]
    
    if not losses:
        return float('inf') if wins else 0
    
    return sum(wins) / sum(losses) if wins else 0

def cleanup_old_profiles():
    """Remove old, unused profiles to save memory"""
    current_time = datetime.now()
    profiles_to_remove = []
    
    for profile_key, stats in memory_db["profiles"].items():
        # Remove profiles not seen in PROFILE_EXPIRY_DAYS
        if stats.get("last_seen"):
            last_seen = datetime.fromisoformat(stats["last_seen"])
            if (current_time - last_seen).days > PROFILE_EXPIRY_DAYS:
                profiles_to_remove.append(profile_key)
        
        # Remove profiles with very few samples after a long time
        elif stats["total"] < 3:
            profiles_to_remove.append(profile_key)
    
    for key in profiles_to_remove:
        del memory_db["profiles"][key]
    
    if profiles_to_remove:
        log(f"🧹 Cleaned up {len(profiles_to_remove)} old profiles")

def is_cache_valid(cache_key: str) -> bool:
    """Check if cached value is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    return (time.time() - _cache_timestamps[cache_key]) < _cache_ttl

def get_market_insights() -> Dict:
    """Get comprehensive market insights from memory"""
    insights = {
        "best_hours": [],
        "best_days": [],
        "best_symbols": [],
        "best_strategies": [],
        "market_conditions": {},
        "recent_performance": {}
    }
    
    # Best trading hours
    hour_performance = []
    for hour, stats in enhanced_memory.performance_by_hour.items():
        if stats["trades"] >= 10:
            win_rate = stats["wins"] / stats["trades"]
            avg_pnl = stats["total_pnl"] / stats["trades"]
            hour_performance.append({
                "hour": hour,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "trades": stats["trades"]
            })
    
    insights["best_hours"] = sorted(hour_performance, 
                                   key=lambda x: x["win_rate"] * x["avg_pnl"], 
                                   reverse=True)[:3]
    
    # Best trading days
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_performance = []
    for day, stats in enhanced_memory.performance_by_day.items():
        if stats["trades"] >= 10:
            win_rate = stats["wins"] / stats["trades"]
            avg_pnl = stats["total_pnl"] / stats["trades"]
            day_performance.append({
                "day": day_names[day],
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "trades": stats["trades"]
            })
    
    insights["best_days"] = sorted(day_performance, 
                                  key=lambda x: x["win_rate"] * x["avg_pnl"], 
                                  reverse=True)[:3]
    
    # Best symbols
    symbol_performance = []
    for symbol, stats in memory_db["symbol_specific"].items():
        if stats["total"] >= 10:
            symbol_performance.append({
                "symbol": symbol,
                "win_rate": stats["win_rate"],
                "avg_pnl": stats["total_pnl"] / stats["total"],
                "trades": stats["total"]
            })
    
    insights["best_symbols"] = sorted(symbol_performance, 
                                     key=lambda x: x["win_rate"] * x["avg_pnl"], 
                                     reverse=True)[:5]
    
    # Recent performance (last 50 trades)
    recent_trades = list(enhanced_memory.recent_trades)[-50:]
    if recent_trades:
        recent_wins = sum(1 for t in recent_trades if t.get("result") == "win")
        recent_pnl = sum(t.get("pnl", 0) for t in recent_trades)
        
        insights["recent_performance"] = {
            "trades": len(recent_trades),
            "win_rate": recent_wins / len(recent_trades),
            "total_pnl": recent_pnl,
            "avg_pnl": recent_pnl / len(recent_trades)
        }
    
    return insights

async def save_memory_async():
    """Asynchronously save memory to disk with compression"""
    try:
        # Create a copy to avoid locking issues
        memory_copy = json.dumps(memory_db, default=str)
        
        # Check if compression is needed
        if len(memory_copy) > COMPRESSION_THRESHOLD:
            # Save compressed
            with gzip.open(MEMORY_FILE_GZ, 'wt', encoding='utf-8') as f:
                f.write(memory_copy)
            log(f"💾 Memory saved (compressed: {len(memory_copy) / 1024 / 1024:.1f}MB)")
        else:
            # Save uncompressed with backup
            # First write to temp file
            temp_file = f"{MEMORY_FILE}.tmp"
            with open(temp_file, 'w') as f:
                f.write(memory_copy)
            
            # Backup existing file
            if os.path.exists(MEMORY_FILE):
                shutil.copy2(MEMORY_FILE, MEMORY_BACKUP)
            
            # Move temp to actual
            shutil.move(temp_file, MEMORY_FILE)
            
    except Exception as e:
        log(f"❌ Failed to save AI memory: {e}", level="ERROR")

def save_memory():
    """Synchronous save wrapper"""
    asyncio.create_task(save_memory_async())

def load_memory():
    """Load memory from disk with compression support"""
    global memory_db, enhanced_memory
    
    try:
        # Try loading compressed file first
        if os.path.exists(MEMORY_FILE_GZ):
            with gzip.open(MEMORY_FILE_GZ, 'rt', encoding='utf-8') as f:
                memory_db = json.load(f)
            log("🔁 AI memory loaded from compressed file")
            
        elif os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                memory_db = json.load(f)
            log("🔁 AI memory loaded from disk")
            
        elif os.path.exists(MEMORY_BACKUP):
            # Try backup if main file is corrupted
            with open(MEMORY_BACKUP, 'r') as f:
                memory_db = json.load(f)
            log("🔁 AI memory loaded from backup")
            
        else:
            log("📝 Starting with fresh AI memory")
            
        # Validate and clean loaded data
        validate_memory_structure()
        
        # Convert JSON lists back to sets where needed
        for profile in memory_db.get("profiles", {}).values():
            if "symbols" in profile and isinstance(profile["symbols"], list):
                profile["symbols"] = set(profile["symbols"])
                
    except Exception as e:
        log(f"❌ Failed to load AI memory: {e}", level="ERROR")
        # Start fresh if loading fails
        memory_db = {
            "profiles": {},
            "patterns": {},
            "market_conditions": {},
            "time_based": {},
            "symbol_specific": {},
            "strategy_stats": {}
        }

def validate_memory_structure():
    """Ensure memory database has correct structure"""
    required_keys = ["profiles", "patterns", "market_conditions", 
                    "time_based", "symbol_specific", "strategy_stats"]
    
    for key in required_keys:
        if key not in memory_db:
            memory_db[key] = {}
            
    # Clean up invalid entries
    profiles_to_remove = []
    for profile_key, data in memory_db["profiles"].items():
        if not isinstance(data, dict) or "total" not in data:
            profiles_to_remove.append(profile_key)
            
    for key in profiles_to_remove:
        del memory_db["profiles"][key]
        
    if profiles_to_remove:
        log(f"🧹 Removed {len(profiles_to_remove)} invalid profiles")

def export_memory_stats(filepath: str = "ai_memory_stats.json"):
    """Export memory statistics for analysis"""
    try:
        stats = {
            "total_profiles": len(memory_db["profiles"]),
            "total_trades": sum(p["total"] for p in memory_db["profiles"].values()),
            "best_performers": get_best_performers(),
            "market_insights": get_market_insights(),
            "symbol_stats": dict(memory_db["symbol_specific"]),
            "strategy_stats": dict(memory_db["strategy_stats"]),
            "export_time": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        log(f"📊 Memory statistics exported to {filepath}")
        
    except Exception as e:
        log(f"❌ Failed to export memory stats: {e}", level="ERROR")

# Periodic cleanup task
async def periodic_cleanup():
    """Periodically clean up old data and optimize memory"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        try:
            cleanup_old_profiles()
            
            # Clear old cache entries
            current_time = time.time()
            keys_to_remove = []
            for key, timestamp in _cache_timestamps.items():
                if current_time - timestamp > _cache_ttl * 2:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                _performance_cache.pop(key, None)
                _cache_timestamps.pop(key, None)
                
            if keys_to_remove:
                log(f"🧹 Cleared {len(keys_to_remove)} expired cache entries")
                
            # Export stats periodically
            if datetime.now().hour == 0:  # Once per day at midnight
                export_memory_stats()
                
        except Exception as e:
            log(f"❌ Error in periodic cleanup: {e}", level="ERROR")


