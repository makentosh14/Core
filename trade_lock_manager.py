#!/usr/bin/env python3
"""
Trade Lock Manager - Prevents duplicate trades with robust locking mechanism
"""

import asyncio
import time
from typing import Dict, Set
from logger import log
from bybit_api import get_positions

class TradeLockManager:
    def __init__(self):
        # Multiple layers of protection
        self.processing_locks: Dict[str, asyncio.Lock] = {}  # Per-symbol locks
        self.pending_trades: Set[str] = set()  # Trades being processed
        self.confirmed_trades: Set[str] = set()  # Confirmed active trades
        self.signal_cooldowns: Dict[str, float] = {}  # Last signal time
        self.failed_attempts: Dict[str, int] = {}  # Count failed attempts
        
        # Configurable timeouts
        self.SIGNAL_COOLDOWN = 3600  # 1 hour between signals
        self.PENDING_TIMEOUT = 60  # 60 seconds for pending trades
        self.MAX_FAILED_ATTEMPTS = 3
        
    def get_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create a lock for a symbol"""
        if symbol not in self.processing_locks:
            self.processing_locks[symbol] = asyncio.Lock()
        return self.processing_locks[symbol]
    
    async def can_process_symbol(self, symbol: str, check_exchange: bool = True) -> tuple[bool, str]:
        """
        Check if we can process a trade for this symbol
        This is the MAIN gate that prevents duplicates
        """
        current_time = time.time()
        
        # Check 1: Is symbol locked for processing?
        lock = self.get_lock(symbol)
        if lock.locked():
            return False, "Symbol is locked (processing)"
        
        # Check 2: Is there a pending trade?
        if symbol in self.pending_trades:
            return False, "Trade pending"
        
        # Check 3: Is there a confirmed trade?
        if symbol in self.confirmed_trades:
            return False, "Active position exists"
        
        # Check 4: Check cooldown
        if symbol in self.signal_cooldowns:
            time_since_signal = current_time - self.signal_cooldowns[symbol]
            if time_since_signal < self.SIGNAL_COOLDOWN:
                remaining = self.SIGNAL_COOLDOWN - time_since_signal
                return False, f"Cooldown active ({remaining:.0f}s)"
        
        # Check 5: Too many failed attempts?
        if self.failed_attempts.get(symbol, 0) >= self.MAX_FAILED_ATTEMPTS:
            return False, "Too many failed attempts"
        
        # Check 6: Verify with exchange (most reliable)
        if check_exchange:
            try:
                positions = await get_positions()
                for pos in positions:
                    if pos.get("symbol") == symbol and float(pos.get("size", 0)) > 0:
                        self.confirmed_trades.add(symbol)
                        return False, "Exchange has position"
            except Exception as e:
                log(f"⚠️ Could not verify with exchange: {e}")
        
        return True, "OK"
    
    async def acquire_trade_lock(self, symbol: str) -> bool:
        """
        Acquire exclusive lock for trading a symbol
        Returns True if lock acquired, False if already locked
        """
        lock = self.get_lock(symbol)
    
        # Try to acquire lock without blocking
        try:
            # Use acquire() instead of acquire_nowait() for async context
            if not lock.locked():
                await lock.acquire()
                # Mark as pending
                self.pending_trades.add(symbol)
                self.signal_cooldowns[symbol] = time.time()
                log(f"🔒 Trade lock acquired for {symbol}")
                return True
            else:
                return False
        except Exception as e:
            log(f"❌ Failed to acquire lock for {symbol}: {e}")
            return False
    
    def release_trade_lock(self, symbol: str, success: bool = False):
        """Release trade lock after processing"""
        lock = self.get_lock(symbol)
        
        # Update state based on result
        if success:
            # Move from pending to confirmed
            self.pending_trades.discard(symbol)
            self.confirmed_trades.add(symbol)
            self.failed_attempts[symbol] = 0
            log(f"✅ Trade confirmed for {symbol}")
        else:
            # Failed attempt
            self.pending_trades.discard(symbol)
            self.failed_attempts[symbol] = self.failed_attempts.get(symbol, 0) + 1
            # Remove cooldown on failure to allow retry
            self.signal_cooldowns.pop(symbol, None)
            log(f"❌ Trade failed for {symbol} (attempt {self.failed_attempts[symbol]})")
        
        # Release the lock
        if lock.locked():
            lock.release()
            log(f"🔓 Trade lock released for {symbol}")
    
    def mark_position_closed(self, symbol: str):
        """Mark a position as closed"""
        self.confirmed_trades.discard(symbol)
        self.pending_trades.discard(symbol)
        self.signal_cooldowns[symbol] = time.time()  # Start cooldown
        log(f"📤 Position closed for {symbol}")
    
    async def sync_with_exchange(self):
        """Sync confirmed trades with exchange positions"""
        try:
            positions = await get_positions()
            active_symbols = set()
            
            for pos in positions:
                if float(pos.get("size", 0)) > 0:
                    active_symbols.add(pos.get("symbol"))
            
            # Update confirmed trades
            self.confirmed_trades = active_symbols
            
            # Clean up pending trades that are actually confirmed
            for symbol in list(self.pending_trades):
                if symbol in active_symbols:
                    self.pending_trades.discard(symbol)
                    self.confirmed_trades.add(symbol)
            
            log(f"🔄 Synced with exchange: {len(self.confirmed_trades)} active positions")
            
        except Exception as e:
            log(f"❌ Failed to sync with exchange: {e}")
    
    async def cleanup_stale_locks(self):
        """Clean up stale pending trades"""
        current_time = time.time()
        
        for symbol in list(self.pending_trades):
            if symbol in self.signal_cooldowns:
                time_pending = current_time - self.signal_cooldowns[symbol]
                if time_pending > self.PENDING_TIMEOUT:
                    self.pending_trades.discard(symbol)
                    log(f"🧹 Cleaned stale pending trade for {symbol}")

# Global instance
trade_lock_manager = TradeLockManager()
