import time

# In-memory signal cache to prevent duplicates
signal_cache = {}

def is_duplicate_signal(symbol, cooldown=2400):
    """
    Returns True if a signal was already sent for this symbol within the cooldown period.

    Args:
        symbol (str): The trading symbol (e.g., BTCUSDT).
        cooldown (int): Cooldown in seconds before the next signal is allowed (default 1800s = 30min).

    Returns:
        bool: True if duplicate, False if allowed.
    """
    now = time.time()
    last_signal_time = signal_cache.get(symbol)
    if last_signal_time and (now - last_signal_time) < cooldown:
        return True
    return False

def log_signal(symbol):
    """
    Logs the current time for a signal sent, to avoid duplicates.

    Args:
        symbol (str): The trading symbol.
    """
    signal_cache[symbol] = time.time()
