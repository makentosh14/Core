from scanner import symbol_category_map

def get_symbol_category(symbol):
    """
    Returns 'linear' (futures) or 'spot' based on cached scanner result.
    Default is 'linear' for safety (most trading is done on futures).
    """
    return symbol_category_map.get(symbol, "linear")

def is_spot_symbol(symbol):
    return get_symbol_category(symbol) == "spot"

def is_futures_symbol(symbol):
    return get_symbol_category(symbol) == "linear"

def format_symbol_for_log(symbol):
    """
    Returns formatted string like: [SPOT] DOGEUSDT or [LINEAR] BTCUSDT
    """
    category = get_symbol_category(symbol)
    return f"[{category.upper()}] {symbol}"
