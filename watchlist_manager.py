watchlist = set()

def add_to_watchlist(symbol):
    watchlist.add(symbol.upper())

def remove_from_watchlist(symbol):
    watchlist.discard(symbol.upper())

def get_watchlist():
    return list(watchlist)

def is_watched(symbol):
    return symbol.upper() in watchlist
