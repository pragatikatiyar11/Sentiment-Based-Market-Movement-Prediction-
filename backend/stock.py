import yfinance as yf

def get_stock(symbol):
    data = yf.download(symbol, period="5d", interval="1d")
    
    latest = data.tail(1)
    prev = data.tail(2).head(1)

    if latest.empty or prev.empty:
        return None

    latest_close = float(latest["Close"].values[0])
    prev_close = float(prev["Close"].values[0])

    change = latest_close - prev_close

    return {
        "price": latest_close,
        "change": change
    }
