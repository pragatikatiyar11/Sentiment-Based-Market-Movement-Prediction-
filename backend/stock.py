import yfinance as yf

def get_stock(symbol):
    data = yf.download(symbol, period="5d", interval="1d")

    if data.empty:
        return None

    try:
        latest_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]

        change = float(latest_close - prev_close)

        return {
            "price": float(latest_close),
            "change": change
        }

    except:
        return None
