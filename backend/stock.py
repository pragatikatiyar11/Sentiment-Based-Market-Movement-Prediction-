import yfinance as yf

def get_stock(symbol):
    data = yf.download(symbol, period="5d")
    return data.tail(1).to_dict()
