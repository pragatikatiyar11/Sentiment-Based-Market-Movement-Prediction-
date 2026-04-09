from flask import Flask, jsonify
from ml_model.sentiment import analyze
from stock import get_stock
from predictor import predict_action

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Stock Sentiment API Running"}

@app.route("/predict/<symbol>/<text>")
def predict(symbol, text):
    
    sentiment_result = analyze(text)
    stock_data = get_stock(symbol)

    if stock_data is None:
        return {"error": "Stock data not available"}

    action = predict_action(
        sentiment_result['label'],
        stock_data['change']
    )

    return jsonify({
        "stock": symbol,
        "sentiment": sentiment_result['label'],
        "confidence": sentiment_result['score'],
        "price": stock_data['price'],
        "change": stock_data['change'],
        "prediction": action
    })

if __name__ == "__main__":
    app.run(debug=True)
