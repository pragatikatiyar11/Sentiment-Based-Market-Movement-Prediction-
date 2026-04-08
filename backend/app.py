from flask import Flask, jsonify
from ml_model.sentiment import analyze

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API Running Successfully"}

@app.route("/predict/<text>")
def predict(text):
    result = analyze(text)

    sentiment = result[0]['label']

    return jsonify({
        "input": text,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)

from stock import get_stock

@app.route("/stock/<symbol>")
def stock(symbol):
    data = get_stock(symbol)
    return data
