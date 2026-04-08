from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API Running Successfully"}

@app.route("/predict/<symbol>")
def predict(symbol):
    return jsonify({
        "stock": symbol,
        "prediction": "BUY",
        "confidence": 0.85
    })

if __name__ == "__main__":
    app.run(debug=True)
