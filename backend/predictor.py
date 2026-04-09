def predict_action(sentiment, change):
    
    if sentiment == "POSITIVE" and change > 0:
        return "BUY"
    
    elif sentiment == "NEGATIVE" and change < 0:
        return "SELL"
    
    else:
        return "HOLD"
