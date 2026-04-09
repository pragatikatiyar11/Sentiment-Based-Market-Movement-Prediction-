from transformers import pipeline

model = pipeline("sentiment-analysis")

def analyze(text):
    result = model(text)[0]
    return result
