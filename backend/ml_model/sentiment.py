from transformers import pipeline

model = pipeline("sentiment-analysis")

def analyze(text):
    return model(text)
