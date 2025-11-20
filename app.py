from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. Define the Request Schema (What data do we expect?)
class StockNews(BaseModel):
    text: str

app = FastAPI(title="MarketPulse API", description="Financial Sentiment Analysis")

# Global variables to hold the model in memory
model_resources = {}

@app.on_event("startup")
def load_models():
    """
    Load the heavy models (BERT + Classifier) only once when the server starts.
    """
    print("⏳ Loading BERT and Classifier...")
    # Load BERT (Use the same model name as training)
    model_name = "distilbert-base-uncased"
    model_resources["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    model_resources["bert"] = AutoModel.from_pretrained(model_name)
    
    # Load your trained classifier
    try:
        model_resources["classifier"] = joblib.load("models/sentiment_model.pkl")
        print("✅ Models loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: Model file not found. Did you run train.py?")

def get_bert_embedding(text):
    """Helper function to convert text to vector"""
    tokenizer = model_resources["tokenizer"]
    bert_model = model_resources["bert"]
    
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Extract [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

@app.post("/predict")
def predict_sentiment(news: StockNews):
    """
    Endpoint to predict market sentiment.
    """
    # 1. Convert text to numbers (Embedding)
    embedding = get_bert_embedding([news.text])
    
    # 2. Predict using the classifier
    classifier = model_resources["classifier"]
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]
    
    # Map prediction to label (Assuming order: 0=Negative, 1=Neutral, 2=Positive)
    # Note: Verify your label mapping from training! 
    # Usually alphabetic: negative, neutral, positive
    labels = ["negative", "neutral", "positive"]
    sentiment = str(prediction) # Fallback
    
    # Simple logic to map prediction to string
    # If your data was text labels, sklearn preserves them.
    
    return {
        "sentiment": prediction, 
        "confidence": float(max(probabilities))
    }

# Run this with: uvicorn app:app --reload