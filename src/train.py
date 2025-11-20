import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.sklearn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Configuration
DATA_PATH = "data/FinancialPhraseBank/Sentences_50Agree.txt" # Adjust based on your unzip location
MODEL_NAME = "distilbert-base-uncased"
experiment_name = "MarketPulse_Sentiment"

def load_data(path):
    # The dataset format is "Sentence@Sentiment"
    try:
        df = pd.read_csv(path, sep='@', names=['text', 'label'], encoding='latin-1')
        return df
    except FileNotFoundError:
        # Fallback if file is in a subdirectory (common with unzipping)
        for root, dirs, files in os.walk("data"):
            if "Sentences_50Agree.txt" in files:
                return pd.read_csv(os.path.join(root, "Sentences_50Agree.txt"), 
                                 sep='@', names=['text', 'label'], encoding='latin-1')
        raise FileNotFoundError("Could not find Sentences_50Agree.txt in data folder")

def get_bert_embeddings(text_list):
    print("⏳ Loading BERT model (this happens once)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    print("🔄 Generating embeddings (this might take a minute)...")
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Take the embedding of the [CLS] token (first token) as the sentence representation
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def train():
    # 1. Start MLflow Run
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        print("🚀 Starting Training Pipeline...")
        
        # 2. Load Data
        df = load_data(DATA_PATH)
        print(f"Dataset loaded: {len(df)} records")
        
        # Use a small subset for testing (remove .head(100) for full training)
        # Keeping it small so it runs fast on your laptop now
        df = df.head(200) 
        
        # 3. Feature Engineering (BERT Embeddings)
        X = get_bert_embeddings(df['text'].tolist())
        y = df['label']
        
        # 4. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Train Model
        params = {"C": 1.0, "solver": "liblinear"}
        clf = LogisticRegression(**params)
        clf.fit(X_train, y_train)
        
        # 6. Evaluate
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"✅ Model Accuracy: {accuracy:.4f}")
        
        # 7. Log Metrics & Params to MLflow
        mlflow.log_params(params)
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_metric("accuracy", accuracy)
        
        # 8. Log the Model itself
        mlflow.sklearn.log_model(clf, "model")
        print("💾 Model saved to MLflow.")

        import joblib  # Add this to imports at the top

# ... (inside train function, at the very end)
        print("💾 Model saved to MLflow.")
        
        # Add this: Save local copy for the API to use
        joblib.dump(clf, "models/sentiment_model.pkl")
        print("✅ Model saved locally to models/sentiment_model.pkl")

if __name__ == "__main__":
    train()