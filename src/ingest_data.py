import os
import requests
import zipfile
import io

# Stable URL from Hugging Face Datasets (Direct binary link)
DATA_URL = "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
DATA_FOLDER = "data"

def download_data():
    print("🚀 Starting Data Ingestion...")
    
    # 1. Create data folder
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    # 2. Download using requests (more robust than urllib)
    print(f"Downloading from {DATA_URL}...")
    response = requests.get(DATA_URL)
    
    if response.status_code == 200:
        # 3. Unzip directly from memory (no temp file needed)
        print("Extracting data...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DATA_FOLDER)
        print(f"✅ Data successfully stored in {DATA_FOLDER}/")
    else:
        print(f"❌ Failed to download. Status Code: {response.status_code}")

if __name__ == "__main__":
    download_data()