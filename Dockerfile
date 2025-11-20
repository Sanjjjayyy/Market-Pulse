# 1. Base Image (Lightweight Python)
FROM python:3.9-slim

# 2. Set work directory
WORKDIR /app

# 3. Copy requirements first (This caches the installation step)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code
COPY . .

# 6. Expose the port for FastAPI
EXPOSE 8000

# 7. Command to run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]