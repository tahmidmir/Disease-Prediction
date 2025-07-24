# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ✅ Set Hugging Face writable cache directory
ENV HF_HOME=/code/.hf_cache
RUN mkdir -p /code/.hf_cache && chmod -R 777 /code/.hf_cache


# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend /code



# Set environment and expose port
ENV PORT=7860
EXPOSE $PORT

# Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
