FROM python:3.10-slim

# Install system dependencies TensorFlow might need
RUN apt-get update && apt-get install -y \
    liblzma-dev libx11-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY . /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]







