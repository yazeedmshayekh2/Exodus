# Use Ubuntu as base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=http://localhost:11434
ENV MODEL_NAME=llama3.1:8b
ENV EMBEDDINGS_CACHE_FILE=/app/embeddings_cache.npz

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install SQL Server ODBC Driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for embeddings cache
RUN mkdir -p /app/data && chmod 777 /app/data

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama service\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
until curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    sleep 1\n\
done\n\
\n\
# Pull the model\n\
echo "Pulling Llama model..."\n\
ollama pull llama3.1:8b\n\
\n\
# Start the Python application\n\
echo "Starting the application..."\n\
python3 main.py\n\
' > /app/start.sh \
    && chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the embeddings cache file location
ENV EMBEDDINGS_CACHE_FILE=/app/data/embeddings_cache.npz

# Command to run the application
CMD ["/app/start.sh"] 