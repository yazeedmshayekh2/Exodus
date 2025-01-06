# Bilingual FAQ Chatbot

A production-ready FAQ chatbot that supports English and Arabic languages, using semantic search and the Llama 3.1 8B model.

## Features

- Bilingual support (English and Arabic)
- Semantic search using sentence transformers
- Vector similarity search with Qdrant
- Markdown formatting for responses
- Caching of embeddings for faster startup
- Configurable via environment variables
- Production-ready with Gunicorn and Uvicorn
- CORS and security headers
- Comprehensive logging

## Prerequisites

- Docker
- At least 16GB RAM (recommended)
- 50GB free disk space

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create your environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and run with Docker:
```bash
docker build -t faq-chatbot .
docker run -p 8000:8000 --env-file .env faq-chatbot
```

The chatbot will be available at http://localhost:8000

## Environment Variables

See `.env.example` for all available configuration options.

Key variables:
- `DB_SERVER`: SQL Server address
- `DB_NAME`: Database name
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `OLLAMA_BASE_URL`: Ollama API endpoint
- `MODEL_NAME`: Llama model to use
- `EMBEDDINGS_CACHE_FILE`: Location to store embeddings cache

## API Endpoints

- `GET /`: Chat interface
- `POST /chat/`: Chat API endpoint
- `GET /api/docs`: Swagger documentation
- `GET /api/redoc`: ReDoc documentation

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run in development mode:
```bash
python main.py
```

## Production Deployment

The Docker container is configured for production use with:
- Gunicorn for process management
- Multiple workers
- Error logging
- Security headers
- CORS protection

## Caching

The chatbot caches FAQ embeddings to improve startup time:
- First run: Processes FAQs and creates cache
- Subsequent runs: Loads from cache
- Cache location configurable via `EMBEDDINGS_CACHE_FILE`

## Troubleshooting

1. If the chatbot fails to start:
   - Check database connectivity
   - Verify Ollama is running
   - Ensure sufficient memory for the model

2. If responses are slow:
   - Adjust number of workers
   - Check database connection
   - Verify Ollama performance

3. If embeddings cache isn't working:
   - Check write permissions
   - Verify cache file path
   - Check disk space

## License

[Your License]

## Author

Basel Anaya | AI Engineer