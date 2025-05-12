# Bilingual FAQ Chatbot Server

This project implements a FastAPI-based server for a bilingual FAQ chatbot that supports English and Arabic languages. It uses semantic search with sentence transformers and Qdrant vector database for finding relevant FAQ matches from a SQL Server database.

## Features

- **FastAPI server** with CORS and security middleware.
- **Sentence transformer** (MPNet-base-v2) for semantic embeddings.
- **Local in-memory Qdrant** for vector similarity search.
- **Ollama Llama 3.1 8B LLM** as the brain of the chatbot.
- **SSL support** with self-signed certificates for secure communication.
- **Ngrok integration** for exposing the server to the internet.
- **Bilingual support** for English and Arabic languages.

## Requirements

- Python 3.8+
- SQL Server database
- Ollama server running locally or remotely
- Ngrok (optional but recommended for public access)
- OpenSSL (for SSL certificate generation)

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd bilingual-faq-chatbot
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Copy the example `.env.example` file and modify it according to your requirements:

```bash
cp .env.example .env
```

Edit the `.env` file to set your configuration:

```
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# SSL Configuration
USE_SSL=true
SSL_CERT_DIR=./ssl

# Ngrok Configuration
USE_NGROK=true
NGROK_AUTH_TOKEN=your_ngrok_auth_token  # Get from https://dashboard.ngrok.com/

# Database Configuration
DB_SERVER=your_db_server
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.1:8b
```

## Running the Application

Start the application with:

```bash
python main.py
```

The application will:

1. Generate a self-signed SSL certificate (if enabled and doesn't exist)
2. Start an ngrok tunnel (if enabled)
3. Start the FastAPI server

You can access:
- Local URL: `https://localhost:8000` (or http if SSL is disabled)
- Ngrok URL: Check the console output for the public URL
- API documentation: `<your-url>/api/docs` or `<your-url>/api/redoc`
- Server info: `<your-url>/debug/server-info`

## Ngrok Setup (For Public Access)

1. **Create an account** at [ngrok.com](https://ngrok.com/)
2. **Get your auth token** from the dashboard
3. **Add the token to your .env file**:
   ```
   NGROK_AUTH_TOKEN=your_token_here
   ```

## SSL Certificate

The application generates a self-signed SSL certificate automatically in the `./ssl` directory. For production use, consider replacing these with proper certificates from a trusted Certificate Authority.

## Using the API

The chatbot API can be accessed through:

- **Chat endpoint**: `POST /chat/`
  ```json
  {
    "query": "Your question here"
  }
  ```

- **Debug endpoints**:
  - Server info: `GET /debug/server-info`
  - FAQ search: `GET /debug/faq/{query}`
  - FAQ count: `GET /debug/faqs`

## Browser Access

The chat interface is available at the root URL (`/`).

## Security Considerations

1. The self-signed certificate will cause browser warnings. For production, use a proper CA-issued certificate.
2. The ngrok URL changes each time you restart the application unless you have a paid ngrok account.
3. The application includes security headers and CORS protection.

## Troubleshooting

- **Certificate issues**: Delete the `./ssl` directory to regenerate certificates
- **Ngrok errors**: Check your auth token and ensure ngrok is properly installed
- **Database connection issues**: Verify your database credentials and network connectivity

## License

[MIT License](LICENSE)

## Author

Basel Anaya | AI Engineer