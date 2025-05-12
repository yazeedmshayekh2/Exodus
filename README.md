# Exodus - Multilingual FAQ Chatbot with React Frontend

This project implements a modern FAQ chatbot system with a React frontend and FastAPI backend. It supports multiple languages and uses advanced NLP techniques for semantic search and response generation.

## Features

- **Modern React Frontend** with beautiful UI/UX
- **FastAPI Backend** with CORS and security middleware
- **Multilingual Support** with automatic language detection
- **Advanced NLP Features:**
  - Semantic search using Sentence Transformers
  - Local vector similarity search with Qdrant
  - Integration with Ollama LLM for natural responses
- **Security & Infrastructure:**
  - SSL support with self-signed certificates
  - Comprehensive logging system
  - Docker support for easy deployment
  - Environment-based configuration

## Tech Stack

### Frontend
- React.js
- Modern UI components
- Responsive design

### Backend
- FastAPI
- Sentence Transformers (MPNet-base-v2)
- Qdrant vector database
- Ollama LLM integration
- Langchain for LLM operations

## Requirements

- Python 3.8+
- Node.js and npm (for frontend)
- Ollama server (local or remote)
- OpenSSL (for SSL certificates)

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Exodus
```

2. **Install backend dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies:**
```bash
cd frontend
npm install
```

4. **Environment Setup:**
Create a `.env` file in the root directory with your configuration:
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# SSL Configuration
USE_SSL=true
SSL_CERT_DIR=./ssl

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.1:8b
```

## Running the Application

1. **Start the backend:**
```bash
python main.py
```

2. **Start the frontend development server:**
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: `http://localhost:3000`
- Backend API: `https://localhost:8000`
- API documentation: `https://localhost:8000/docs`

## Project Structure
```
Exodus/
├── frontend/          # React frontend application
├── app/              # FastAPI backend application
├── src/              # Core backend logic
├── assets/           # Static assets
├── ssl/              # SSL certificates
├── requirements.txt  # Python dependencies
└── Dockerfile       # Docker configuration
```

## API Endpoints

- **Chat API**: `POST /chat/`
  ```json
  {
    "query": "Your question here"
  }
  ```
- **System endpoints:**
  - Health check: `GET /health`
  - API documentation: `GET /docs`

## Development

- The project uses Python for the backend and React for the frontend
- All API endpoints are documented using OpenAPI (Swagger)
- Frontend development follows React best practices
- Backend includes comprehensive logging and error handling

## Docker Support

Build and run using Docker:
```bash
docker build -t exodus-chatbot .
docker run -p 8000:8000 exodus-chatbot
```

## Security

- SSL encryption for API communication
- Environment-based configuration
- Secure headers and CORS protection
- Comprehensive error logging

## Troubleshooting

- **SSL Issues**: Delete the `./ssl` directory to regenerate certificates
- **Frontend Build**: Ensure all npm dependencies are installed
- **Backend Errors**: Check the `error.log` file for detailed logs

## License

[MIT License](LICENSE)

## Author

Yazeed Mshayekh and Basel Anaya | AI Engineers