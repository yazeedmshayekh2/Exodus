"""
Bilingual FAQ Chatbot Server

This project implements a FastAPI-based server for a bilingual FAQ chatbot that supports
English and Arabic languages. It uses semantic search with sentence transformers and
Qdrant vector database for finding relevant FAQ matches from a SQL Server database.

Key components:
- FastAPI server with CORS and security middleware.
- Sentence transformer MPNet-base-v2 for semantic embeddings.
- Local in-memory Qdrant for vector similarity search.
- Ollama Llama 3.1 8B LLM as the brain of the chatbot.
- Markdown formatting for responses.

Built by: Yazeed Mshayekh | AI Engineer
Date: 6/1/2025
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv

# Configure logging 
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import app and configure middleware
from app import app
from app.middleware.security import log_requests, add_security_headers
from app.middleware.cors import setup_cors
from app.utils.ssl import generate_self_signed_cert
from app.utils.tunnel import setup_ngrok

# Import API routes - ensure imports are executed, not just specified
from app.api import chat, debug

# Setup middleware
app.middleware("http")(log_requests)
app.middleware("http")(add_security_headers)
setup_cors()

if __name__ == "__main__":
    try:
        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load environment variables
        port = int(os.getenv('PORT', 8000))
        host = os.getenv('HOST', '0.0.0.0')
        # Force SSL to be disabled for testing
        use_ssl = False  # Hardcoding for testing
        use_ngrok = False  # Hardcoding for testing
        
        logger.info(f"Starting server on {host}:{port} with SSL={use_ssl}...")
        
        ssl_context = None
        cert_file = None
        key_file = None
        
        if use_ssl:
            # Generate or use existing SSL certificate
            cert_file, key_file = generate_self_signed_cert()
            logger.info(f"SSL enabled with certificate: {cert_file}")
        else:
            logger.info("SSL disabled for testing")
        
        # Development server setup
        logger.info("Starting development server with Uvicorn...")
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,  # Disable reload for testing
            workers=1,  # Single worker
            log_level=os.getenv('LOG_LEVEL', 'info').lower(),
            ssl_certfile=cert_file if use_ssl else None,
            ssl_keyfile=key_file if use_ssl else None
        )
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
    