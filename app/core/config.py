"""
Central configuration module for the application
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Server configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
USE_SSL = os.getenv('USE_SSL', 'true').lower() == 'true'
USE_NGROK = os.getenv('USE_NGROK', 'true').lower() == 'true'

# Database configuration
DB_SERVER = os.getenv('DB_SERVER', '192.168.3.120')
DB_NAME = os.getenv('DB_NAME', 'agencyDB_Live')
DB_USER = os.getenv('DB_USER', 'sa')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'P@ssw0rdSQL')

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
# MODEL_NAME options: 'llama3.1:8b', 'Qwen/Qwen2.5-7B-Instruct-AWQ', 'yazeed-mshayekh/Exodus-Arabic-Model'
MODEL_NAME = os.getenv('MODEL_NAME', 'llama3.1:8b')

# CORS configuration
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true"

# Ngrok configuration
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

# Similarity thresholds for FAQ matching
SIMILARITY_THRESHOLDS = {
    'ar': float(os.getenv('SIMILARITY_THRESHOLD_AR', '0.3')),
    'en': float(os.getenv('SIMILARITY_THRESHOLD_EN', '0.4'))
} 