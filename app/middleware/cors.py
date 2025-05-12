"""
CORS configuration for the FastAPI application
"""

import os
import logging
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from app import app

# Configure logging
logger = logging.getLogger(__name__)

def get_allowed_origins() -> List[str]:
    """
    Get allowed origins for CORS configuration.
    
    Returns:
        List[str]: List of allowed origin URLs
    """
    origins_env = os.getenv("ALLOWED_ORIGINS")
    allowed_origins = []
    
    if origins_env:
        allowed_origins.extend([origin.strip() for origin in origins_env.split(",")])
    else:
        # Default origins
        allowed_origins.extend([
            "http://localhost:8000",
            "http://localhost:3000",
            "https://localhost:8000",
            "https://localhost:3000",
        ])
    
    # Add ngrok URL if available
    ngrok_url = os.getenv("NGROK_URL")
    if ngrok_url:
        allowed_origins.append(ngrok_url)
        # Also add the https and http variations
        if ngrok_url.startswith("http://"):
            allowed_origins.append(ngrok_url.replace("http://", "https://"))
        elif ngrok_url.startswith("https://"):
            allowed_origins.append(ngrok_url.replace("https://", "http://"))
    
    # Allow all origins in development if specifically enabled
    if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
        allowed_origins = ["*"]
        
    logger.info(f"Allowed origins for CORS: {allowed_origins}")
    return allowed_origins

ALLOWED_ORIGINS = get_allowed_origins()

ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]

ALLOWED_HEADERS = [
    "Content-Type",
    "Authorization",
    "Accept",
    "Origin",
    "X-Requested-With",
]

def setup_cors():
    """Configure CORS middleware for the FastAPI application"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
        max_age=3600,  # Cache preflight requests for 1 hour
        expose_headers=["Content-Length"],  # Headers that can be exposed to the browser
    ) 