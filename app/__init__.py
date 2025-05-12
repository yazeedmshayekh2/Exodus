"""
FastAPI Application Package Initialization
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app with additional configuration
app = FastAPI(
    title="FAQ Chatbot API",
    description="Production-ready FAQ Chatbot API with multilingual support",
    version="2.0.0",
    docs_url="/api/docs",  # Secure Swagger UI location
    redoc_url="/api/redoc"  # Secure ReDoc location
)

# Mount templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory for the frontend
static_dir = os.path.join(os.getcwd(), "frontend/build")
if os.path.exists(static_dir):
    # Mount at root AFTER API routes are registered (will be done in main.py)
    # This will allow the API routes to take precedence over static files
    logger.info(f"Static directory found at {static_dir}")
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")
    
    # We'll handle index.html separately in a route function to make sure API routes work
