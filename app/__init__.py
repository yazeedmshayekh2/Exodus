"""
FastAPI Application Package Initialization
"""

import logging
import os
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Apply SSL compatibility patch before importing FastAPI
try:
    from app.utils.asyncio_patch import apply_asyncio_patch
    if apply_asyncio_patch():
        logger.info("Successfully applied SSL compatibility patch")
    else:
        logger.warning("Failed to apply SSL compatibility patch, there may be issues with SSL/asyncio")
except Exception as e:
    logger.warning(f"Could not import and apply SSL compatibility patch: {e}")

# Now import FastAPI and other dependencies
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# Setup cross-origin resource sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Arabic language support if enabled
if os.getenv("ENABLE_ARABIC", "true").lower() == "true":
    try:
        logger.info("Setting up Arabic language support...")
        from app.utils.setup_arabic import setup_arabic_support
        
        # Setup in a non-blocking way
        import threading
        setup_thread = threading.Thread(target=setup_arabic_support)
        setup_thread.daemon = True  # Allow the app to exit even if thread is running
        setup_thread.start()
        
        logger.info("Arabic language support setup started in background")
    except Exception as e:
        logger.error(f"Failed to initialize Arabic support: {e}")
        logger.info("Continuing without enhanced Arabic support")
else:
    logger.info("Arabic language support disabled by environment variable")
