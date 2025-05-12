"""
API package initialization
"""

# Import API modules to ensure routes are registered
import app.api.chat
import app.api.debug

# Add main frontend route
import os
from fastapi.responses import FileResponse
from app import app, logger

@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html file"""
    static_dir = os.path.join(os.getcwd(), "frontend/build")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to the FAQ Chatbot API"}

@app.get("/favicon.svg")
async def serve_favicon():
    """Serve the favicon.svg file"""
    static_dir = os.path.join(os.getcwd(), "frontend/build")
    favicon_path = os.path.join(static_dir, "favicon.svg")
    logger.info(f"Serving favicon from {favicon_path}")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"message": "Favicon not found"}
