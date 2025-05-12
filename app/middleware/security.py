"""
Security middleware for FastAPI
"""

import os
import logging
from fastapi import Request

# Configure logging
logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    """Log all incoming HTTP requests."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Basic security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Determine if we're using HTTPS (either directly or through ngrok)
    use_ssl = os.getenv('USE_SSL', 'true').lower() == 'true'
    using_ngrok = os.getenv('USE_NGROK', 'false').lower() == 'true'
    
    # Add HSTS header if using HTTPS
    if use_ssl or using_ngrok:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Add Content-Security-Policy header
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
    
    # Add Referrer-Policy header
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response 