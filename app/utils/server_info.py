"""
Server information utilities
"""

import os
import socket
import logging
import pyngrok.ngrok as ngrok
import shutil
from app.middleware.cors import ALLOWED_ORIGINS

# Configure logging
logger = logging.getLogger(__name__)

def get_local_ip():
    """
    Get the local IP address for LAN access.
    
    Returns:
        str: Local IP address
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't actually establish a connection
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

def get_server_info():
    """
    Get information about the server, including connection details.
    
    Returns:
        dict: Server information
    """
    try:
        use_ssl = os.getenv('USE_SSL', 'true').lower() == 'true'
        use_ngrok = os.getenv('USE_NGROK', 'true').lower() == 'true'
        
        # Get current ngrok tunnels if available
        ngrok_url = None
        ngrok_status = "disabled"
        
        if use_ngrok:
            ngrok_status = "enabled but not connected"
            try:
                tunnels = ngrok.get_tunnels()
                if tunnels:
                    ngrok_url = tunnels[0].public_url
                    ngrok_status = "connected"
                else:
                    # Check for existence of ngrok in PATH
                    ngrok_installed = shutil.which("ngrok") is not None
                    if not ngrok_installed:
                        ngrok_status = "not installed"
                    else:
                        # Check if auth token is set
                        if not os.getenv("NGROK_AUTH_TOKEN"):
                            ngrok_status = "missing auth token"
                        else:
                            ngrok_status = "connection failed"
            except Exception as e:
                logger.warning(f"Error getting ngrok tunnels: {e}")
                ngrok_status = f"error: {str(e)}"
        
        local_ip = get_local_ip()
        port = os.getenv('PORT', 8000)
        
        # Construct URLs
        local_url = f"{'https' if use_ssl else 'http'}://{local_ip}:{port}"
        localhost_url = f"{'https' if use_ssl else 'http'}://localhost:{port}"
                
        return {
            "server": {
                "local_ip": local_ip,
                "localhost_url": localhost_url,
                "local_lan_url": local_url,
                "ssl_enabled": use_ssl,
                "ngrok_status": ngrok_status,
                "ngrok_url": ngrok_url,
            },
            "api_docs": {
                "swagger_ui": f"{localhost_url}/api/docs",
                "redoc": f"{localhost_url}/api/redoc",
                "ngrok_swagger_ui": f"{ngrok_url}/api/docs" if ngrok_url else None,
                "ngrok_redoc": f"{ngrok_url}/api/redoc" if ngrok_url else None,
            },
            "environment": {
                "host": os.getenv('HOST', '0.0.0.0'),
                "port": port,
                "allowed_origins": ALLOWED_ORIGINS,
            },
            "troubleshooting": {
                "ngrok_tips": [
                    "Ngrok free tier allows only 1 simultaneous tunnel",
                    "Check for running ngrok processes with 'ps aux | grep ngrok'",
                    "Kill existing ngrok processes with 'pkill -f ngrok'",
                    "Visit https://dashboard.ngrok.com/agents to manage active sessions",
                    "Set USE_NGROK=false in .env to disable ngrok"
                ] if use_ngrok and not ngrok_url else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        raise 