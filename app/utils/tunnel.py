"""
Ngrok tunnel utilities for exposing local server to the internet
"""

import os
import logging
import shutil
import subprocess
import time
import pyngrok.ngrok as ngrok

# Configure logging
logger = logging.getLogger(__name__)

def setup_ngrok(port):
    """
    Set up an ngrok tunnel to the specified port.
    
    Args:
        port (int): The local port to expose
        
    Returns:
        str: The public ngrok URL or None if setup fails
    """
    try:
        # Check if ngrok is installed
        if not shutil.which("ngrok"):
            logger.warning("ngrok executable not found in PATH. Make sure ngrok is installed.")
            return None
        
        # Check for existing tunnels first and try to use them
        try:
            existing_tunnels = ngrok.get_tunnels()
            if existing_tunnels:
                for tunnel in existing_tunnels:
                    if str(port) in tunnel.public_url or f":{port}" in tunnel.config['addr']:
                        logger.info(f"Using existing ngrok tunnel: {tunnel.public_url}")
                        # Store the URL in environment for CORS configuration
                        os.environ["NGROK_URL"] = tunnel.public_url
                        return tunnel.public_url
                logger.info("Found existing ngrok tunnels, but none for our port. Will try to create a new one.")
        except Exception as e:
            logger.warning(f"Error checking existing tunnels: {e}")
        
        # Get ngrok auth token from environment variable
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if ngrok_auth_token:
            ngrok.set_auth_token(ngrok_auth_token)
            logger.info("Ngrok auth token configured.")
        else:
            logger.warning("NGROK_AUTH_TOKEN not set. Ngrok may not work properly without authentication.")
        
        # Try to create an HTTPS tunnel
        logger.info(f"Starting ngrok tunnel to port {port}...")
        
        # Kill existing ngrok processes if we're having issues with limits
        try:
            # This is a more aggressive approach if we're having auth issues
            subprocess.run(["pkill", "-f", "ngrok"], check=False)
            logger.info("Killed existing ngrok processes")
            # Wait a moment for processes to fully terminate
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Could not kill existing ngrok processes: {e}")
        
        # Try to connect with a more robust approach
        try:
            https_tunnel = ngrok.connect(port, "http")
            public_url = https_tunnel.public_url
            
            # Store the URL in environment for CORS configuration
            os.environ["NGROK_URL"] = public_url
            
            logger.info(f"Ngrok tunnel established: {public_url}")
            return public_url
        except Exception as e:
            if "ERR_NGROK_108" in str(e):
                logger.warning("Ngrok session limit reached. You can only have one ngrok session at a time with the free plan.")
                logger.warning("Please check if you have another ngrok tunnel running elsewhere.")
                logger.warning("You can stop other sessions at https://dashboard.ngrok.com/agents")
                logger.warning("Continuing without ngrok integration...")
            else:
                logger.warning(f"Error connecting to ngrok: {e}")
            return None
    
    except Exception as e:
        logger.warning(f"Error setting up ngrok: {e}")
        logger.warning("Continuing without ngrok integration...")
        return None 