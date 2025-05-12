"""
SSL certificate utilities
"""

import os
import logging
import subprocess
import ssl

# Backport SSLWantReadError for Python 3.12 compatibility
if not hasattr(ssl, 'SSLWantReadError'):
    class SSLWantReadError(ssl.SSLError):
        pass
    ssl.SSLWantReadError = SSLWantReadError

# Configure logging
logger = logging.getLogger(__name__)

def generate_self_signed_cert(cert_dir="./ssl"):
    """
    Generate a self-signed SSL certificate if it doesn't exist already.
    
    Args:
        cert_dir (str): Directory to store the SSL certificate and key
        
    Returns:
        tuple: Paths to the certificate and key files
    """
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(cert_dir):
            os.makedirs(cert_dir)
            
        cert_file = os.path.join(cert_dir, "server.crt")
        key_file = os.path.join(cert_dir, "server.key")
        
        # Check if certificate already exists
        if os.path.exists(cert_file) and os.path.exists(key_file):
            logger.info(f"Using existing SSL certificate at {cert_file}")
            return cert_file, key_file
        
        # Generate a self-signed certificate using OpenSSL
        logger.info("Generating self-signed SSL certificate...")
        
        # Generate a private key
        subprocess.run([
            'openssl', 'genrsa',
            '-out', key_file,
            '2048'
        ], check=True)
        
        # Generate a certificate signing request (CSR)
        subprocess.run([
            'openssl', 'req',
            '-new',
            '-key', key_file,
            '-out', os.path.join(cert_dir, 'server.csr'),
            '-subj', '/C=US/ST=State/L=City/O=Organization/OU=Department/CN=localhost'
        ], check=True)
        
        # Generate a self-signed certificate
        subprocess.run([
            'openssl', 'x509',
            '-req',
            '-days', '365',
            '-in', os.path.join(cert_dir, 'server.csr'),
            '-signkey', key_file,
            '-out', cert_file
        ], check=True)
        
        logger.info(f"Successfully generated SSL certificate at {cert_file}")
        return cert_file, key_file
    
    except Exception as e:
        logger.error(f"Error generating SSL certificate: {e}")
        raise

def create_ssl_context(cert_file, key_file):
    """
    Create an SSL context for HTTPS server.
    
    Args:
        cert_file (str): Path to SSL certificate file
        key_file (str): Path to SSL key file
        
    Returns:
        ssl.SSLContext: Configured SSL context
    """
    # Create SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(cert_file, key_file)
    
    return ssl_context 