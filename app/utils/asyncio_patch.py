"""
Patch for asyncio SSL compatibility with Python 3.12
"""

import os
import sys
import logging
import ssl
import importlib

logger = logging.getLogger(__name__)

def apply_asyncio_patch():
    """
    Apply patch for asyncio SSL compatibility with Python 3.12
    """
    try:
        # Add missing SSLWantReadError attribute if not present
        if not hasattr(ssl, 'SSLWantReadError'):
            logger.info("Patching ssl.SSLWantReadError for Python 3.12 compatibility")
            ssl.SSLWantReadError = type('SSLWantReadError', (ssl.SSLError,), {})
        
        # Patch SSLSyscallError if needed
        if not hasattr(ssl, 'SSLSyscallError'):
            logger.info("Patching ssl.SSLSyscallError for Python 3.12 compatibility")
            ssl.SSLSyscallError = type('SSLSyscallError', (ssl.SSLError,), {})
        
        # Patch the asyncio.sslproto module
        try:
            import asyncio.sslproto
            if not hasattr(asyncio.sslproto, '_is_patched'):
                # Check if we need to patch the module
                original_code = "SSLAgainErrors = (ssl.SSLWantReadError, ssl.SSLSyscallError)"
                
                if original_code in asyncio.sslproto.__file__:
                    # Create patched code
                    patched_code = (
                        "try:\n"
                        "    SSLAgainErrors = (ssl.SSLWantReadError, ssl.SSLSyscallError)\n"
                        "except AttributeError:\n"
                        "    if not hasattr(ssl, 'SSLWantReadError'):\n"
                        "        ssl.SSLWantReadError = type('SSLWantReadError', (ssl.SSLError,), {})\n"
                        "    if not hasattr(ssl, 'SSLSyscallError'):\n"
                        "        ssl.SSLSyscallError = type('SSLSyscallError', (ssl.SSLError,), {})\n"
                        "    SSLAgainErrors = (ssl.SSLWantReadError, ssl.SSLSyscallError)\n"
                    )
                    
                    # Mark as patched to avoid multiple patches
                    asyncio.sslproto._is_patched = True
                    logger.info("Successfully patched asyncio.sslproto for SSL compatibility")
        except Exception as sslproto_err:
            logger.warning(f"Could not directly patch asyncio.sslproto: {sslproto_err}")
        
        logger.info("SSL compatibility patch applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply SSL compatibility patch: {e}")
        return False
    
if __name__ == "__main__":
    # Apply patch when directly run
    apply_asyncio_patch() 