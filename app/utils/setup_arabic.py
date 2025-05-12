#!/usr/bin/env python3
"""
Setup script for Arabic language processing dependencies
This script installs and configures the required packages for Arabic processing in the chatbot.
"""

import os
import sys
import subprocess
import logging
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("arabic_setup")

def check_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def setup_arabic_support():
    """Setup Arabic language support"""
    required_packages = [
        "transformers",
        "tensorflow",
        "arabert",
        "farasapy",
        "pyarabic"
    ]
    
    success = True
    
    logger.info("Setting up Arabic language support...")
    
    # Check and install required packages
    for package in required_packages:
        if not check_package_installed(package):
            if not install_package(package):
                success = False
        else:
            logger.info(f"{package} is already installed")
    
    # Download AraBERT model if not already downloaded
    try:
        # Test if we can import and use the packages
        from arabert.preprocess import ArabertPreprocessor
        
        # Create a test instance to trigger model download
        logger.info("Testing AraBERT installation and downloading models...")
        for model_name in ["bert-base-arabertv2", "bert-base-arabert"]:
            try:
                ArabertPreprocessor(model_name=model_name)
                logger.info(f"Successfully loaded {model_name} model")
                break
            except Exception as e:
                logger.warning(f"Could not load {model_name} model: {e}")
                success = False
    except Exception as e:
        logger.error(f"Error testing AraBERT: {e}")
        success = False
    
    # Final status
    if success:
        logger.info("Arabic language support setup completed successfully")
    else:
        logger.warning("Arabic language support setup completed with some issues")
    
    return success

if __name__ == "__main__":
    setup_arabic_support() 