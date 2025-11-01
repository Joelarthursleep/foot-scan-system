#!/usr/bin/env python3
"""
Start script for the Enhanced Foot Scan System UI
Launches the Streamlit web interface
"""

import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_virtual_environment():
    """Check if virtual environment is activated"""
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        logger.warning("Virtual environment not detected. Please activate it first:")
        logger.warning("source venv/bin/activate")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import tensorflow
        import numpy
        import pandas
        logger.info("Core dependencies verified")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

def start_ui():
    """Start the Streamlit UI"""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        logger.error("app.py not found!")
        return False

    logger.info("Starting Enhanced Foot Scan System UI...")
    logger.info("Access the UI at: http://localhost:8501")
    logger.info("Press Ctrl+C to stop the server")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start UI: {e}")
        return False

    return True

def main():
    """Main function"""
    print("=" * 60)
    print("ENHANCED FOOT SCAN SYSTEM - STARTUP")
    print("=" * 60)

    if not check_virtual_environment():
        return 1

    if not check_dependencies():
        return 1

    if not start_ui():
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())