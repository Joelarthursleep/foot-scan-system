# Enhanced Foot Scan System - Setup Instructions

## Quick Setup Guide

### 1. System Requirements
- Python 3.9 or higher
- 8GB+ RAM
- 50GB+ storage space
- macOS, Linux, or Windows

### 2. Installation Steps

```bash
# Navigate to the project directory
cd foot-scan-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Start the system
python start_system.py
```

### 3. Access the Web Interface
- Open your browser to: `http://localhost:8501`
- The system will initialize automatically
- Navigate through the tabs to configure and use the system

## System Overview

This system processes 3D foot scans to:
- Detect medical conditions (bunions, flat feet, etc.)
- Compare against healthy baselines
- Recommend custom shoe lasts
- Generate 3D printing modifications

## Next Steps

1. **Configure APIs** (when available):
   - Go to "API Configuration" tab
   - Enter Volumental API credentials
   - Test connectivity

2. **Upload Sample Data**:
   - Use "Process Scan" tab to upload OBJ/JSON files
   - Or use demo mode to see system capabilities

3. **Import Data** (when provided):
   - Import healthy foot baseline files
   - Load last library specifications
   - Configure manufacturing parameters

## Getting Started Without APIs

The system includes demo/mock data to explore functionality:
- Sample healthy foot profiles
- Basic last library entries
- Mock segmentation for testing

Run `python run_enhanced_system.py` to see a command-line demo.

## Support

If you encounter issues:
1. Ensure virtual environment is activated
2. Check that all dependencies installed successfully
3. Review terminal output for error messages
4. Verify Python version is 3.9+

The system is ready for API integration once credentials are provided.