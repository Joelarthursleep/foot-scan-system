#!/bin/bash

# Enhanced Foot Scan System Startup Script

echo "🚀 Starting Enhanced Foot Scan System..."
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if enhanced features are available
echo "🔍 Checking enhanced AI features..."
python -c "
try:
    import sys
    sys.path.append('src')
    from features.enhanced_medical_analyzer import EnhancedMedicalAnalyzer
    print('✅ Enhanced AI features available!')
except ImportError as e:
    print('⚠️  Enhanced AI features not available:', e)
    print('Basic features will be used.')
"

# Start Streamlit app
echo "🌐 Starting Streamlit dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "============================================="

streamlit run app.py --server.port 8501 --server.address localhost