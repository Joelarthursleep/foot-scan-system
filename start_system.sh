#!/bin/bash

# Foot Scan to Custom Last System - Startup Script

echo "🦶 Starting Foot Scan to Custom Last System..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p data/lasts
mkdir -p data/models
mkdir -p output

# Start the API server
echo "Starting API server on http://localhost:8000"
echo "Web interface will be available at http://localhost:8000/web"

# Serve web files with Python's simple HTTP server in background
python -m http.server 8080 --directory web &
WEB_PID=$!

# Start FastAPI server
cd src/api
python main.py &
API_PID=$!

echo ""
echo "✅ System started successfully!"
echo ""
echo "📊 API Documentation: http://localhost:8000/docs"
echo "🌐 Web Interface: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the system"

# Wait for interrupt
trap "kill $WEB_PID $API_PID; exit" INT
wait