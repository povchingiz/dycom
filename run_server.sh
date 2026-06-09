#!/bin/bash
# FaceSim Demo Server Startup Script

set -e

echo "🚀 Starting FaceSim Demo Server..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it to set your password."
    echo "   Then run this script again."
    exit 0
fi

# Load environment variables
source .env

# Check if Python environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected."
    echo "   Activate your Python 3.12 environment first:"
    echo "   source .venv312/bin/activate"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Clean up old sessions (older than 1 week)
echo "🧹 Cleaning up old sessions..."
find server/sessions -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true

# Start server
echo "🌐 Starting server on http://0.0.0.0:8000"
echo "   Press Ctrl+C to stop"
echo ""
cd server
python main.py
