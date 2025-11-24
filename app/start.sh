#!/bin/bash
# Quick start script for FastAPI Violence Detection System

echo "🚀 Starting Violence Detection System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.dependencies_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.dependencies_installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🎯 Configuration:"
echo "   You need to set model checkpoint paths before starting."
echo "   Options:"
echo "   1. Copy .env.example to .env and edit it"
echo "   2. Set environment variables manually"
echo "   3. Edit core/config.py directly"
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "✅ Found .env file"
else
    echo "⚠️  No .env file found. You can:"
    echo "   cp .env.example .env"
    echo "   Then edit .env with your checkpoint paths"
    echo ""
fi

echo "🌐 Starting FastAPI server..."
echo "   Live Monitor: http://localhost:8000/live"
echo "   Offline Analyzer: http://localhost:8000/offline"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python main.py
