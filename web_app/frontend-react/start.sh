#!/bin/bash

# Start React Frontend for Smart Streaming Instagram Captioner

echo "🚀 Starting Smart Streaming Instagram Captioner - React Frontend"
echo "================================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if backend server is running
echo "🔍 Checking backend server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend server is running"
else
    echo "⚠️  Backend server is not running. Please start the video context server first."
    echo "   Run: python model_server/video_context_server.py"
    echo ""
    echo "   The frontend will still start, but some features may not work."
fi

echo ""
echo "🎬 Starting React development server..."
echo "   Frontend will be available at: http://localhost:3000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
