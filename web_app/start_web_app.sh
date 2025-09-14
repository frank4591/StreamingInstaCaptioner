#!/bin/bash
# Start Web-based Streaming Instagram Captioner

echo "🚀 Starting Streaming Instagram Captioner Web App"
echo "=================================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "flockTest" ]]; then
    echo "⚠️  Activating conda environment 'flockTest'..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the web_app directory"
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies for model server
echo "🐍 Installing Python dependencies..."
pip install fastapi uvicorn python-multipart

# Start all services
echo "🎬 Starting all services..."
echo "   - Model Server (GPU 1): http://localhost:8000"
echo "   - Web Server: http://localhost:5000"
echo "   - Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start services in background
npm run model-server &
MODEL_PID=$!

# Wait a bit for model server to start
sleep 10

npm run backend &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 5

npm run frontend &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all services..."
    kill $MODEL_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
