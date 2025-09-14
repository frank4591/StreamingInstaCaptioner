#!/bin/bash
# Start Web-based Streaming Instagram Captioner - Fixed Version

echo "🚀 Starting Streaming Instagram Captioner Web App (Fixed)"
echo "========================================================"

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

# Install Node.js dependencies with proper error handling
echo "📦 Installing Node.js dependencies..."
if ! npm install; then
    echo "❌ Failed to install Node.js dependencies"
    echo "Trying to install dependencies individually..."
    
    # Install dependencies one by one
    npm install express@^4.18.2
    npm install socket.io@^4.7.2
    npm install cors@^2.8.5
    npm install multer@^1.4.4
    npm install axios@^1.5.0
    npm install concurrently@^8.2.2
fi

# Install Python dependencies for model server
echo "🐍 Installing Python dependencies..."
pip install fastapi uvicorn python-multipart

# Check if GPU 1 is available
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs available:"
    nvidia-smi --list-gpus
else
    echo "⚠️  nvidia-smi not found, GPU detection may not work"
fi

# Start all services
echo "🎬 Starting all services..."
echo "   - Model Server (GPU 1): http://localhost:8000"
echo "   - Web Server: http://localhost:5000"
echo "   - Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start services in background
echo "Starting Model Server..."
npm run model-server &
MODEL_PID=$!

# Wait for model server to start
echo "Waiting for model server to initialize..."
sleep 15

echo "Starting Backend Server..."
npm run backend &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

echo "Starting Frontend Server..."
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

# Show status
echo ""
echo "✅ All services started!"
echo "🌐 Open your browser to: http://localhost:3000"
echo ""

# Wait for all background processes
wait


