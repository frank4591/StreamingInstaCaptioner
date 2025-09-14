#!/bin/bash
# Start Continuous Instagram Captioner - Real-time Streaming Version

echo "🚀 Starting Continuous Instagram Captioner"
echo "==========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "flockTest" ]]; then
    echo "⚠️  Activating conda environment 'flockTest'..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
fi

# Check if we're in the right directory
if [ ! -f "frontend/continuous_captioner.html" ]; then
    echo "❌ Please run this script from the web_app directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    if [ ! -z "$MODEL_PID" ]; then
        kill $MODEL_PID 2>/dev/null
    fi
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null
    fi
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start model server
echo "🤖 Starting Model Server..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/model_server_fixed.py &
MODEL_PID=$!

# Wait for model server to start
echo "⏳ Waiting for model server to initialize..."
sleep 15

# Test model server
echo "🧪 Testing model server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Model server is working!"
else
    echo "❌ Model server failed to start"
    cleanup
    exit 1
fi

# Start web server
echo "🌐 Starting web server..."
cd frontend
python -m http.server 3000 &
WEB_PID=$!

echo ""
echo "✅ All services started!"
echo "🌐 Open your browser to: http://localhost:3000/continuous_captioner.html"
echo ""
echo "Features:"
echo "  📷 Real-time camera streaming"
echo "  ⏱️  Adjustable caption interval (2-30 seconds)"
echo "  📝 Continuous Instagram caption generation"
echo "  📊 Live statistics and caption history"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
