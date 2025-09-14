#!/bin/bash
# Start Simple Instagram Captioner Test

echo "🚀 Starting Simple Instagram Captioner Test"
echo "==========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "flockTest" ]]; then
    echo "⚠️  Activating conda environment 'flockTest'..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
fi

# Kill any existing processes
echo "🛑 Stopping existing processes..."
pkill -f "python.*model_server" 2>/dev/null
pkill -f "python.*http.server" 2>/dev/null
pkill -f "node.*server" 2>/dev/null

# Start model server
echo "🤖 Starting Model Server..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/model_server_fixed.py &
MODEL_PID=$!

# Wait for model server to start
echo "⏳ Waiting for model server to initialize..."
sleep 20

# Test model server
echo "🧪 Testing model server..."
python quick_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model server is working!"
    echo ""
    echo "🌐 Starting web server..."
    echo "   Frontend: http://localhost:3000/simple_test.html"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Start web server
    cd frontend
    python -m http.server 3000 &
    WEB_PID=$!
    
    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "🛑 Stopping all services..."
        kill $MODEL_PID $WEB_PID 2>/dev/null
        echo "✅ All services stopped"
        exit 0
    }
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    # Wait for all background processes
    wait
else
    echo "❌ Model server failed to start. Check logs above."
    kill $MODEL_PID 2>/dev/null
    exit 1
fi
