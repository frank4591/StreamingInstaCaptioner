#!/bin/bash
# Start Simple Enhanced Streaming Instagram Captioner

echo "🚀 Starting Simple Enhanced Streaming Instagram Captioner v4.0"
echo "============================================================="
echo "Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Basic video context extraction from multiple frames"
echo "  📝 Enhanced Instagram caption generation"
echo "  🔧 Simplified version without complex dependencies"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "flockTest" ]]; then
    echo "⚠️  Activating conda environment 'flockTest'..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
fi

# Check if we're in the right directory
if [ ! -f "frontend/enhanced_streaming_captioner.html" ]; then
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

# Start simple enhanced model server
echo "🤖 Starting Simple Enhanced Model Server..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/simple_enhanced_server.py &
MODEL_PID=$!

# Wait for model server to start
echo "⏳ Waiting for simple enhanced model server to initialize..."
sleep 25

# Test model server
echo "🧪 Testing simple enhanced model server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Simple enhanced model server is working!"
    
    # Check server response
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "📊 Server status: $HEALTH_RESPONSE"
else
    echo "❌ Simple enhanced model server failed to start"
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
echo "🌐 Open your browser to: http://localhost:3000/enhanced_streaming_captioner.html"
echo ""
echo "Simple Enhanced Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Basic video context extraction (last 10 frames)"
echo "  📝 Context-aware Instagram caption generation"
echo "  📊 Live context consistency metrics"
echo "  🔄 Automatic context aggregation"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
