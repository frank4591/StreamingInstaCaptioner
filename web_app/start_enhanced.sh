#!/bin/bash
# Start Enhanced Streaming Instagram Captioner with Video Context Integration

echo "🚀 Starting Enhanced Streaming Instagram Captioner v4.0"
echo "====================================================="
echo "Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction from multiple frames"
echo "  📝 Enhanced Instagram caption generation"
echo "  🔗 Integration with VideoContextImageCaptioning project"
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

# Check if VideoContextImageCaptioning project exists
if [ ! -d "../VideoContextImageCaptioning" ]; then
    echo "❌ VideoContextImageCaptioning project not found!"
    echo "   Please ensure the VideoContextImageCaptioning project is in the parent directory"
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

# Start enhanced model server
echo "🤖 Starting Enhanced Model Server with Video Context..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/enhanced_model_server.py &
MODEL_PID=$!

# Wait for model server to start
echo "⏳ Waiting for enhanced model server to initialize..."
sleep 20

# Test model server
echo "🧪 Testing enhanced model server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Enhanced model server is working!"
    
    # Check if video context is available
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    if echo "$HEALTH_RESPONSE" | grep -q '"video_context_available": true'; then
        echo "✅ Video context integration is available!"
    else
        echo "⚠️  Video context integration not available (fallback mode)"
    fi
else
    echo "❌ Enhanced model server failed to start"
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
echo "Enhanced Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction (last 10 frames)"
echo "  📝 Context-aware Instagram caption generation"
echo "  📊 Live context consistency metrics"
echo "  🔄 Automatic context aggregation"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
