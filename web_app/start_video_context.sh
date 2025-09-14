#!/bin/bash
# Start Video Context Server using actual VideoContextImageCaptioning pipeline

echo "🚀 Starting Video Context Server v1.0"
echo "====================================="
echo "Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction using VideoContextImageCaptioning pipeline"
echo "  📝 Instagram caption generation with proper video context"
echo "  🔗 Direct integration with VideoContextImageCaptioning project"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "flockTest" ]]; then
    echo "⚠️  Activating conda environment 'flockTest'..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
fi

# Check if we're in the right directory
if [ ! -f "frontend/smart_streaming_captioner.html" ]; then
    echo "❌ Please run this script from the web_app directory"
    exit 1
fi

# Check if VideoContextImageCaptioning project exists
if [ ! -d "/home/frank/VideoContextImageCaptioning" ]; then
    echo "❌ VideoContextImageCaptioning project not found!"
    echo "   Please ensure the VideoContextImageCaptioning project is at /home/frank/VideoContextImageCaptioning"
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

# Start video context server
echo "🤖 Starting Video Context Server..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/video_context_server.py &
MODEL_PID=$!

# Wait for server to start
echo "⏳ Waiting for video context server to initialize..."
sleep 35

# Test server
echo "🧪 Testing video context server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Video context server is working!"
    
    # Check server response
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "📊 Server status: $HEALTH_RESPONSE"
    
    # Check if video context is available
    if echo "$HEALTH_RESPONSE" | grep -q '"video_context_available": true'; then
        echo "✅ VideoContextImageCaptioning pipeline integration is available!"
    else
        echo "⚠️  VideoContextImageCaptioning pipeline integration not available (fallback mode)"
    fi
else
    echo "❌ Video context server failed to start"
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
echo "🌐 Open your browser to: http://localhost:3000/smart_streaming_captioner.html"
echo ""
echo "Video Context Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction using VideoContextImageCaptioning pipeline"
echo "  📝 Instagram caption generation with proper video context"
echo "  📊 Live context quality metrics"
echo "  🔄 Automatic context aggregation"
echo "  🎯 Proper frame analysis (not Instagram captions)"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
