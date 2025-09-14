#!/bin/bash
# Start Pipeline Integration Streaming Instagram Captioner

echo "🚀 Starting Pipeline Integration Streaming Instagram Captioner v4.0"
echo "=================================================================="
echo "Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction using existing VideoContextImageCaptioning pipeline"
echo "  📝 Enhanced Instagram caption generation"
echo "  🔗 Direct integration with working pipeline"
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

# Start pipeline integration server
echo "🤖 Starting Pipeline Integration Server..."
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/pipeline_integration_server.py &
MODEL_PID=$!

# Wait for server to start
echo "⏳ Waiting for pipeline integration server to initialize..."
sleep 30

# Test server
echo "🧪 Testing pipeline integration server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Pipeline integration server is working!"
    
    # Check server response
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "📊 Server status: $HEALTH_RESPONSE"
    
    # Check if video context is available
    if echo "$HEALTH_RESPONSE" | grep -q '"video_context_available": true'; then
        echo "✅ Video context pipeline integration is available!"
    else
        echo "⚠️  Video context pipeline integration not available (fallback mode)"
    fi
else
    echo "❌ Pipeline integration server failed to start"
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
echo "Pipeline Integration Features:"
echo "  📷 Real-time camera streaming"
echo "  🎬 Video context extraction using existing pipeline"
echo "  📝 Context-aware Instagram caption generation"
echo "  📊 Live context consistency metrics"
echo "  🔄 Automatic context aggregation"
echo "  🎯 Direct integration with VideoContextImageCaptioning"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
