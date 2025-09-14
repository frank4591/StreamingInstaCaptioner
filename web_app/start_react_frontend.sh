#!/bin/bash

# Smart Streaming Instagram Captioner - React Frontend Startup Script
# This script starts both the backend server and React frontend

echo "🚀 Smart Streaming Instagram Captioner - React Frontend"
echo "======================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi

print_status "Checking system requirements..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"
FRONTEND_DIR="$SCRIPT_DIR/frontend-react"

print_status "Backend directory: $BACKEND_DIR"
print_status "Frontend directory: $FRONTEND_DIR"

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    print_error "Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

# Function to start backend server
start_backend() {
    print_status "Starting backend server..."
    cd "$BACKEND_DIR"
    
    # Activate conda environment
    print_status "Activating conda environment: flockTest"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flockTest
    
    # Check if video context server is already running
    if pgrep -f "video_context_server.py" > /dev/null; then
        print_warning "Backend server is already running"
        return 0
    fi
    
    # Start the server in background
    nohup python model_server/video_context_server.py > backend.log 2>&1 &
    BACKEND_PID=$!
    
    # Wait a moment for server to start
    sleep 3
    
    # Check if server started successfully
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend server started (PID: $BACKEND_PID)"
        echo $BACKEND_PID > backend.pid
    else
        print_error "Failed to start backend server"
        return 1
    fi
}

# Function to start frontend
start_frontend() {
    print_status "Starting React frontend..."
    cd "$FRONTEND_DIR"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_status "Installing dependencies..."
        npm install
        if [ $? -ne 0 ]; then
            print_error "Failed to install dependencies"
            return 1
        fi
    fi
    
    # Start the development server
    print_status "Starting Vite development server..."
    npm run dev
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    
    # Kill backend server if we started it
    if [ -f "$BACKEND_DIR/backend.pid" ]; then
        BACKEND_PID=$(cat "$BACKEND_DIR/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_status "Stopping backend server (PID: $BACKEND_PID)"
            kill $BACKEND_PID
            rm -f "$BACKEND_DIR/backend.pid"
        fi
    fi
    
    # Kill any remaining video context server processes
    pkill -f "video_context_server.py" 2>/dev/null || true
    
    print_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend
start_backend
if [ $? -ne 0 ]; then
    print_error "Failed to start backend. Exiting."
    exit 1
fi

# Wait a bit for backend to fully start
print_status "Waiting for backend to initialize..."
sleep 5

# Check if backend is responding
print_status "Checking backend health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend is healthy and responding"
else
    print_warning "Backend may not be fully ready yet"
fi

echo ""
print_success "Backend server is running on http://localhost:8000"
print_success "Starting React frontend on http://localhost:3000"
echo ""
print_status "Press Ctrl+C to stop both servers"
echo ""

# Start frontend (this will block)
start_frontend
