# Smart Streaming Instagram Captioner

A real-time video context-aware Instagram caption generator that uses the VideoContextImageCaptioning pipeline with LFM2-VL models.

## 🚀 Features

- **Real-time Video Context Analysis**: Extract context from video frames using advanced AI models
- **Smart Instagram Caption Generation**: Generate context-aware captions for images
- **Live Camera Integration**: Capture frames directly from your camera
- **Image Upload Support**: Upload images from your device
- **Video Context Integration**: Use video context to enhance image captions
- **Quality Metrics**: Track analysis quality and consistency scores

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Web browser with camera access
- Conda environment (flockTest)

## 🛠️ Installation

1. **Activate the conda environment:**
   ```bash
   conda activate flockTest
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure VideoContextImageCaptioning is available:**
   - The system expects the VideoContextImageCaptioning project at `/home/frank/VideoContextImageCaptioning`
   - Make sure the LFM2-VL model is available at `/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M`

## 🚀 Quick Start

### 1. Start the Video Context Server

```bash
cd /home/frank/StreamingInstagramCaptioner/web_app
python model_server/video_context_server.py
```

The server will start on `http://localhost:8000` and show:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start the Web Frontend

In a new terminal:

```bash
cd /home/frank/StreamingInstagramCaptioner/web_app/frontend
python -m http.server 3000
```

The frontend will be available at `http://localhost:3000/smart_streaming_captioner.html`

## 🎯 Usage Workflow

### Option 1: With Video Context (Recommended)

1. **Open the frontend** in your browser: `http://localhost:3000/smart_streaming_captioner.html`

2. **Start the camera** by clicking "Start Camera"

3. **Collect video context:**
   - Click "Start Context Collection" to begin capturing frames
   - The system will automatically extract and analyze frames
   - You'll see "Frame Descriptions" and "Aggregated Context" populated

4. **Capture or upload your final image:**
   - **Option A:** Click "Capture Current Frame" to use the current video frame
   - **Option B:** Click "Upload Image" to select an image file

5. **Generate Instagram caption:**
   - Click "Generate Instagram Caption"
   - The system will use video context + your image to create a smart caption

### Option 2: Direct Image Captioning

1. **Open the frontend** in your browser
2. **Upload an image** directly using "Upload Image" button
3. **Generate caption** - works without video context

## 🔧 API Endpoints

### Video Context Server (Port 8000)

- `GET /health` - Server health check
- `POST /analyze_frames` - Analyze video frames for context
- `POST /generate_video_context_caption` - Generate Instagram caption with video context

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Analyze frames
curl -X POST http://localhost:8000/analyze_frames \
  -H "Content-Type: application/json" \
  -d '{"frames": ["data:image/jpeg;base64,..."]}'
```

## 📊 Features Explained

### Video Context Analysis
- **Frame Extraction**: Automatically extracts key frames from video
- **AI Description**: Uses LFM2-VL model to describe each frame
- **Context Aggregation**: Combines frame descriptions into coherent context
- **Quality Metrics**: Tracks analysis quality and consistency

### Smart Caption Generation
- **Context Integration**: Uses video context to enhance image captions
- **Instagram Optimization**: Generates captions optimized for social media
- **Real-time Processing**: Fast caption generation with processing time tracking
- **History Tracking**: Maintains caption history with timestamps

## 🐛 Troubleshooting

### Common Issues

1. **"Video context components not available"**
   - Check if VideoContextImageCaptioning project is at the correct path
   - Verify LFM2-VL model is available
   - Check server logs for detailed error messages

2. **"Cannot read properties of undefined"**
   - This JavaScript error has been fixed in the latest version
   - Refresh the browser page if you see this error

3. **Camera not working**
   - Ensure browser has camera permissions
   - Try refreshing the page
   - Check browser console for errors

4. **Server won't start**
   - Make sure conda environment is activated: `conda activate flockTest`
   - Check if port 8000 is available
   - Verify all dependencies are installed

### Server Logs

Check server logs for detailed error information:
```bash
# View server logs
tail -f /path/to/server/logs
```

## 🔄 Development

### File Structure

```
web_app/
├── model_server/
│   └── video_context_server.py    # Main API server
├── frontend/
│   └── smart_streaming_captioner.html  # Web interface
├── backend/
│   └── server.js                  # Node.js backend (alternative)
└── README.md                      # This file
```

### Key Components

- **Video Context Server**: FastAPI server handling video analysis and caption generation
- **Frontend**: HTML/JavaScript interface with camera integration
- **VideoContextImageCaptioning**: AI pipeline for video context extraction
- **LFM2-VL Model**: Advanced vision-language model for image understanding

## 📈 Performance

- **Processing Time**: Typically 2-5 seconds per caption
- **Memory Usage**: ~2GB GPU memory for LFM2-VL model
- **Concurrent Users**: Supports multiple simultaneous users
- **Quality Score**: 25-95% depending on image quality and context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the StreamingInstagramCaptioner system.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review server logs for error details
3. Ensure all prerequisites are met
4. Verify the VideoContextImageCaptioning pipeline is working

---

**Happy Captioning! 🎬📸✨**
