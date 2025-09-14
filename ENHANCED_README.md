# Enhanced Streaming Instagram Captioner v4.0

## 🎯 Overview

The Enhanced Streaming Instagram Captioner integrates video context extraction functionality from the `VideoContextImageCaptioning` project into a real-time streaming application. This creates a sophisticated system that:

1. **Captures video frames** from a webcam stream
2. **Extracts context** from multiple frames using LFM2-VL models
3. **Generates enhanced Instagram captions** using the video context
4. **Provides real-time feedback** with context consistency metrics

## 🚀 Key Features

### Core Functionality
- 📷 **Real-time camera streaming** with webcam access
- 🎬 **Video context extraction** from the last 10 frames
- 📝 **Context-aware Instagram caption generation**
- 🔄 **Continuous captioning** with adjustable intervals (3-30 seconds)
- 📊 **Live context consistency metrics**

### Video Context Integration
- **Frame Analysis**: Each captured frame is analyzed for visual content
- **Context Aggregation**: Multiple frames are combined to create rich context
- **Consistency Tracking**: Measures how consistent the video context is over time
- **Smart Descriptions**: Generates detailed descriptions for each frame

### Enhanced UI
- **Context Visualization**: Shows the last 10 frames being used for context
- **Real-time Stats**: Processing time, caption count, uptime tracking
- **Context Metrics**: Frame count, consistency scores, aggregated context display
- **History Management**: Caption history with timestamps

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Enhanced Model  │    │ Video Context   │
│                 │    │     Server       │    │   Pipeline      │
│ • Camera Stream │───▶│                  │───▶│                 │
│ • Frame Capture │    │ • LFM2-VL Model  │    │ • Context Extr. │
│ • Context UI    │    │ • Video Context  │    │ • Frame Analysis│
│ • Caption Display│   │ • Caption Gen.   │    │ • Aggregation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
StreamingInstagramCaptioner/
├── web_app/
│   ├── frontend/
│   │   └── enhanced_streaming_captioner.html  # Enhanced UI
│   ├── model_server/
│   │   └── enhanced_model_server.py           # Enhanced backend
│   ├── start_enhanced.sh                      # Startup script
│   └── test_enhanced.py                       # Test script
└── VideoContextImageCaptioning/               # Video context project
    └── src/
        ├── pipeline.py                        # Video context pipeline
        ├── context_extractor.py               # Context extraction
        ├── video_processor.py                 # Video processing
        └── liquid_integration.py              # LFM2-VL integration
```

## 🛠️ Installation & Setup

### Prerequisites
1. **VideoContextImageCaptioning Project**: Must be in the parent directory
2. **LFM2-VL Model**: Located at `/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M`
3. **Conda Environment**: `flockTest` environment activated
4. **Python Dependencies**: All requirements from both projects

### Quick Start

1. **Start the Enhanced Server**:
   ```bash
   cd /home/frank/StreamingInstagramCaptioner/web_app
   ./start_enhanced.sh
   ```

2. **Open the Enhanced UI**:
   ```
   http://localhost:3000/enhanced_streaming_captioner.html
   ```

3. **Test the System**:
   ```bash
   python test_enhanced.py
   ```

## 🎮 Usage Guide

### Basic Workflow

1. **Start Camera**: Click "Start Camera" to begin video streaming
2. **Enable Enhanced Captioning**: Click "Start Enhanced Captioning"
3. **Adjust Interval**: Use the slider to set caption generation frequency (3-30s)
4. **Monitor Context**: Watch the video context panel for frame analysis
5. **View Captions**: See real-time Instagram captions with video context

### Advanced Features

#### Video Context Panel
- **Context Frames**: Visual display of the last 10 frames
- **Aggregated Context**: Combined description from all frames
- **Consistency Score**: How consistent the video context is
- **Frame Count**: Number of frames being used for context

#### Real-time Statistics
- **Total Captions**: Number of captions generated
- **Average Processing Time**: Mean time per caption
- **Uptime**: How long the system has been running
- **Current Interval**: Current caption generation frequency

#### Context Management
- **Clear Context**: Reset video context to start fresh
- **Automatic Aggregation**: Smart combination of frame descriptions
- **Consistency Tracking**: Monitor context quality over time

## 🔧 API Endpoints

### Enhanced Model Server (`http://localhost:8000`)

#### Health Check
```http
GET /health
```
Returns server status and video context availability.

#### Basic Caption Generation
```http
POST /generate_caption
Content-Type: application/json

{
  "image_data": "base64_encoded_image",
  "context": "Video context description",
  "style": "instagram",
  "prompt": "Custom prompt"
}
```

#### Video Context Caption Generation
```http
POST /generate_video_context_caption
Content-Type: application/json

{
  "frames": ["base64_frame1", "base64_frame2", ...],
  "current_frame": "base64_current_frame",
  "style": "instagram",
  "prompt": "Custom prompt"
}
```

#### Context Extraction
```http
POST /extract_context
Content-Type: application/json

{
  "image_data": "base64_encoded_image",
  "style": "descriptive",
  "prompt": "Describe this frame"
}
```

## 🧪 Testing

### Automated Tests
```bash
python test_enhanced.py
```

### Manual Testing
1. Start the enhanced server
2. Open the web interface
3. Test camera access
4. Verify context extraction
5. Check caption generation quality

## 🔍 Troubleshooting

### Common Issues

#### Video Context Not Available
- **Cause**: VideoContextImageCaptioning project not found
- **Solution**: Ensure the project is in the parent directory

#### Camera Access Denied
- **Cause**: Browser permissions or WSL camera issues
- **Solution**: Grant camera permissions and check WSL camera setup

#### Model Loading Errors
- **Cause**: LFM2-VL model path incorrect or missing
- **Solution**: Verify model path and ensure model files exist

#### Context Generation Slow
- **Cause**: Processing multiple frames takes time
- **Solution**: Reduce context frame count or increase interval

### Debug Mode
Enable detailed logging by setting environment variables:
```bash
export LOG_LEVEL=DEBUG
export ENABLE_VIDEO_CONTEXT_DEBUG=true
```

## 📊 Performance Metrics

### Typical Performance
- **Frame Processing**: 0.5-1.0 seconds per frame
- **Context Aggregation**: 0.2-0.5 seconds
- **Caption Generation**: 1.0-2.0 seconds
- **Total Processing**: 2.0-3.5 seconds per caption

### Optimization Tips
- Use fewer context frames for faster processing
- Increase caption interval for better quality
- Monitor GPU memory usage
- Clear context periodically to prevent memory buildup

## 🔮 Future Enhancements

### Planned Features
- **Object Tracking**: Track specific objects across frames
- **Scene Detection**: Automatic scene change detection
- **Emotion Analysis**: Analyze emotional context from frames
- **Multi-language Support**: Support for multiple languages
- **Custom Context Weights**: Adjustable context importance

### Integration Possibilities
- **Mobile App**: React Native or Flutter app
- **Desktop App**: Electron-based desktop application
- **Cloud Deployment**: AWS/Azure deployment with scaling
- **API Integration**: RESTful API for third-party integration

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comprehensive docstrings
- Include type hints where appropriate

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Liquid AI**: For the LFM2-VL models and architecture
- **Hugging Face**: For the transformers library
- **VideoContextImageCaptioning**: For the video context extraction logic
- **OpenCV**: For video processing capabilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test results
3. Check server logs
4. Create an issue with detailed information

---

**Enhanced Streaming Instagram Captioner v4.0** - Real-time video context-aware caption generation powered by LFM2-VL models.
