# 🎥 Streaming Instagram Captioner

A real-time streaming application that uses LFM2-VL models to generate Instagram captions from live video streams. This demonstrates the "constantly on" vision capabilities of Liquid AI's LFM2-VL models for privacy-preserving, on-device processing.

## 🌟 Features

- **Real-time video streaming** from webcam or mobile camera
- **Live context extraction** using LFM2-VL models
- **Continuous Instagram caption generation** based on current scene
- **Privacy-preserving** on-device processing
- **Multiple streaming interfaces** (webcam, mobile, file)
- **Configurable caption styles** and update intervals

## 🏗️ Architecture

```
StreamingInstagramCaptioner/
├── src/
│   ├── stream_processor.py      # Real-time video stream processing
│   ├── context_buffer.py        # Sliding window context management
│   ├── caption_generator.py     # Live caption generation
│   └── model_manager.py         # LFM2-VL model management
├── streaming_apps/
│   ├── webcam_app.py           # Webcam streaming application
│   ├── mobile_app.py           # Mobile camera streaming
│   └── file_stream_app.py      # File-based streaming
├── config/
│   └── streaming_config.yaml   # Configuration settings
└── examples/
    └── demo_streams/           # Sample video streams
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install opencv-python numpy torch transformers pillow
   ```

2. **Run webcam streaming:**
   ```bash
   python streaming_apps/webcam_app.py
   ```

3. **Run mobile streaming:**
   ```bash
   python streaming_apps/mobile_app.py
   ```

## 🎯 Use Cases

- **Live Instagram content creation** with real-time captions
- **Smart security cameras** with contextual descriptions
- **AI tour guides** providing live scene descriptions
- **Accessibility tools** for visually impaired users
- **Live translation** via video context

## 🔧 Configuration

Edit `config/streaming_config.yaml` to customize:
- Model settings and device preferences
- Stream resolution and frame rates
- Context buffer size and update intervals
- Caption styles and generation parameters

## 📱 Mobile Integration

The app supports mobile camera integration through:
- **WebRTC** for browser-based mobile streaming
- **REST API** for mobile app integration
- **Real-time WebSocket** communication

## 🔒 Privacy & Security

- **On-device processing** - no data leaves your device
- **No cloud dependencies** - fully local operation
- **Configurable data retention** - control context buffer size
- **Secure streaming** - encrypted local communication

## 🎨 Caption Styles

- **Instagram-style** with hashtags and emojis
- **Descriptive** for accessibility
- **Minimal** for clean aesthetics
- **Custom** user-defined styles

## 📊 Performance

- **Low latency** real-time processing
- **Efficient memory usage** with sliding window buffers
- **GPU acceleration** when available
- **Configurable quality vs speed** trade-offs

## 🛠️ Development

This application demonstrates LFM2-VL's capabilities for:
- **Always-on vision** processing
- **Real-time context understanding**
- **Privacy-preserving AI** applications
- **Edge computing** with vision models

Perfect for the Liquid AI LFM2-VL challenge showcasing innovative applications of their vision-language models!
