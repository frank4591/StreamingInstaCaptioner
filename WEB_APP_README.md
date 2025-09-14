# 🌐 Web-Based Streaming Instagram Captioner

## 🎯 **Real-Time Web Application with LFM2-VL on GPU 1**

A complete web-based streaming application that uses LFM2-VL models running on GPU 1 for real-time Instagram caption generation from browser camera access.

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│  Node.js Server  │───▶│  Python Model   │
│  (Camera Access)│    │   (Express)      │    │  Server (GPU 1) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTML5 Camera  │    │   WebSocket      │    │   LFM2-VL Model │
│   + Controls    │    │   Communication  │    │   + Context     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🚀 **Quick Start**

### **1. Start the Web Application**
```bash
cd /home/frank/StreamingInstagramCaptioner/web_app
./start_web_app.sh
```

### **2. Access the Application**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Model Server**: http://localhost:8000

### **3. Use the Interface**
1. Click "Start Camera" to access your webcam
2. Choose caption style (Instagram, Trendy, Minimal, Descriptive)
3. Click "Generate Caption" to create captions
4. View real-time context and caption history

---

## 🎮 **Features**

### **Real-Time Camera Access**
- **HTML5 Camera API** - Direct browser camera access
- **No WSL issues** - Works perfectly on WSL/Windows
- **Live video streaming** with real-time processing
- **Frame capture** and context extraction

### **Interactive Controls**
- **Caption Styles**: Instagram, Trendy, Minimal, Descriptive
- **Update Intervals**: 1-10 seconds configurable
- **Real-time Controls**: Start/stop camera, generate captions
- **Context Management**: Clear context, view history

### **LFM2-VL Integration**
- **GPU 1 Deployment** - Uses NVIDIA RTX GPU
- **Real-time Inference** - Low-latency caption generation
- **Context Awareness** - Video context integration
- **Privacy-Preserving** - On-device processing

---

## 🔧 **Technical Details**

### **Frontend (HTML5/JavaScript)**
- **Camera Access**: getUserMedia API
- **Real-time Communication**: Socket.IO
- **Responsive Design**: Mobile-friendly interface
- **Live Updates**: WebSocket streaming

### **Backend (Node.js/Express)**
- **API Server**: RESTful endpoints
- **WebSocket Server**: Real-time communication
- **Image Processing**: Base64 encoding/decoding
- **CORS Support**: Cross-origin requests

### **Model Server (Python/FastAPI)**
- **GPU 1 Deployment**: CUDA_VISIBLE_DEVICES=1
- **LFM2-VL Integration**: Real-time inference
- **Context Buffer**: Sliding window management
- **Caption Generation**: Multiple styles

---

## 📱 **Browser Compatibility**

- **Chrome/Chromium**: ✅ Full support
- **Firefox**: ✅ Full support
- **Safari**: ✅ Full support
- **Edge**: ✅ Full support
- **Mobile Browsers**: ✅ Responsive design

---

## 🎯 **Use Cases**

### **1. Live Instagram Content Creation**
- Real-time caption generation for live streams
- Multiple caption styles for different content
- Automatic hashtag generation
- Context-aware captions

### **2. Social Media Management**
- Batch caption generation for posts
- Style consistency across content
- Real-time content analysis
- Automated content creation

### **3. Accessibility Tools**
- Live scene descriptions
- Real-time environmental awareness
- Contextual information display
- Multi-modal content understanding

### **4. Content Creation Tools**
- Video caption generation
- Real-time content analysis
- Style-based caption generation
- Context-aware descriptions

---

## 🔒 **Privacy & Security**

- **On-Device Processing** - No data leaves your machine
- **No Cloud Dependencies** - Fully local operation
- **Secure Communication** - Local network only
- **Data Control** - Complete control over your data

---

## 🚀 **Performance**

- **GPU Acceleration** - Uses NVIDIA RTX GPU
- **Low Latency** - Real-time processing
- **Efficient Memory** - Optimized for streaming
- **Scalable Architecture** - Handles multiple users

---

## 🛠️ **Development**

### **File Structure**
```
web_app/
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── styles.css          # CSS styling
│   └── app.js             # JavaScript application
├── backend/
│   └── server.js           # Node.js server
├── model_server/
│   └── model_server.py     # Python FastAPI server
├── package.json            # Node.js dependencies
└── start_web_app.sh        # Startup script
```

### **API Endpoints**
- `GET /` - Frontend page
- `GET /api/status` - Server status
- `POST /api/caption` - Generate caption
- `POST /api/context` - Extract context
- `GET /health` - Health check

### **WebSocket Events**
- `camera_frame` - Send camera frame
- `generate_caption` - Request caption
- `context_update` - Context update
- `caption_result` - Caption result

---

## 🎉 **Ready to Use!**

The web-based Streaming Instagram Captioner is now ready to demonstrate LFM2-VL's capabilities with:

✅ **Real-time camera access** through browser  
✅ **GPU 1 deployment** for optimal performance  
✅ **Interactive web interface** with full controls  
✅ **Privacy-preserving** on-device processing  
✅ **Multiple caption styles** and customization  
✅ **Context-aware** caption generation  

**Perfect for the Liquid AI LFM2-VL challenge! 🚀**


