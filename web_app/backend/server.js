const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Configuration
const MODEL_SERVER_URL = 'http://localhost:8000';
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// Store active connections
const activeConnections = new Map();

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.get('/api/status', (req, res) => {
    res.json({
        status: 'running',
        modelServer: MODEL_SERVER_URL,
        activeConnections: activeConnections.size,
        timestamp: new Date().toISOString()
    });
});

app.post('/api/caption', async (req, res) => {
    try {
        const { imageData, context, style, prompt } = req.body;

        // Forward to model server
        const response = await axios.post(`${MODEL_SERVER_URL}/generate_caption`, {
            image_data: imageData,
            context: context,
            style: style,
            prompt: prompt
        });ac

        res.json(response.data);
    } catch (error) {
        console.error('Error generating caption:', error.message);
        res.status(500).json({ error: 'Failed to generate caption' });
    }
});

app.post('/api/context', async (req, res) => {
    try {
        const { imageData, description } = req.body;

        // Forward to model server
        const response = await axios.post(`${MODEL_SERVER_URL}/extract_context`, {
            image_data: imageData,
            description: description
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error extracting context:', error.message);
        res.status(500).json({ error: 'Failed to extract context' });
    }
});

// WebSocket connection handling
io.on('connection', (socket) => {
    console.log(`Client connected: ${socket.id}`);

    activeConnections.set(socket.id, {
        socket,
        connectedAt: new Date(),
        lastActivity: new Date()
    });

    // Handle camera stream
    socket.on('camera_frame', async (data) => {
        try {
            const { imageData, timestamp } = data;

            // Update activity
            const connection = activeConnections.get(socket.id);
            if (connection) {
                connection.lastActivity = new Date();
            }

            // Extract context from frame
            const contextResponse = await axios.post(`${MODEL_SERVER_URL}/extract_context`, {
                image_data: imageData,
                description: "Describe this video frame in detail, focusing on visual elements, objects, and scene composition."
            });

            // Send context update
            socket.emit('context_update', {
                context: contextResponse.data.context,
                confidence: contextResponse.data.confidence,
                timestamp: timestamp
            });

        } catch (error) {
            console.error('Error processing camera frame:', error.message);
            socket.emit('error', { message: 'Failed to process frame' });
        }
    });

    // Handle caption generation request
    socket.on('generate_caption', async (data) => {
        try {
            const { imageData, context, style, prompt } = data;

            // Generate caption
            const captionResponse = await axios.post(`${MODEL_SERVER_URL}/generate_caption`, {
                image_data: imageData,
                context: context,
                style: style,
                prompt: prompt
            });

            // Send caption result
            socket.emit('caption_result', {
                caption: captionResponse.data.caption,
                raw_output: captionResponse.data.raw_output,
                confidence: captionResponse.data.confidence,
                processing_time: captionResponse.data.processing_time,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error('Error generating caption:', error.message);
            socket.emit('error', { message: 'Failed to generate caption' });
        }
    });

    // Handle style change
    socket.on('change_style', (data) => {
        console.log(`Style changed to: ${data.style}`);
        socket.emit('style_changed', { style: data.style });
    });

    // Handle disconnect
    socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
        activeConnections.delete(socket.id);
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        activeConnections: activeConnections.size
    });
});

// Start server
server.listen(PORT, () => {
    console.log(`🚀 Web server running on port ${PORT}`);
    console.log(`📱 Frontend: http://localhost:${PORT}`);
    console.log(`🔗 Model server: ${MODEL_SERVER_URL}`);
    console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});


