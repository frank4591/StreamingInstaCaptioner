// Streaming Instagram Captioner - Frontend JavaScript

class StreamingCaptionerApp {
    constructor() {
        this.socket = null;
        this.camera = null;
        this.stream = null;
        this.isStreaming = false;
        this.captionHistory = [];
        this.frameCount = 0;
        this.captionCount = 0;
        this.startTime = Date.now();

        this.initializeElements();
        this.initializeSocket();
        this.initializeEventListeners();
        this.checkServerStatus();
    }

    initializeElements() {
        // Camera elements
        this.video = document.getElementById('camera-video');
        this.canvas = document.getElementById('camera-canvas');
        this.ctx = this.canvas.getContext('2d');

        // Control elements
        this.startCameraBtn = document.getElementById('start-camera');
        this.stopCameraBtn = document.getElementById('stop-camera');
        this.captureFrameBtn = document.getElementById('capture-frame');
        this.generateCaptionBtn = document.getElementById('generate-caption');
        this.clearContextBtn = document.getElementById('clear-context');
        this.captionStyleSelect = document.getElementById('caption-style');
        this.updateIntervalSlider = document.getElementById('update-interval');
        this.intervalValueSpan = document.getElementById('interval-value');

        // Display elements
        this.currentCaptionDiv = document.getElementById('current-caption');
        this.videoContextDiv = document.getElementById('video-context');
        this.captionHistoryDiv = document.getElementById('caption-history');
        this.captionConfidenceSpan = document.getElementById('caption-confidence');
        this.captionTimeSpan = document.getElementById('caption-time');
        this.contextFramesSpan = document.getElementById('context-frames');
        this.contextConsistencySpan = document.getElementById('context-consistency');

        // Status elements
        this.modelStatusDot = document.getElementById('model-status');
        this.cameraStatusDot = document.getElementById('camera-status');
        this.connectionStatusDot = document.getElementById('connection-status');

        // Stats elements
        this.totalCaptionsSpan = document.getElementById('total-captions');
        this.totalFramesSpan = document.getElementById('total-frames');
        this.uptimeSpan = document.getElementById('uptime');

        // Modal elements
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.errorModal = document.getElementById('error-modal');
        this.errorMessage = document.getElementById('error-message');
        this.closeModal = document.querySelector('.close');
    }

    initializeSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });

        this.socket.on('context_update', (data) => {
            this.updateVideoContext(data);
        });

        this.socket.on('caption_result', (data) => {
            this.updateCaption(data);
        });

        this.socket.on('style_changed', (data) => {
            console.log('Style changed to:', data.style);
        });

        this.socket.on('error', (data) => {
            this.showError(data.message);
        });
    }

    initializeEventListeners() {
        // Camera controls
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.captureFrameBtn.addEventListener('click', () => this.captureFrame());

        // Caption controls
        this.generateCaptionBtn.addEventListener('click', () => this.generateCaption());
        this.clearContextBtn.addEventListener('click', () => this.clearContext());

        // Style change
        this.captionStyleSelect.addEventListener('change', (e) => {
            this.changeStyle(e.target.value);
        });

        // Update interval
        this.updateIntervalSlider.addEventListener('input', (e) => {
            this.intervalValueSpan.textContent = e.target.value + 's';
        });

        // Modal close
        this.closeModal.addEventListener('click', () => {
            this.errorModal.style.display = 'none';
        });

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === this.errorModal) {
                this.errorModal.style.display = 'none';
            }
        });

        // Update stats periodically
        setInterval(() => this.updateStats(), 1000);
    }

    async checkServerStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.status === 'running') {
                this.updateModelStatus(true);
                console.log('Server is running');
            } else {
                this.updateModelStatus(false);
            }
        } catch (error) {
            console.error('Server status check failed:', error);
            this.updateModelStatus(false);
        }
    }

    async startCamera() {
        try {
            this.showLoading('Starting camera...');

            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.isStreaming = true;
                this.updateCameraStatus(true);
                this.startCameraBtn.disabled = true;
                this.stopCameraBtn.disabled = false;
                this.captureFrameBtn.disabled = false;
                this.generateCaptionBtn.disabled = false;
                this.hideLoading();

                // Start frame processing
                this.startFrameProcessing();
            };

        } catch (error) {
            console.error('Camera access failed:', error);
            this.showError('Camera access failed: ' + error.message);
            this.hideLoading();
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.video.srcObject = null;
        this.isStreaming = false;
        this.updateCameraStatus(false);
        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.captureFrameBtn.disabled = true;
        this.generateCaptionBtn.disabled = true;
    }

    startFrameProcessing() {
        if (!this.isStreaming) return;

        // Process frame every 2 seconds for context extraction
        setInterval(() => {
            if (this.isStreaming) {
                this.processFrameForContext();
            }
        }, 2000);
    }

    processFrameForContext() {
        if (!this.isStreaming) return;

        try {
            // Capture frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Send to server for context extraction
            this.socket.emit('camera_frame', {
                imageData: imageData,
                timestamp: Date.now()
            });

            this.frameCount++;

        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }

    captureFrame() {
        if (!this.isStreaming) return;

        try {
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Generate caption for this frame
            this.generateCaptionForFrame(imageData);

        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }

    generateCaptionForFrame(imageData) {
        this.showLoading('Generating caption...');

        this.socket.emit('generate_caption', {
            imageData: imageData,
            context: this.getCurrentContext(),
            style: this.captionStyleSelect.value,
            prompt: `Create an ${this.captionStyleSelect.value}-style caption for this image.`
        });
    }

    generateCaption() {
        if (!this.isStreaming) {
            this.showError('Please start the camera first');
            return;
        }

        this.captureFrame();
    }

    changeStyle(style) {
        this.socket.emit('change_style', { style: style });
    }

    clearContext() {
        // Clear context buffer
        this.videoContextDiv.textContent = 'No context available';
        this.contextFramesSpan.textContent = 'Frames: 0';
        this.contextConsistencySpan.textContent = 'Consistency: --';

        // Reset frame count
        this.frameCount = 0;
    }

    updateVideoContext(data) {
        this.videoContextDiv.textContent = data.context || 'No context available';
        this.contextFramesSpan.textContent = `Frames: ${data.frame_count || 0}`;
        this.contextConsistencySpan.textContent = `Consistency: ${(data.consistency || 0).toFixed(3)}`;
    }

    updateCaption(data) {
        // Use instagram_caption if available, otherwise fallback to caption
        const caption = data.instagram_caption || data.caption || 'No caption generated';
        this.currentCaptionDiv.textContent = caption;
        this.captionConfidenceSpan.textContent = `Confidence: ${(data.confidence || 0).toFixed(3)}`;
        this.captionTimeSpan.textContent = `Time: ${(data.processing_time || 0).toFixed(2)}s`;

        // Add to history
        this.addToHistory({ ...data, caption: caption });

        this.captionCount++;
        this.hideLoading();
    }

    addToHistory(data) {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <strong>${data.caption}</strong><br>
            <small>Style: ${data.style || 'instagram'} | Confidence: ${(data.confidence || 0).toFixed(3)} | Time: ${new Date().toLocaleTimeString()}</small>
        `;

        this.captionHistoryDiv.insertBefore(historyItem, this.captionHistoryDiv.firstChild);

        // Keep only last 10 items
        const items = this.captionHistoryDiv.querySelectorAll('.history-item');
        if (items.length > 10) {
            this.captionHistoryDiv.removeChild(items[items.length - 1]);
        }
    }

    getCurrentContext() {
        return this.videoContextDiv.textContent;
    }

    updateModelStatus(isOnline) {
        this.modelStatusDot.className = `status-dot ${isOnline ? 'active' : 'error'}`;
    }

    updateCameraStatus(isActive) {
        this.cameraStatusDot.className = `status-dot ${isActive ? 'active' : 'error'}`;
    }

    updateConnectionStatus(isConnected) {
        this.connectionStatusDot.className = `status-dot ${isConnected ? 'active' : 'error'}`;
    }

    updateStats() {
        this.totalCaptionsSpan.textContent = `Captions: ${this.captionCount}`;
        this.totalFramesSpan.textContent = `Frames: ${this.frameCount}`;

        const uptime = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(uptime / 60);
        const seconds = uptime % 60;
        this.uptimeSpan.textContent = `Uptime: ${minutes}m ${seconds}s`;
    }

    showLoading(message = 'Processing...') {
        this.loadingOverlay.style.display = 'flex';
        this.loadingOverlay.querySelector('p').textContent = message;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'block';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StreamingCaptionerApp();
});
