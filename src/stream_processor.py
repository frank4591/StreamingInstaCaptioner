"""
Stream Processor for Streaming Instagram Captioner

Handles real-time video stream processing and frame extraction.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Callable, Dict, Any, Tuple
from threading import Thread, Event
import queue

class StreamingVideoProcessor:
    """
    Processes video streams in real-time for context extraction.
    
    Handles frame extraction, preprocessing, and callback management
    for streaming applications.
    """
    
    def __init__(
        self,
        frame_callback: Optional[Callable] = None,
        target_fps: int = 10,
        frame_size: Tuple[int, int] = (640, 480),
        enable_preprocessing: bool = True
    ):
        """
        Initialize the stream processor.
        
        Args:
            frame_callback: Callback function for processed frames
            target_fps: Target frames per second
            frame_size: Target frame size (width, height)
            enable_preprocessing: Whether to enable frame preprocessing
        """
        self.frame_callback = frame_callback
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.enable_preprocessing = enable_preprocessing
        
        # Stream state
        self.is_streaming = False
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Threading
        self.capture_thread = None
        self.stop_event = Event()
        
        logging.info(f"StreamingVideoProcessor initialized: {target_fps} FPS, {frame_size}")
    
    def start_webcam_stream(self, camera_index: int = 0) -> bool:
        """Start streaming from webcam."""
        try:
            logging.info(f"Starting webcam stream from camera {camera_index}")
            
            self.capture = cv2.VideoCapture(camera_index)
            
            if not self.capture.isOpened():
                logging.error(f"Failed to open camera {camera_index}")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Start streaming
            self.is_streaming = True
            self.stop_event.clear()
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logging.info("Webcam stream started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting webcam stream: {str(e)}")
            return False
    
    def start_file_stream(self, video_path: str) -> bool:
        """Start streaming from video file."""
        try:
            logging.info(f"Starting file stream from {video_path}")
            
            self.capture = cv2.VideoCapture(video_path)
            
            if not self.capture.isOpened():
                logging.error(f"Failed to open video file {video_path}")
                return False
            
            # Get video properties
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.info(f"Video properties: {fps} FPS, {total_frames} frames")
            
            # Start streaming
            self.is_streaming = True
            self.stop_event.clear()
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logging.info("File stream started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting file stream: {str(e)}")
            return False
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self.target_fps
        last_frame_time = 0
        
        while self.is_streaming and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                # Capture frame
                ret, frame = self.capture.read()
                
                if not ret:
                    if self.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.capture.get(cv2.CAP_PROP_FRAME_COUNT):
                        # End of video file
                        logging.info("End of video file reached")
                        break
                    else:
                        logging.warning("Failed to read frame")
                        continue
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': processed_frame,
                        'original_frame': frame,
                        'timestamp': current_time,
                        'frame_number': self.frame_count
                    })
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait({
                            'frame': processed_frame,
                            'original_frame': frame,
                            'timestamp': current_time,
                            'frame_number': self.frame_count
                        })
                    except queue.Empty:
                        pass
                
                # Update counters
                self.frame_count += 1
                self.fps_counter += 1
                last_frame_time = current_time
                
                # Update FPS counter
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Call frame callback if provided
                if self.frame_callback:
                    try:
                        self.frame_callback(processed_frame, current_time)
                    except Exception as e:
                        logging.warning(f"Frame callback error: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)
        
        logging.info("Capture loop ended")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better processing."""
        if not self.enable_preprocessing:
            return frame
        
        try:
            # Resize frame
            processed = cv2.resize(frame, self.frame_size)
            
            # Apply basic enhancements
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
            
            return processed
            
        except Exception as e:
            logging.warning(f"Frame preprocessing error: {str(e)}")
            return frame
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get the next frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get the latest frame, discarding older ones."""
        latest_frame = None
        
        # Get all available frames
        while not self.frame_queue.empty():
            try:
                latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        return latest_frame
    
    def stop_stream(self):
        """Stop the video stream."""
        logging.info("Stopping video stream...")
        
        self.is_streaming = False
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logging.info("Video stream stopped")
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "is_streaming": self.is_streaming,
            "frame_count": self.frame_count,
            "current_fps": self.current_fps,
            "target_fps": self.target_fps,
            "queue_size": self.frame_queue.qsize(),
            "frame_size": self.frame_size
        }
    
    def set_frame_callback(self, callback: Callable):
        """Set the frame callback function."""
        self.frame_callback = callback
    
    def is_ready(self) -> bool:
        """Check if stream is ready."""
        return self.is_streaming and self.capture is not None and self.capture.isOpened()


