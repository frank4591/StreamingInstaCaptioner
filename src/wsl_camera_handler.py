"""
WSL Camera Handler for Streaming Instagram Captioner

Handles camera access on WSL (Windows Subsystem for Linux) systems.
"""

import cv2
import logging
import subprocess
import os
from typing import Optional, List

class WSLCameraHandler:
    """
    Handles camera access on WSL systems.
    
    WSL doesn't have direct access to Windows cameras, so we provide
    alternative methods for camera access and testing.
    """
    
    def __init__(self):
        self.is_wsl = self._detect_wsl()
        self.available_cameras = []
        
        logging.info(f"WSL Camera Handler initialized (WSL detected: {self.is_wsl})")
    
    def _detect_wsl(self) -> bool:
        """Detect if running on WSL."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def get_available_cameras(self) -> List[int]:
        """Get list of available camera indices."""
        cameras = []
        
        if self.is_wsl:
            logging.info("WSL detected - checking for Windows camera access...")
            
            # Try to access Windows camera through WSLg
            for i in range(5):  # Check first 5 camera indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cameras.append(i)
                            logging.info(f"Camera {i} available")
                        cap.release()
                except Exception as e:
                    logging.debug(f"Camera {i} not available: {e}")
        else:
            # Standard Linux camera detection
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cameras.append(i)
                            logging.info(f"Camera {i} available")
                        cap.release()
                except Exception as e:
                    logging.debug(f"Camera {i} not available: {e}")
        
        self.available_cameras = cameras
        return cameras
    
    def create_test_video_source(self) -> Optional[cv2.VideoCapture]:
        """Create a test video source for WSL systems."""
        if not self.available_cameras:
            logging.warning("No cameras available, creating test video source...")
            
            # Try to create a test pattern video source
            try:
                # Create a simple test pattern
                cap = cv2.VideoCapture(0)  # Try default camera first
                if cap.isOpened():
                    return cap
            except:
                pass
            
            # If no camera available, we'll use a file-based approach
            return None
        
        return None
    
    def get_camera_info(self, camera_index: int) -> dict:
        """Get information about a specific camera."""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return {"available": False, "error": "Camera not accessible"}
            
            info = {
                "available": True,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "backend": cap.getBackendName(),
                "is_wsl": self.is_wsl
            }
            
            cap.release()
            return info
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def suggest_alternatives(self) -> List[str]:
        """Suggest alternatives for WSL users."""
        alternatives = []
        
        if self.is_wsl:
            alternatives.extend([
                "1. Use Windows Camera app to test camera access",
                "2. Try running from Windows PowerShell with WSL integration",
                "3. Use file stream mode with video files instead",
                "4. Install Windows camera drivers in WSL",
                "5. Use WSLg for GUI applications (Windows 11)",
                "6. Try different camera indices (0, 1, 2, etc.)"
            ])
        else:
            alternatives.extend([
                "1. Check camera permissions",
                "2. Try different camera indices",
                "3. Install camera drivers",
                "4. Use file stream mode as alternative"
            ])
        
        return alternatives
    
    def create_file_stream_demo(self) -> str:
        """Create a demo video file for testing."""
        try:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))
            
            # Generate test frames
            for i in range(100):  # 5 seconds at 20 FPS
                # Create a test pattern frame
                frame = self._create_test_frame(i)
                out.write(frame)
            
            out.release()
            
            logging.info("Created test video: test_video.mp4")
            return "test_video.mp4"
            
        except Exception as e:
            logging.error(f"Error creating test video: {e}")
            return None
    
    def _create_test_frame(self, frame_number: int) -> 'np.ndarray':
        """Create a test pattern frame."""
        import numpy as np
        
        # Create a colorful test pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving elements
        center_x = 320 + int(100 * np.sin(frame_number * 0.1))
        center_y = 240 + int(50 * np.cos(frame_number * 0.1))
        
        # Draw moving circles
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y), 30, (255, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
        
        # Add text
        text = f"Test Frame {frame_number}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = f"Time: {frame_number * 0.05:.1f}s"
        cv2.putText(frame, timestamp, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


