#!/usr/bin/env python3
"""
Webcam Streaming Instagram Captioner

Real-time Instagram caption generation from webcam feed using LFM2-VL models.
"""

import sys
import os
import cv2
import numpy as np
import logging
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_manager import StreamingModelManager
from context_buffer import StreamingContextBuffer
from stream_processor import StreamingVideoProcessor
from caption_generator import StreamingCaptionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebcamInstagramCaptioner:
    """
    Main application for webcam-based Instagram caption generation.
    """
    
    def __init__(
        self,
        model_path: str,
        camera_index: int = 0,
        caption_style: str = "instagram",
        update_interval: float = 3.0
    ):
        """
        Initialize the webcam captioner.
        
        Args:
            model_path: Path to LFM2-VL model
            camera_index: Camera index (0 for default webcam)
            caption_style: Style of captions to generate
            update_interval: How often to generate new captions (seconds)
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.caption_style = caption_style
        self.update_interval = update_interval
        
        # Initialize components
        self.model_manager = None
        self.context_buffer = None
        self.stream_processor = None
        self.caption_generator = None
        
        # Application state
        self.is_running = False
        self.current_caption = ""
        self.caption_display_time = 0
        self.caption_duration = 5.0  # Show caption for 5 seconds
        
        # Performance tracking
        self.frame_count = 0
        self.caption_count = 0
        self.start_time = 0
        
        logger.info(f"WebcamInstagramCaptioner initialized: camera {camera_index}, style {caption_style}")
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing components...")
            
            # Initialize model manager
            self.model_manager = StreamingModelManager(
                model_path=self.model_path,
                device="cuda",
                enable_warmup=True
            )
            
            if not self.model_manager.load_model():
                logger.error("Failed to load model")
                return False
            
            # Initialize context buffer
            self.context_buffer = StreamingContextBuffer(
                max_frames=15,
                context_window_seconds=30.0,
                update_interval=2.0,
                min_context_frames=3
            )
            
            # Initialize stream processor
            self.stream_processor = StreamingVideoProcessor(
                frame_callback=self._on_frame_received,
                target_fps=10,
                frame_size=(640, 480),
                enable_preprocessing=True
            )
            
            # Initialize caption generator
            self.caption_generator = StreamingCaptionGenerator(
                model_manager=self.model_manager,
                context_buffer=self.context_buffer,
                caption_style=self.caption_style,
                update_interval=self.update_interval,
                caption_callback=self._on_caption_generated
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def _on_frame_received(self, frame: np.ndarray, timestamp: float):
        """Handle received frame for context extraction."""
        try:
            # Generate frame description for context
            result = self.model_manager.generate_caption(
                image=frame,
                text_prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition.",
                temperature=0.1,
                max_new_tokens=100
            )
            
            if result.get("error"):
                logger.warning(f"Frame description error: {result['error']}")
                return
            
            # Add frame to context buffer
            self.context_buffer.add_frame(
                description=result["caption"],
                features=None,  # We're not using features for now
                confidence=result["confidence"],
                timestamp=timestamp
            )
            
            self.frame_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing frame: {str(e)}")
    
    def _on_caption_generated(self, caption_result: dict):
        """Handle generated caption."""
        try:
            self.current_caption = caption_result["caption"]
            self.caption_display_time = time.time()
            self.caption_count += 1
            
            logger.info(f"New caption generated: {self.current_caption[:50]}...")
            
        except Exception as e:
            logger.warning(f"Error handling caption: {str(e)}")
    
    def run(self):
        """Run the main application loop."""
        if not self.initialize():
            logger.error("Failed to initialize application")
            return
        
        logger.info("Starting webcam streaming...")
        
        # Start webcam stream
        if not self.stream_processor.start_webcam_stream(self.camera_index):
            logger.error("Failed to start webcam stream")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Main display loop
            while self.is_running:
                # Get latest frame
                frame_data = self.stream_processor.get_latest_frame()
                
                if frame_data is None:
                    time.sleep(0.01)
                    continue
                
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                # Generate caption if needed
                caption_result = self.caption_generator.generate_caption(frame)
                
                # Display frame with caption
                self._display_frame(frame, timestamp)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 's' for style change
                    self._cycle_caption_style()
                elif key == ord('f'):  # 'f' for force caption
                    self.caption_generator.force_generate_caption(frame)
                elif key == ord('i'):  # 'i' for info
                    self._show_info()
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        
        finally:
            self.cleanup()
    
    def _display_frame(self, frame: np.ndarray, timestamp: float):
        """Display frame with caption overlay."""
        try:
            # Create display frame
            display_frame = frame.copy()
            
            # Add caption overlay
            if self.current_caption and (time.time() - self.caption_display_time) < self.caption_duration:
                self._add_caption_overlay(display_frame, self.current_caption)
            
            # Add status overlay
            self._add_status_overlay(display_frame, timestamp)
            
            # Display frame
            cv2.imshow('Streaming Instagram Captioner', display_frame)
            
        except Exception as e:
            logger.warning(f"Error displaying frame: {str(e)}")
    
    def _add_caption_overlay(self, frame: np.ndarray, caption: str):
        """Add caption text overlay to frame."""
        try:
            # Prepare text
            lines = self._wrap_text(caption, 50)
            
            # Calculate position
            y_start = 30
            line_height = 25
            
            # Add background rectangle
            text_height = len(lines) * line_height + 10
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, y_start + text_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, y_start + text_height), (255, 255, 255), 2)
            
            # Add text
            for i, line in enumerate(lines):
                y = y_start + (i * line_height)
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.warning(f"Error adding caption overlay: {str(e)}")
    
    def _add_status_overlay(self, frame: np.ndarray, timestamp: float):
        """Add status information overlay."""
        try:
            # Get stats
            stream_stats = self.stream_processor.get_stream_stats()
            buffer_stats = self.context_buffer.get_buffer_stats()
            caption_stats = self.caption_generator.get_performance_stats()
            
            # Prepare status text
            status_lines = [
                f"FPS: {stream_stats['current_fps']:.1f}",
                f"Frames: {self.frame_count}",
                f"Captions: {self.caption_count}",
                f"Context: {buffer_stats['frame_count']} frames",
                f"Style: {self.caption_style}",
                f"Runtime: {int(time.time() - self.start_time)}s"
            ]
            
            # Add status text
            y_start = frame.shape[0] - 150
            for i, line in enumerate(status_lines):
                y = y_start + (i * 20)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add controls
            controls = [
                "Controls:",
                "Q/ESC - Quit",
                "S - Change style",
                "F - Force caption",
                "I - Show info"
            ]
            
            x_start = frame.shape[1] - 200
            for i, line in enumerate(controls):
                y = 30 + (i * 20)
                cv2.putText(frame, line, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        except Exception as e:
            logger.warning(f"Error adding status overlay: {str(e)}")
    
    def _wrap_text(self, text: str, max_length: int) -> list:
        """Wrap text to fit within max_length characters per line."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines[:5]  # Limit to 5 lines max
    
    def _cycle_caption_style(self):
        """Cycle through available caption styles."""
        styles = ["instagram", "descriptive", "minimal", "trendy"]
        current_index = styles.index(self.caption_style) if self.caption_style in styles else 0
        next_index = (current_index + 1) % len(styles)
        
        self.caption_style = styles[next_index]
        self.caption_generator.set_caption_style(self.caption_style)
        
        logger.info(f"Caption style changed to: {self.caption_style}")
    
    def _show_info(self):
        """Show detailed information."""
        try:
            stream_stats = self.stream_processor.get_stream_stats()
            buffer_stats = self.context_buffer.get_buffer_stats()
            caption_stats = self.caption_generator.get_performance_stats()
            model_stats = self.model_manager.get_performance_stats()
            
            print("\n" + "="*50)
            print("STREAMING INSTAGRAM CAPTIONER - INFO")
            print("="*50)
            print(f"Runtime: {int(time.time() - self.start_time)}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Captions generated: {self.caption_count}")
            print(f"Current FPS: {stream_stats['current_fps']:.1f}")
            print(f"Context frames: {buffer_stats['frame_count']}")
            print(f"Context consistency: {buffer_stats['context_consistency']:.3f}")
            print(f"Avg caption time: {caption_stats['avg_generation_time']:.3f}s")
            print(f"Avg model time: {model_stats['avg_processing_time']:.3f}s")
            print(f"Current style: {self.caption_style}")
            print("="*50)
            
        except Exception as e:
            logger.warning(f"Error showing info: {str(e)}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        self.is_running = False
        
        if self.stream_processor:
            self.stream_processor.stop_stream()
        
        cv2.destroyAllWindows()
        
        logger.info("Cleanup completed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Streaming Instagram Captioner")
    parser.add_argument("--model-path", required=True, help="Path to LFM2-VL model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--style", default="instagram", choices=["instagram", "descriptive", "minimal", "trendy"], help="Caption style")
    parser.add_argument("--interval", type=float, default=3.0, help="Caption update interval in seconds")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Create and run application
    app = WebcamInstagramCaptioner(
        model_path=args.model_path,
        camera_index=args.camera,
        caption_style=args.style,
        update_interval=args.interval
    )
    
    try:
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()
