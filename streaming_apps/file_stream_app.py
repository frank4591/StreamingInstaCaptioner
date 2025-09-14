#!/usr/bin/env python3
"""
File Stream Instagram Captioner

Process video files and generate Instagram captions in real-time using LFM2-VL models.
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

class FileStreamInstagramCaptioner:
    """
    Main application for file-based Instagram caption generation.
    """
    
    def __init__(
        self,
        model_path: str,
        video_path: str,
        caption_style: str = "instagram",
        update_interval: float = 2.0,
        playback_speed: float = 1.0
    ):
        """
        Initialize the file stream captioner.
        
        Args:
            model_path: Path to LFM2-VL model
            video_path: Path to input video file
            caption_style: Style of captions to generate
            update_interval: How often to generate new captions (seconds)
            playback_speed: Playback speed multiplier
        """
        self.model_path = model_path
        self.video_path = video_path
        self.caption_style = caption_style
        self.update_interval = update_interval
        self.playback_speed = playback_speed
        
        # Initialize components
        self.model_manager = None
        self.context_buffer = None
        self.stream_processor = None
        self.caption_generator = None
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.current_caption = ""
        self.caption_display_time = 0
        self.caption_duration = 3.0  # Show caption for 3 seconds
        
        # Performance tracking
        self.frame_count = 0
        self.caption_count = 0
        self.start_time = 0
        
        logger.info(f"FileStreamInstagramCaptioner initialized: {video_path}, style {caption_style}")
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing components...")
            
            # Check if video file exists
            if not Path(self.video_path).exists():
                logger.error(f"Video file does not exist: {self.video_path}")
                return False
            
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
                max_frames=20,
                context_window_seconds=45.0,
                update_interval=1.5,
                min_context_frames=3
            )
            
            # Initialize stream processor
            self.stream_processor = StreamingVideoProcessor(
                frame_callback=self._on_frame_received,
                target_fps=15,
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
                features=None,
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
        
        logger.info(f"Starting video file streaming: {self.video_path}")
        
        # Start file stream
        if not self.stream_processor.start_file_stream(self.video_path):
            logger.error("Failed to start file stream")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Main display loop
            while self.is_running:
                # Get latest frame
                frame_data = self.stream_processor.get_latest_frame()
                
                if frame_data is None:
                    # End of video or no frame available
                    if not self.is_paused:
                        logger.info("End of video file reached")
                        break
                    else:
                        time.sleep(0.01)
                        continue
                
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                # Generate caption if needed
                caption_result = self.caption_generator.generate_caption(frame)
                
                # Display frame with caption
                self._display_frame(frame, timestamp)
                
                # Control playback speed
                if self.playback_speed != 1.0:
                    time.sleep((1.0 / 30.0) / self.playback_speed)  # Adjust for playback speed
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # Spacebar for pause/play
                    self._toggle_pause()
                elif key == ord('s'):  # 's' for style change
                    self._cycle_caption_style()
                elif key == ord('f'):  # 'f' for force caption
                    self.caption_generator.force_generate_caption(frame)
                elif key == ord('i'):  # 'i' for info
                    self._show_info()
                elif key == ord('r'):  # 'r' for restart
                    self._restart_video()
        
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
            
            # Add pause indicator
            if self.is_paused:
                self._add_pause_overlay(display_frame)
            
            # Display frame
            cv2.imshow('File Stream Instagram Captioner', display_frame)
            
        except Exception as e:
            logger.warning(f"Error displaying frame: {str(e)}")
    
    def _add_caption_overlay(self, frame: np.ndarray, caption: str):
        """Add caption text overlay to frame."""
        try:
            # Prepare text
            lines = self._wrap_text(caption, 60)
            
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
            
            # Calculate video progress
            video_capture = self.stream_processor.capture
            if video_capture:
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                progress = (current_frame / total_frames * 100) if total_frames > 0 else 0
            else:
                progress = 0
            
            # Prepare status text
            status_lines = [
                f"File: {Path(self.video_path).name}",
                f"Progress: {progress:.1f}%",
                f"FPS: {stream_stats['current_fps']:.1f}",
                f"Frames: {self.frame_count}",
                f"Captions: {self.caption_count}",
                f"Context: {buffer_stats['frame_count']} frames",
                f"Style: {self.caption_style}",
                f"Speed: {self.playback_speed:.1f}x"
            ]
            
            # Add status text
            y_start = frame.shape[0] - 180
            for i, line in enumerate(status_lines):
                y = y_start + (i * 20)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add controls
            controls = [
                "Controls:",
                "Q/ESC - Quit",
                "SPACE - Pause/Play",
                "S - Change style",
                "F - Force caption",
                "R - Restart",
                "I - Show info"
            ]
            
            x_start = frame.shape[1] - 200
            for i, line in enumerate(controls):
                y = 30 + (i * 20)
                cv2.putText(frame, line, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        except Exception as e:
            logger.warning(f"Error adding status overlay: {str(e)}")
    
    def _add_pause_overlay(self, frame: np.ndarray):
        """Add pause indicator overlay."""
        try:
            # Add semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add pause text
            text = "PAUSED"
            font_scale = 2.0
            thickness = 3
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            
        except Exception as e:
            logger.warning(f"Error adding pause overlay: {str(e)}")
    
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
        
        return lines[:6]  # Limit to 6 lines max
    
    def _toggle_pause(self):
        """Toggle pause/play state."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "PLAYING"
        logger.info(f"Video {status}")
    
    def _cycle_caption_style(self):
        """Cycle through available caption styles."""
        styles = ["instagram", "descriptive", "minimal", "trendy"]
        current_index = styles.index(self.caption_style) if self.caption_style in styles else 0
        next_index = (current_index + 1) % len(styles)
        
        self.caption_style = styles[next_index]
        self.caption_generator.set_caption_style(self.caption_style)
        
        logger.info(f"Caption style changed to: {self.caption_style}")
    
    def _restart_video(self):
        """Restart video from beginning."""
        logger.info("Restarting video...")
        
        # Stop current stream
        self.stream_processor.stop_stream()
        
        # Restart stream
        if self.stream_processor.start_file_stream(self.video_path):
            self.frame_count = 0
            self.caption_count = 0
            self.context_buffer.clear_buffer()
            logger.info("Video restarted")
        else:
            logger.error("Failed to restart video")
    
    def _show_info(self):
        """Show detailed information."""
        try:
            stream_stats = self.stream_processor.get_stream_stats()
            buffer_stats = self.context_buffer.get_buffer_stats()
            caption_stats = self.caption_generator.get_performance_stats()
            model_stats = self.model_manager.get_performance_stats()
            
            print("\n" + "="*50)
            print("FILE STREAM INSTAGRAM CAPTIONER - INFO")
            print("="*50)
            print(f"Video: {self.video_path}")
            print(f"Runtime: {int(time.time() - self.start_time)}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Captions generated: {self.caption_count}")
            print(f"Current FPS: {stream_stats['current_fps']:.1f}")
            print(f"Context frames: {buffer_stats['frame_count']}")
            print(f"Context consistency: {buffer_stats['context_consistency']:.3f}")
            print(f"Avg caption time: {caption_stats['avg_generation_time']:.3f}s")
            print(f"Avg model time: {model_stats['avg_processing_time']:.3f}s")
            print(f"Current style: {self.caption_style}")
            print(f"Playback speed: {self.playback_speed:.1f}x")
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
    parser = argparse.ArgumentParser(description="File Stream Instagram Captioner")
    parser.add_argument("--model-path", required=True, help="Path to LFM2-VL model")
    parser.add_argument("--video-path", required=True, help="Path to input video file")
    parser.add_argument("--style", default="instagram", choices=["instagram", "descriptive", "minimal", "trendy"], help="Caption style")
    parser.add_argument("--interval", type=float, default=2.0, help="Caption update interval in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Check if video path exists
    if not Path(args.video_path).exists():
        logger.error(f"Video file does not exist: {args.video_path}")
        return
    
    # Create and run application
    app = FileStreamInstagramCaptioner(
        model_path=args.model_path,
        video_path=args.video_path,
        caption_style=args.style,
        update_interval=args.interval,
        playback_speed=args.speed
    )
    
    try:
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()
