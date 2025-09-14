#!/usr/bin/env python3
"""
Demo script for Streaming Instagram Captioner

Demonstrates the real-time streaming capabilities with LFM2-VL models.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

def demo_webcam():
    """Run webcam demo."""
    print("🎥 Starting Webcam Demo")
    print("=" * 50)
    print("This demo will:")
    print("- Start your webcam")
    print("- Extract context from video frames")
    print("- Generate Instagram captions in real-time")
    print("- Display captions overlaid on video")
    print("\nControls:")
    print("- Q/ESC: Quit")
    print("- S: Change caption style")
    print("- F: Force generate caption")
    print("- I: Show info")
    print("=" * 50)
    
    input("Press Enter to start webcam demo...")
    
    # Import and run webcam app
    from streaming_apps.webcam_app import main as webcam_main
    
    # Set up arguments
    sys.argv = [
        "webcam_app.py",
        "--model-path", "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
        "--camera", "0",
        "--style", "instagram",
        "--interval", "3.0"
    ]
    
    webcam_main()

def demo_file_stream():
    """Run file stream demo."""
    print("📁 Starting File Stream Demo")
    print("=" * 50)
    print("This demo will:")
    print("- Process a video file")
    print("- Extract context from video frames")
    print("- Generate Instagram captions in real-time")
    print("- Display captions overlaid on video")
    print("\nControls:")
    print("- Q/ESC: Quit")
    print("- SPACE: Pause/Play")
    print("- S: Change caption style")
    print("- F: Force generate caption")
    print("- R: Restart video")
    print("- I: Show info")
    print("=" * 50)
    
    # Get video file path
    video_path = input("Enter path to video file (or press Enter for default): ").strip()
    if not video_path:
        video_path = "../VideoContextImageCaptioning/examples/sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("Please provide a valid video file path.")
        return
    
    input("Press Enter to start file stream demo...")
    
    # Import and run file stream app
    from streaming_apps.file_stream_app import main as file_main
    
    # Set up arguments
    sys.argv = [
        "file_stream_app.py",
        "--model-path", "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
        "--video-path", video_path,
        "--style", "instagram",
        "--interval", "2.0",
        "--speed", "1.0"
    ]
    
    file_main()

def demo_architecture():
    """Show architecture overview."""
    print("🏗️ Streaming Instagram Captioner Architecture")
    print("=" * 60)
    print()
    print("📊 COMPONENTS:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 1. Model Manager (LFM2-VL)                                 │")
    print("│    - Loads and manages LFM2-VL model                      │")
    print("│    - Handles real-time inference                          │")
    print("│    - Optimized for streaming performance                   │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 2. Stream Processor (OpenCV)                               │")
    print("│    - Captures video from webcam or file                    │")
    print("│    - Preprocesses frames for better quality                │")
    print("│    - Manages frame rate and buffering                      │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 3. Context Buffer (Sliding Window)                        │")
    print("│    - Maintains recent frame descriptions                   │")
    print("│    - Aggregates context over time window                   │")
    print("│    - Provides temporal consistency                        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 4. Caption Generator (Real-time)                          │")
    print("│    - Generates Instagram captions with context            │")
    print("│    - Supports multiple caption styles                     │")
    print("│    - Manages caption timing and display                    │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("🔄 DATA FLOW:")
    print("Video Stream → Frame Processing → Context Extraction → Caption Generation")
    print("     ↓              ↓                    ↓                    ↓")
    print("  Webcam/File → Preprocessing → Context Buffer → Instagram Caption")
    print()
    print("🎯 KEY FEATURES:")
    print("• Real-time processing with low latency")
    print("• Privacy-preserving on-device processing")
    print("• Multiple caption styles (Instagram, descriptive, minimal, trendy)")
    print("• Configurable update intervals and context windows")
    print("• Performance monitoring and statistics")
    print("• Support for webcam and file-based streaming")
    print()

def demo_use_cases():
    """Show use cases and applications."""
    print("🌟 Use Cases & Applications")
    print("=" * 50)
    print()
    print("📱 LIVE INSTAGRAM CONTENT CREATION:")
    print("• Real-time caption generation for live streams")
    print("• Automatic hashtag and emoji addition")
    print("• Multiple caption styles for different content types")
    print()
    print("🔒 SMART SECURITY CAMERAS:")
    print("• Contextual descriptions of security footage")
    print("• Real-time scene understanding")
    print("• Privacy-preserving on-device processing")
    print()
    print("♿ ACCESSIBILITY TOOLS:")
    print("• Live scene descriptions for visually impaired users")
    print("• Real-time environmental awareness")
    print("• Continuous context understanding")
    print()
    print("🗺️ AI TOUR GUIDES:")
    print("• Live commentary on surroundings")
    print("• Contextual information about locations")
    print("• Real-time translation and description")
    print()
    print("🎥 CONTENT CREATION:")
    print("• Automated video captioning")
    print("• Real-time content analysis")
    print("• Social media content generation")
    print()
    print("🔬 RESEARCH & DEVELOPMENT:")
    print("• LFM2-VL model evaluation")
    print("• Real-time vision-language research")
    print("• Edge AI performance testing")
    print()

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Streaming Instagram Captioner Demo")
    parser.add_argument("--demo", choices=["webcam", "file", "architecture", "use-cases"], 
                       default="webcam", help="Demo to run")
    
    args = parser.parse_args()
    
    print("🎥 Streaming Instagram Captioner Demo")
    print("=" * 50)
    print("Demonstrating LFM2-VL 'constantly on' vision capabilities")
    print("=" * 50)
    print()
    
    if args.demo == "webcam":
        demo_webcam()
    elif args.demo == "file":
        demo_file_stream()
    elif args.demo == "architecture":
        demo_architecture()
    elif args.demo == "use-cases":
        demo_use_cases()
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main()


