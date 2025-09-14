#!/usr/bin/env python3
"""
Quick Start Script for Streaming Instagram Captioner

Easy setup and launch of the streaming application.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'cv2', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def check_model_path():
    """Check if LFM2-VL model is available."""
    print("\n🔍 Checking LFM2-VL model...")
    
    model_path = "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"
    
    if Path(model_path).exists():
        print(f"✅ Model found at: {model_path}")
        return model_path
    else:
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure the LFM2-VL model is available at the specified path.")
        return None

def check_camera():
    """Check if camera is available."""
    print("\n🔍 Checking camera...")
    
    try:
        import cv2
        
        # Check if running on WSL
        is_wsl = False
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
        except:
            pass
        
        if is_wsl:
            print("ℹ️  WSL detected - camera access may be limited")
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Camera available")
                cap.release()
                return True
            else:
                print("❌ Camera not accessible")
                cap.release()
                return False
        else:
            print("❌ Camera not available")
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def run_webcam_demo():
    """Run webcam demo."""
    print("\n🎥 Starting Webcam Demo...")
    
    model_path = check_model_path()
    if not model_path:
        return False
    
    if not check_camera():
        print("⚠️ Camera not available, but continuing...")
    
    try:
        # Run webcam app
        cmd = [
            sys.executable, "streaming_apps/webcam_app.py",
            "--model-path", model_path,
            "--camera", "0",
            "--style", "instagram",
            "--interval", "3.0"
        ]
        
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error running webcam demo: {e}")
        return False

def run_file_demo():
    """Run file stream demo."""
    print("\n📁 Starting File Stream Demo...")
    
    model_path = check_model_path()
    if not model_path:
        return False
    
    # Check for sample video
    sample_video = "../VideoContextImageCaptioning/examples/sample_video.mp4"
    if not Path(sample_video).exists():
        print(f"❌ Sample video not found: {sample_video}")
        print("Please provide a video file path.")
        video_path = input("Enter video file path: ").strip()
        if not Path(video_path).exists():
            print("❌ Video file not found")
            return False
    else:
        video_path = sample_video
        print(f"✅ Using sample video: {video_path}")
    
    try:
        # Run file stream app
        cmd = [
            sys.executable, "streaming_apps/file_stream_app.py",
            "--model-path", model_path,
            "--video-path", video_path,
            "--style", "instagram",
            "--interval", "2.0",
            "--speed", "1.0"
        ]
        
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error running file demo: {e}")
        return False

def show_menu():
    """Show main menu."""
    print("\n" + "="*60)
    print("🎥 STREAMING INSTAGRAM CAPTIONER - QUICK START")
    print("="*60)
    print()
    print("Choose a demo to run:")
    print()
    print("1. 🎥 Webcam Demo")
    print("   - Real-time webcam streaming")
    print("   - Live Instagram caption generation")
    print("   - Interactive controls")
    print()
    print("2. 📁 File Stream Demo")
    print("   - Process video files")
    print("   - Generate captions with context")
    print("   - Playback controls")
    print()
    print("3. 🏗️ Architecture Overview")
    print("   - Show system architecture")
    print("   - Explain components and data flow")
    print()
    print("4. 🌟 Use Cases")
    print("   - Show potential applications")
    print("   - Demonstrate LFM2-VL capabilities")
    print()
    print("5. 🔧 Check System")
    print("   - Verify dependencies and setup")
    print()
    print("0. Exit")
    print()

def main():
    """Main function."""
    print("🚀 Welcome to Streaming Instagram Captioner!")
    print("Demonstrating LFM2-VL 'constantly on' vision capabilities")
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                if check_dependencies():
                    run_webcam_demo()
                else:
                    print("❌ Please install missing dependencies first")
            elif choice == "2":
                if check_dependencies():
                    run_file_demo()
                else:
                    print("❌ Please install missing dependencies first")
            elif choice == "3":
                from demo import demo_architecture
                demo_architecture()
            elif choice == "4":
                from demo import demo_use_cases
                demo_use_cases()
            elif choice == "5":
                print("\n🔧 System Check")
                print("="*30)
                check_dependencies()
                check_model_path()
                check_camera()
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
