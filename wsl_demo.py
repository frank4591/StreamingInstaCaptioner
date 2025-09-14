#!/usr/bin/env python3
"""
WSL Demo for Streaming Instagram Captioner

Specialized demo for WSL (Windows Subsystem for Linux) users.
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Add src to path
sys.path.append('src')

def check_wsl_environment():
    """Check WSL environment and camera access."""
    print("🔍 Checking WSL Environment...")
    print("=" * 50)
    
    # Check if running on WSL
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower():
                print("✅ WSL detected")
                print(f"   Version: {version_info.strip()}")
            else:
                print("ℹ️  Not running on WSL")
    except:
        print("❌ Cannot determine WSL status")
    
    # Check camera access
    print("\n📷 Checking Camera Access...")
    try:
        from wsl_camera_handler import WSLCameraHandler
        
        handler = WSLCameraHandler()
        cameras = handler.get_available_cameras()
        
        if cameras:
            print(f"✅ Found {len(cameras)} camera(s): {cameras}")
            for cam in cameras:
                info = handler.get_camera_info(cam)
                print(f"   Camera {cam}: {info}")
        else:
            print("❌ No cameras detected")
            print("\n💡 WSL Camera Alternatives:")
            alternatives = handler.suggest_alternatives()
            for alt in alternatives:
                print(f"   {alt}")
            
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False
    
    return True

def run_file_stream_demo():
    """Run file stream demo (works well on WSL)."""
    print("\n📁 Running File Stream Demo...")
    print("=" * 50)
    
    # Check for sample video
    sample_videos = [
        "../VideoContextImageCaptioning/examples/sample_video.mp4",
        "test_video.mp4",
        "examples/sample_video.mp4"
    ]
    
    video_path = None
    for path in sample_videos:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("📹 Creating test video...")
        try:
            from wsl_camera_handler import WSLCameraHandler
            handler = WSLCameraHandler()
            video_path = handler.create_file_stream_demo()
            
            if not video_path:
                print("❌ Failed to create test video")
                return False
        except Exception as e:
            print(f"❌ Error creating test video: {e}")
            return False
    
    print(f"✅ Using video: {video_path}")
    
    # Run file stream app
    try:
        cmd = [
            sys.executable, "streaming_apps/file_stream_app.py",
            "--model-path", "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
            "--video-path", video_path,
            "--style", "instagram",
            "--interval", "2.0",
            "--speed", "1.0"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error running file stream demo: {e}")
        return False

def run_webcam_demo():
    """Run webcam demo (may not work on WSL)."""
    print("\n🎥 Running Webcam Demo...")
    print("=" * 50)
    print("⚠️  Note: Webcam access may not work on WSL")
    print("   If it fails, try the file stream demo instead.")
    print()
    
    input("Press Enter to try webcam demo...")
    
    try:
        cmd = [
            sys.executable, "streaming_apps/webcam_app.py",
            "--model-path", "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
            "--camera", "0",
            "--style", "instagram",
            "--interval", "3.0"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error running webcam demo: {e}")
        return False

def show_wsl_help():
    """Show WSL-specific help."""
    print("\n🆘 WSL Help & Troubleshooting")
    print("=" * 50)
    print()
    print("📋 WSL Camera Access Issues:")
    print("   • WSL doesn't have direct access to Windows cameras")
    print("   • Use WSLg (Windows 11) for better GUI support")
    print("   • Try running from Windows PowerShell with WSL integration")
    print("   • Install Windows camera drivers in WSL")
    print()
    print("🔧 Alternative Solutions:")
    print("   1. Use File Stream Demo (recommended for WSL)")
    print("   2. Create test videos for demonstration")
    print("   3. Use Windows Camera app to test camera access")
    print("   4. Try different camera indices (0, 1, 2, etc.)")
    print()
    print("💡 Best Practices for WSL:")
    print("   • Use file stream mode for reliable operation")
    print("   • Test with sample videos first")
    print("   • Check WSLg installation for GUI apps")
    print("   • Consider running from Windows for camera access")
    print()

def main():
    """Main WSL demo function."""
    print("🎥 Streaming Instagram Captioner - WSL Demo")
    print("=" * 60)
    print("Specialized demo for WSL (Windows Subsystem for Linux) users")
    print("=" * 60)
    
    # Check environment
    camera_available = check_wsl_environment()
    
    while True:
        print("\n" + "="*60)
        print("🎥 WSL DEMO MENU")
        print("="*60)
        print()
        print("1. 📁 File Stream Demo (Recommended for WSL)")
        print("   - Process video files")
        print("   - Works reliably on WSL")
        print("   - Generate captions with context")
        print()
        print("2. 🎥 Webcam Demo (May not work on WSL)")
        print("   - Real-time webcam streaming")
        print("   - Try if camera is available")
        print()
        print("3. 🆘 WSL Help & Troubleshooting")
        print("   - Camera access solutions")
        print("   - Alternative approaches")
        print()
        print("4. 🔧 Check System Again")
        print("   - Re-check camera access")
        print("   - Verify dependencies")
        print()
        print("0. Exit")
        print()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                run_file_stream_demo()
            elif choice == "2":
                if camera_available:
                    run_webcam_demo()
                else:
                    print("❌ No camera available. Try file stream demo instead.")
            elif choice == "3":
                show_wsl_help()
            elif choice == "4":
                camera_available = check_wsl_environment()
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


