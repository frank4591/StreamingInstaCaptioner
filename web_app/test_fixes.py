#!/usr/bin/env python3
"""
Test script to verify the web app fixes
"""

import sys
import os
import subprocess
import time
import requests
import json

def test_node_dependencies():
    """Test if Node.js dependencies are installed."""
    print("🧪 Testing Node.js dependencies...")
    
    try:
        # Check if node_modules exists
        if os.path.exists('node_modules'):
            print("✅ node_modules directory found")
            
            # Check specific packages
            packages = ['express', 'socket.io', 'cors', 'multer', 'axios']
            for package in packages:
                if os.path.exists(f'node_modules/{package}'):
                    print(f"✅ {package} installed")
                else:
                    print(f"❌ {package} not found")
        else:
            print("❌ node_modules directory not found")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

def test_python_dependencies():
    """Test if Python dependencies are installed."""
    print("\n🧪 Testing Python dependencies...")
    
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn installed")
        return True
    except ImportError as e:
        print(f"❌ Python dependencies missing: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("\n🧪 Testing GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
            
            # Test GPU 1 specifically
            if gpu_count > 1:
                print("✅ GPU 1 is available")
                return True
            else:
                print("⚠️  Only 1 GPU available, will use GPU 0")
                return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def test_model_server():
    """Test if model server can start."""
    print("\n🧪 Testing Model Server...")
    
    try:
        # Start model server in background
        process = subprocess.Popen([
            'python', 'model_server/model_server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(10)
        
        # Check if server is running
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print("✅ Model server is running")
                data = response.json()
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   GPU Device: {data.get('gpu_device', 'unknown')}")
                return True
            else:
                print(f"❌ Model server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException:
            print("❌ Model server not responding")
            return False
        finally:
            # Kill the process
            process.terminate()
            process.wait()
            
    except Exception as e:
        print(f"❌ Error testing model server: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Web App Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        ("Node.js Dependencies", test_node_dependencies),
        ("Python Dependencies", test_python_dependencies),
        ("GPU Availability", test_gpu_availability),
        ("Model Server", test_model_server)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web app should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()


