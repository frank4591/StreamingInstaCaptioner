#!/usr/bin/env python3
"""
Test GPU availability in WSL environment
"""

import sys
import os

def test_gpu():
    """Test GPU availability and provide detailed information."""
    print("🔍 Testing GPU Availability in WSL")
    print("=" * 50)
    
    # Check if we're in WSL
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            if 'microsoft' in version.lower() or 'wsl' in version.lower():
                print("✅ Running in WSL environment")
            else:
                print("ℹ️  Not running in WSL")
    except:
        print("ℹ️  Could not detect WSL environment")
    
    # Check CUDA availability
    try:
        import torch
        print(f"\n🐍 PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test device selection
            print(f"\n🧪 Testing device selection:")
            try:
                torch.cuda.set_device(0)
                print("✅ GPU 0: Available")
            except Exception as e:
                print(f"❌ GPU 0: Error - {e}")
            
            if gpu_count > 1:
                try:
                    torch.cuda.set_device(1)
                    print("✅ GPU 1: Available")
                except Exception as e:
                    print(f"❌ GPU 1: Error - {e}")
            
            return True
        else:
            print("❌ CUDA not available")
            print("   This could be due to:")
            print("   - NVIDIA drivers not installed in WSL")
            print("   - CUDA toolkit not installed")
            print("   - WSL2 not configured for GPU passthrough")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def test_nvidia_smi():
    """Test nvidia-smi command."""
    print(f"\n🔧 Testing nvidia-smi command:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            print("GPU Information:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")

def main():
    """Main test function."""
    print("🧪 GPU Availability Test for WSL")
    print("=" * 50)
    
    # Test nvidia-smi
    test_nvidia_smi()
    
    # Test PyTorch CUDA
    gpu_available = test_gpu()
    
    print(f"\n📊 SUMMARY:")
    if gpu_available:
        print("✅ GPU is available - Model can run on GPU")
        print("💡 Recommendation: Use GPU for better performance")
    else:
        print("⚠️  GPU not available - Model will run on CPU")
        print("💡 Recommendation: Install NVIDIA drivers and CUDA for WSL")
        print("   - Install NVIDIA drivers on Windows")
        print("   - Install CUDA toolkit in WSL")
        print("   - Configure WSL2 for GPU passthrough")
    
    print("=" * 50)

if __name__ == "__main__":
    main()


