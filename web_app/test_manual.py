#!/usr/bin/env python3
"""
Manual test to verify model server works
"""

import requests
import base64
import json
from PIL import Image
import io
import time

def create_test_image():
    """Create a simple test image."""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='lightblue')
    
    # Add some text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 100), "Test Image", fill='white', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_b64

def test_model_server():
    """Test the model server."""
    print("🧪 Manual Model Server Test")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model ready: {data.get('model_ready')}")
            print(f"   Device: {data.get('device')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        print("   Make sure to start the model server first:")
        print("   python model_server/model_server_fixed.py")
        return False
    
    # Test caption generation
    print("\n2. Testing caption generation...")
    try:
        test_image = create_test_image()
        response = requests.post("http://localhost:8000/generate_caption", 
                               json={
                                   "image_data": test_image,
                                   "context": "A beautiful blue sky",
                                   "style": "instagram",
                                   "prompt": "Create an Instagram-style caption for this image."
                               }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Caption generation passed")
            print(f"   Caption: {data.get('caption', '')}")
            print(f"   Instagram caption: {data.get('instagram_caption', '')}")
            print(f"   Processing time: {data.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Caption generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Caption generation error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_server()
    if success:
        print("\n🎉 Test passed! You can now use the web interface.")
        print("Open: http://localhost:3000/minimal_test.html")
    else:
        print("\n❌ Test failed. Check the model server logs.")
