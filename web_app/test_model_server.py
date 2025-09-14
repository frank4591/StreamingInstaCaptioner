#!/usr/bin/env python3
"""
Test the fixed model server
"""

import requests
import base64
import json
from PIL import Image
import io

def create_test_image():
    """Create a test image."""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='lightblue')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_b64

def test_model_server():
    """Test the model server endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Fixed Model Server")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
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
        return False
    
    # Test context extraction
    print("\n2. Testing context extraction...")
    try:
        test_image = create_test_image()
        response = requests.post(f"{base_url}/extract_context", json={
            "image_data": test_image,
            "description": "Describe this image in detail."
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Context extraction passed")
            print(f"   Context: {data.get('context', '')[:100]}...")
            print(f"   Processing time: {data.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Context extraction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Context extraction error: {e}")
        return False
    
    # Test caption generation
    print("\n3. Testing caption generation...")
    try:
        test_image = create_test_image()
        response = requests.post(f"{base_url}/generate_caption", json={
            "image_data": test_image,
            "context": "A beautiful blue sky with clouds",
            "style": "instagram",
            "prompt": "Create an Instagram-style caption for this image."
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Caption generation passed")
            print(f"   Caption: {data.get('caption', '')}")
            print(f"   Instagram caption: {data.get('instagram_caption', '')}")
            print(f"   Processing time: {data.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Caption generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Caption generation error: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_model_server()


