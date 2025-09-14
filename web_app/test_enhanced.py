#!/usr/bin/env python3
"""
Test script for Enhanced Streaming Instagram Captioner
Tests the video context integration functionality
"""

import requests
import base64
import json
from PIL import Image
import io
import time

def create_test_image(color, text, size=(224, 224)):
    """Create a test image with specified color and text."""
    img = Image.new('RGB', size, color=color)
    
    # Add text to image
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_b64

def test_enhanced_server():
    """Test the enhanced model server."""
    print("🧪 Testing Enhanced Model Server...")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   📱 Device: {health_data.get('device', 'unknown')}")
            print(f"   🎬 Video context available: {health_data.get('video_context_available', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test basic caption generation
    print("\n2. Testing basic caption generation...")
    try:
        test_image = create_test_image('lightblue', 'Test Image')
        
        response = requests.post('http://localhost:8000/generate_caption', json={
            'image_data': test_image,
            'context': 'A test image for captioning',
            'style': 'instagram',
            'prompt': 'Create an Instagram-style caption for this image.'
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Basic caption generated")
            print(f"   📝 Caption: {data['instagram_caption']}")
            print(f"   ⏱️  Processing time: {data['processing_time']:.2f}s")
        else:
            print(f"   ❌ Basic caption failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Basic caption failed: {e}")
        return False
    
    # Test video context caption generation
    print("\n3. Testing video context caption generation...")
    try:
        # Create multiple test frames
        frames = [
            create_test_image('lightblue', 'Frame 1'),
            create_test_image('lightgreen', 'Frame 2'),
            create_test_image('lightcoral', 'Frame 3'),
            create_test_image('lightyellow', 'Frame 4'),
            create_test_image('lightpink', 'Frame 5')
        ]
        
        current_frame = create_test_image('lightsteelblue', 'Current Frame')
        
        response = requests.post('http://localhost:8000/generate_video_context_caption', json={
            'frames': frames,
            'current_frame': current_frame,
            'style': 'instagram',
            'prompt': 'Create an Instagram-style caption using the video context.'
        }, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Video context caption generated")
            print(f"   📝 Caption: {data['instagram_caption']}")
            print(f"   🎬 Video context: {data['video_context'][:100]}...")
            print(f"   📊 Context consistency: {data['context_consistency']:.2f}")
            print(f"   ⏱️  Processing time: {data['processing_time']:.2f}s")
            print(f"   📸 Frame descriptions: {len(data['frame_descriptions'])} frames processed")
        else:
            print(f"   ❌ Video context caption failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Video context caption failed: {e}")
        return False
    
    # Test context extraction
    print("\n4. Testing context extraction...")
    try:
        test_image = create_test_image('lightgray', 'Context Frame')
        
        response = requests.post('http://localhost:8000/extract_context', json={
            'image_data': test_image,
            'style': 'descriptive',
            'prompt': 'Describe this video frame in detail.'
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Context extracted")
            print(f"   📝 Context: {data['context']}")
            print(f"   ⏱️  Processing time: {data['processing_time']:.2f}s")
        else:
            print(f"   ❌ Context extraction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Context extraction failed: {e}")
        return False
    
    print("\n✅ All tests passed! Enhanced server is working correctly.")
    return True

def main():
    """Main test function."""
    print("🚀 Enhanced Streaming Instagram Captioner Test Suite")
    print("=" * 60)
    print("This script tests the enhanced model server with video context integration.")
    print("Make sure the enhanced model server is running on port 8000.")
    print("")
    
    # Wait a moment for user to read
    time.sleep(2)
    
    success = test_enhanced_server()
    
    if success:
        print("\n🎉 Enhanced server is ready for use!")
        print("🌐 Open your browser to: http://localhost:3000/enhanced_streaming_captioner.html")
    else:
        print("\n❌ Enhanced server tests failed!")
        print("Please check the server logs and try again.")

if __name__ == "__main__":
    main()
