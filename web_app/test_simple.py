#!/usr/bin/env python3
import requests
import base64
from PIL import Image
import io

def test_api():
    print("Testing backend API...")
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_data = base64.b64encode(buffer.getvalue()).decode()
    
    # Test health endpoint
    try:
        health_response = requests.get('http://localhost:8000/health')
        print(f"Health check: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health data: {health_response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test analyze_frames endpoint
    try:
        response = requests.post('http://localhost:8000/analyze_frames', 
                               json={'frames': [f'data:image/jpeg;base64,{img_data}']})
        print(f"Analyze frames status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API is working!")
            print(f"Frame descriptions: {data.get('frame_descriptions', [])}")
            print(f"Aggregated context: {data.get('aggregated_context', 'N/A')}")
            return True
        else:
            print(f"❌ API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API error: {e}")
        return False

if __name__ == "__main__":
    test_api()
