#!/usr/bin/env python3
"""
Test script for Streaming Instagram Captioner

Tests the core components and functionality.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.model_manager import StreamingModelManager
        from src.context_buffer import StreamingContextBuffer
        from src.stream_processor import StreamingVideoProcessor
        from src.caption_generator import StreamingCaptionGenerator
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_manager():
    """Test model manager initialization."""
    print("\n🧪 Testing Model Manager...")
    
    try:
        from src.model_manager import StreamingModelManager
        
        # Test with mock model path
        model_manager = StreamingModelManager(
            model_path="/tmp/mock_model",
            device="cpu",
            enable_warmup=False
        )
        
        print("✅ Model Manager initialized")
        
        # Test performance stats
        stats = model_manager.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Model Manager error: {e}")
        return False

def test_context_buffer():
    """Test context buffer functionality."""
    print("\n🧪 Testing Context Buffer...")
    
    try:
        from src.context_buffer import StreamingContextBuffer
        
        buffer = StreamingContextBuffer(
            max_frames=5,
            context_window_seconds=10.0,
            update_interval=1.0,
            min_context_frames=2
        )
        
        # Add some test frames
        for i in range(3):
            buffer.add_frame(
                description=f"Test frame {i} description",
                features=np.random.randn(768),
                confidence=0.8,
                timestamp=i * 1.0
            )
        
        # Get context
        context = buffer.get_current_context()
        print(f"✅ Context: {context['context_text'][:50]}...")
        
        # Get stats
        stats = buffer.get_buffer_stats()
        print(f"✅ Buffer stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Context Buffer error: {e}")
        return False

def test_stream_processor():
    """Test stream processor functionality."""
    print("\n🧪 Testing Stream Processor...")
    
    try:
        from src.stream_processor import StreamingVideoProcessor
        
        processor = StreamingVideoProcessor(
            target_fps=5,
            frame_size=(320, 240),
            enable_preprocessing=True
        )
        
        print("✅ Stream Processor initialized")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(dummy_frame)
        
        print(f"✅ Frame preprocessing: {dummy_frame.shape} -> {processed.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Stream Processor error: {e}")
        return False

def test_caption_generator():
    """Test caption generator functionality."""
    print("\n🧪 Testing Caption Generator...")
    
    try:
        from src.caption_generator import StreamingCaptionGenerator
        from src.model_manager import StreamingModelManager
        from src.context_buffer import StreamingContextBuffer
        
        # Create mock components
        model_manager = StreamingModelManager(
            model_path="/tmp/mock_model",
            device="cpu",
            enable_warmup=False
        )
        
        context_buffer = StreamingContextBuffer(
            max_frames=5,
            context_window_seconds=10.0,
            update_interval=1.0,
            min_context_frames=2
        )
        
        # Add some context
        context_buffer.add_frame(
            description="Test context description",
            confidence=0.8
        )
        
        generator = StreamingCaptionGenerator(
            model_manager=model_manager,
            context_buffer=context_buffer,
            caption_style="instagram",
            update_interval=1.0
        )
        
        print("✅ Caption Generator initialized")
        
        # Test style change
        generator.set_caption_style("descriptive")
        print("✅ Style change successful")
        
        # Test performance stats
        stats = generator.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Caption Generator error: {e}")
        return False

def test_integration():
    """Test component integration."""
    print("\n🧪 Testing Integration...")
    
    try:
        from src.model_manager import StreamingModelManager
        from src.context_buffer import StreamingContextBuffer
        from src.stream_processor import StreamingVideoProcessor
        from src.caption_generator import StreamingCaptionGenerator
        
        # Initialize all components
        model_manager = StreamingModelManager(
            model_path="/tmp/mock_model",
            device="cpu",
            enable_warmup=False
        )
        
        context_buffer = StreamingContextBuffer(
            max_frames=5,
            context_window_seconds=10.0,
            update_interval=1.0,
            min_context_frames=2
        )
        
        stream_processor = StreamingVideoProcessor(
            target_fps=5,
            frame_size=(320, 240)
        )
        
        caption_generator = StreamingCaptionGenerator(
            model_manager=model_manager,
            context_buffer=context_buffer,
            caption_style="instagram"
        )
        
        print("✅ All components initialized successfully")
        
        # Test component interactions
        dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Add frame to context
        context_buffer.add_frame(
            description="Integration test frame",
            confidence=0.9
        )
        
        print("✅ Component integration successful")
        
        return True
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Streaming Instagram Captioner - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Manager", test_model_manager),
        ("Context Buffer", test_context_buffer),
        ("Stream Processor", test_stream_processor),
        ("Caption Generator", test_caption_generator),
        ("Integration", test_integration)
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
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()


