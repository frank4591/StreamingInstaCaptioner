#!/usr/bin/env python3
"""
Model Server for Web-based Streaming Instagram Captioner

FastAPI server that runs LFM2-VL model on GPU 1 for web application.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
import base64
import io
import time
from typing import Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import torch

# Import our streaming components
from model_manager import StreamingModelManager
from context_buffer import StreamingContextBuffer
from caption_generator import StreamingCaptionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lifespan event handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    await initialize_models()
    yield
    # Cleanup on shutdown if needed

# FastAPI app
app = FastAPI(
    title="Streaming Instagram Captioner Model Server",
    description="LFM2-VL model server for real-time caption generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model components
model_manager = None
context_buffer = None
caption_generator = None

def check_gpu_availability():
    """Check GPU availability and provide detailed information."""
    logger.info("🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            logger.warning("❌ CUDA not available")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False

# Pydantic models
class ImageData(BaseModel):
    image_data: str  # Base64 encoded image
    context: Optional[str] = None
    style: Optional[str] = "instagram"
    prompt: Optional[str] = None

class ContextRequest(BaseModel):
    image_data: str  # Base64 encoded image
    description: str

class CaptionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    context: Optional[str] = None
    style: Optional[str] = "instagram"
    prompt: Optional[str] = None

# Utility functions
def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def encode_image(image_array: np.ndarray) -> str:
    """Encode numpy array to base64 image data."""
    try:
        # Convert to PIL Image
        image = Image.fromarray(image_array.astype(np.uint8))
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_data}"
        
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to encode image")

# Initialize model components
async def initialize_models():
    """Initialize LFM2-VL model and components."""
    global model_manager, context_buffer, caption_generator
    
    try:
        logger.info("Initializing LFM2-VL model...")
        
        # Check GPU availability first
        check_gpu_availability()
        
        # Check GPU availability and set device accordingly
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            
            if gpu_count > 1:
                # Use GPU 1 if available
                os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                torch.cuda.set_device(1)
                device = 'cuda:1'
                logger.info("Using GPU 1 for model inference")
            else:
                # Use GPU 0 if only one GPU available
                device = 'cuda:0'
                logger.info("Using GPU 0 for model inference")
        else:
            # Fallback to CPU
            device = 'cpu'
            logger.warning("CUDA not available, using CPU for model inference")
        
        # Initialize model manager
        model_manager = StreamingModelManager(
            model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            device=device,
            enable_warmup=True
        )
        
        if not model_manager.load_model():
            raise Exception("Failed to load LFM2-VL model")
        
        # Initialize context buffer
        context_buffer = StreamingContextBuffer(
            max_frames=20,
            context_window_seconds=45.0,
            update_interval=2.0,
            min_context_frames=3
        )
        
        # Initialize caption generator
        caption_generator = StreamingCaptionGenerator(
            model_manager=model_manager,
            context_buffer=context_buffer,
            caption_style="instagram",
            update_interval=3.0
        )
        
        logger.info("✅ All model components initialized successfully")
        logger.info(f"Model device: {model_manager.model.device if model_manager.model else 'Unknown'}")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Streaming Instagram Captioner Model Server",
        "status": "running",
        "model_loaded": model_manager is not None and model_manager.is_ready(),
        "gpu_device": os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_ready": model_manager is not None and model_manager.is_ready(),
        "context_buffer_ready": context_buffer is not None,
        "caption_generator_ready": caption_generator is not None,
        "gpu_device": os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        "timestamp": time.time()
    }

@app.post("/extract_context")
async def extract_context(request: ContextRequest):
    """Extract context from image frame."""
    try:
        if not model_manager or not model_manager.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Decode image
        image_array = decode_image(request.image_data)
        
        # Generate frame description
        result = model_manager.generate_caption(
            image=image_array,
            text_prompt=request.description,
            temperature=0.1,
            max_new_tokens=100
        )
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Add to context buffer
        context_buffer.add_frame(
            description=result["caption"],
            features=None,
            confidence=result["confidence"],
            timestamp=time.time()
        )
        
        # Get current context
        context_info = context_buffer.get_current_context()
        
        return {
            "context": context_info["context_text"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "context_consistency": context_info["consistency"],
            "frame_count": context_info["frame_count"]
        }
        
    except Exception as e:
        logger.error(f"Error extracting context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_caption")
async def generate_caption(request: CaptionRequest):
    """Generate Instagram caption with context."""
    try:
        if not model_manager or not model_manager.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Decode image
        image_array = decode_image(request.image_data)
        
        # Generate caption
        result = model_manager.generate_caption(
            image=image_array,
            text_prompt=request.prompt or f"Create an {request.style}-style caption for this image.",
            temperature=0.1,
            max_new_tokens=150
        )
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Process caption based on style
        processed_caption = process_caption_for_style(result["caption"], request.style)
        
        return {
            "caption": processed_caption,
            "raw_output": result["raw_output"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "style": request.style,
            "context": request.context or "No context available"
        }
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_caption_for_style(caption: str, style: str) -> str:
    """Process caption based on style."""
    if not caption:
        return "A beautiful moment captured in time."
    
    # Clean up the caption
    caption = clean_caption(caption)
    
    # Apply style-specific formatting
    if style == "instagram":
        caption = add_instagram_formatting(caption)
    elif style == "trendy":
        caption = add_trendy_formatting(caption)
    elif style == "minimal":
        caption = add_minimal_formatting(caption)
    elif style == "descriptive":
        caption = add_descriptive_formatting(caption)
    
    return caption

def clean_caption(caption: str) -> str:
    """Clean up caption text."""
    if not caption:
        return "A beautiful moment captured in time."
    
    # Remove common artifacts
    artifacts = ['<|reserved_', '<|endoftext|>', '<|startoftext|>', '<pad>', '<unk>']
    for artifact in artifacts:
        caption = caption.replace(artifact, '')
    
    # Clean whitespace
    import re
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # Remove quotes if present
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1]
    elif caption.startswith("'") and caption.endswith("'"):
        caption = caption[1:-1]
    
    return caption if caption else "A beautiful moment captured in time."

def add_instagram_formatting(caption: str) -> str:
    """Add Instagram-style formatting."""
    # Add relevant hashtags
    hashtags = generate_hashtags(caption)
    if hashtags:
        return f"{caption}\n\n{hashtags}"
    return caption

def add_trendy_formatting(caption: str) -> str:
    """Add trendy formatting."""
    hashtags = generate_hashtags(caption)
    if hashtags:
        return f"{caption}\n\n{hashtags}"
    return caption

def add_minimal_formatting(caption: str) -> str:
    """Add minimal formatting."""
    # Keep it simple and clean
    return caption

def add_descriptive_formatting(caption: str) -> str:
    """Add descriptive formatting."""
    # Keep detailed and informative
    return caption

def generate_hashtags(caption: str) -> str:
    """Generate relevant hashtags."""
    hashtags = []
    
    content_lower = caption.lower()
    
    # Common hashtags based on content
    if any(word in content_lower for word in ['sunset', 'sunrise', 'golden hour']):
        hashtags.extend(['#sunset', '#goldenhour', '#nature'])
    
    if any(word in content_lower for word in ['beach', 'ocean', 'water', 'wave']):
        hashtags.extend(['#beach', '#ocean', '#water', '#coastal'])
    
    if any(word in content_lower for word in ['mountain', 'hill', 'peak', 'summit']):
        hashtags.extend(['#mountain', '#nature', '#landscape'])
    
    if any(word in content_lower for word in ['city', 'urban', 'street', 'building']):
        hashtags.extend(['#city', '#urban', '#street'])
    
    if any(word in content_lower for word in ['food', 'meal', 'restaurant', 'cooking']):
        hashtags.extend(['#food', '#foodie', '#delicious'])
    
    # Add generic hashtags
    hashtags.extend(['#photography', '#instagood', '#picoftheday'])
    
    # Limit number of hashtags
    hashtags = hashtags[:8]
    
    return ' '.join(hashtags)

@app.get("/stats")
async def get_stats():
    """Get model and system statistics."""
    try:
        stats = {
            "model_stats": model_manager.get_performance_stats() if model_manager else {},
            "context_stats": context_buffer.get_buffer_stats() if context_buffer else {},
            "caption_stats": caption_generator.get_performance_stats() if caption_generator else {},
            "gpu_device": os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
            "timestamp": time.time()
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Main function
if __name__ == "__main__":
    logger.info("🚀 Starting Model Server on GPU 1...")
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
