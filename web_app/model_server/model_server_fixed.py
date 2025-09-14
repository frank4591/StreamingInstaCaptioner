#!/usr/bin/env python3
"""
Fixed Model Server for Web-based Streaming Instagram Captioner
Based on debug_caption_extraction.py and instagram_caption_generator.py
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
import base64
import io
import time
from typing import Dict, Optional, List
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model components
processor = None
model = None
device = 'cpu'
context_buffer = []

# Pydantic models
class ImageData(BaseModel):
    image_data: str  # Base64 encoded image
    context: Optional[str] = None

class ContextRequest(BaseModel):
    image_data: str
    description: str

class CaptionRequest(BaseModel):
    image_data: str
    context: Optional[str] = None
    style: str = "instagram"
    prompt: Optional[str] = None

def check_gpu_availability():
    """Check GPU availability and provide detailed information."""
    logger.info("🔍 Checking GPU availability...")
    
    try:
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
        
        return np.array(image)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def clean_caption(raw_output: str) -> str:
    """Clean the generated caption by removing special tokens."""
    if not raw_output:
        return "A beautiful moment captured in time."
    
    # Remove special tokens
    cleaned = re.sub(r'<\|reserved_\d+\|>', '', raw_output)
    cleaned = re.sub(r'<\|endoftext\|>', '', cleaned)
    cleaned = re.sub(r'<\|im_start\|>', '', cleaned)
    cleaned = re.sub(r'<\|im_end\|>', '', cleaned)
    
    # Remove excessive whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # If too short, provide fallback
    if len(cleaned) < 10:
        return "A beautiful moment captured in time."
    
    return cleaned

def extract_instagram_caption(raw_output: str) -> str:
    """Extract the Instagram caption from the raw model output."""
    if not raw_output:
        return "A beautiful moment captured in time."
    
    # Look for the last "assistant" response in the output
    parts = re.split(r'assistant\s*', raw_output, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        # Get the last assistant response
        last_response = parts[-1].strip()
        
        # Look for content in quotes
        quote_match = re.search(r'["\']([^"\']*?)["\']', last_response)
        if quote_match:
            caption = quote_match.group(1).strip()
            return clean_caption(caption)
        
        # If no quotes, take everything after the last assistant
        caption = last_response.strip()
        if caption and len(caption) > 10:
            return clean_caption(caption)
    
    # Fallback: try to find content after the last quote
    quote_pattern = r'["\']([^"\']*?)["\']?\s*$'
    match = re.search(quote_pattern, raw_output)
    if match:
        caption = match.group(1).strip()
        return clean_caption(caption)
    
    # Final fallback: clean the entire output
    return clean_caption(raw_output)

async def initialize_models():
    """Initialize LFM2-VL model and components."""
    global processor, model, device
    
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
        
        # Load processor
        logger.info("Loading LFM2-VL processor...")
        processor = AutoProcessor.from_pretrained(
            '/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            trust_remote_code=True
        )
        
        # Load model
        logger.info("Loading LFM2-VL model...")
        model = AutoModelForImageTextToText.from_pretrained(
            '/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Move to device
        if device != 'cpu':
            model = model.to(device)
        
        logger.info("✅ LFM2-VL model loaded successfully")
        logger.info(f"Model device: {device}")
        
        # Warm up model
        logger.info("Warming up model...")
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_text = "Describe this image."
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": dummy_text},
                ],
            },
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(device)
        
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        logger.info("✅ Model warmup completed")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

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

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Streaming Instagram Captioner Model Server",
        "status": "running",
        "model_loaded": model is not None,
        "device": device
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "device": device,
        "timestamp": time.time()
    }

@app.post("/extract_context")
async def extract_context(request: ContextRequest):
    """Extract context from image frame."""
    try:
        if not model or not processor:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Decode image
        image_array = decode_image(request.image_data)
        image = Image.fromarray(image_array)
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.description},
                ],
            },
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(device)
        
        # Generate description
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                min_p=0.15,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode output
        raw_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        caption = clean_caption(raw_output)
        
        processing_time = time.time() - start_time
        
        # Add to context buffer
        context_buffer.append({
            "description": caption,
            "timestamp": time.time(),
            "confidence": 0.8  # Default confidence
        })
        
        # Keep only last 20 frames
        if len(context_buffer) > 20:
            context_buffer.pop(0)
        
        # Get current context
        context_text = " ".join([frame["description"] for frame in context_buffer[-5:]])
        
        return {
            "context": context_text,
            "confidence": 0.8,
            "processing_time": processing_time,
            "context_consistency": 0.7,  # Default consistency
            "frame_count": len(context_buffer)
        }
        
    except Exception as e:
        logger.error(f"Error extracting context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_caption")
async def generate_caption(request: CaptionRequest):
    """Generate Instagram caption with context."""
    try:
        if not model or not processor:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Decode image
        image_array = decode_image(request.image_data)
        image = Image.fromarray(image_array)
        
        # Prepare prompt
        if request.context:
            prompt = f"Based on the video context: '{request.context}', Create an instagram style caption for this image."
        else:
            prompt = "Create an instagram style caption for this image."
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(device)
        
        # Generate caption
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                min_p=0.15,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode output
        raw_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        instagram_caption = extract_instagram_caption(raw_output)
        
        processing_time = time.time() - start_time
        
        return {
            "caption": instagram_caption,
            "raw_output": raw_output,
            "context": request.context or "",
            "prompt": prompt,
            "instagram_caption": instagram_caption,
            "confidence": 0.8,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get model statistics."""
    try:
        return {
            "model_loaded": model is not None,
            "device": device,
            "context_frames": len(context_buffer),
            "uptime": time.time() - start_time if 'start_time' in globals() else 0
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main function
if __name__ == "__main__":
    start_time = time.time()
    logger.info("🚀 Starting Model Server...")
    uvicorn.run(
        "model_server_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


