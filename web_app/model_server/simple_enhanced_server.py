#!/usr/bin/env python3
"""
Simple Enhanced Model Server - Basic version without video context integration
This provides the enhanced UI functionality with basic caption generation
"""

import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import base64
import io
import time
import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
model = None
processor = None
device = None

class CaptionRequest(BaseModel):
    image_data: str
    context: Optional[str] = None
    style: str = "instagram"
    prompt: Optional[str] = None

class VideoContextRequest(BaseModel):
    frames: List[str]  # List of base64 encoded images
    current_frame: str  # Current frame for captioning
    style: str = "instagram"
    prompt: Optional[str] = None

class CaptionResponse(BaseModel):
    caption: str
    instagram_caption: str
    raw_output: str
    context: str
    prompt: str
    confidence: float
    processing_time: float

class VideoContextResponse(BaseModel):
    caption: str
    instagram_caption: str
    raw_output: str
    video_context: str
    frame_descriptions: List[str]
    context_consistency: float
    confidence: float
    processing_time: float

def check_gpu_availability():
    """Check GPU availability and set device."""
    global device
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"   GPU {i}: {gpu_name}")
        
        # Try to use GPU 1 if available, otherwise GPU 0
        if gpu_count > 1:
            device = "cuda:1"
            torch.cuda.set_device(1)
            logger.info(f"Using GPU 1 for model inference")
        else:
            device = "cuda:0"
            torch.cuda.set_device(0)
            logger.info(f"Using GPU 0 for model inference")
    else:
        device = "cpu"
        logger.info("⚠️  CUDA not available, using CPU")
    
    return device

def load_enhanced_model():
    """Load the LFM2-VL model."""
    global model, processor, device
    
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        model_path = "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"
        
        logger.info("Loading LFM2-VL processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info("Loading LFM2-VL model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        logger.info(f"✅ LFM2-VL model loaded successfully")
        logger.info(f"Model device: {device}")
        
        # Warm up the model
        logger.info("Warming up model...")
        warmup_image = Image.new('RGB', (224, 224), color='white')
        warmup_text = "Describe this image"
        
        # Use the correct format for LFM2-VL processor
        inputs = processor(
            images=[warmup_image],  # Pass as list
            text=warmup_text,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        logger.info("✅ Model warmup completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model, processor, device
    
    logger.info("🚀 Starting Simple Enhanced Model Server...")
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Load model
    load_enhanced_model()
    
    logger.info("✅ Simple Enhanced Model Server ready!")
    
    yield
    
    logger.info("🛑 Shutting down Simple Enhanced Model Server...")

# Create FastAPI app
app = FastAPI(
    title="Simple Enhanced Streaming Instagram Captioner API",
    description="API for real-time Instagram caption generation with basic video context",
    version="4.0.0-simple",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_data: str) -> Image.Image:
    """Preprocess base64 image data."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def generate_caption_with_context(
    image: Image.Image,
    context: str = "",
    style: str = "instagram",
    prompt: str = None
) -> Dict:
    """Generate caption using LFM2-VL model with context."""
    global model, processor, device
    
    try:
        # Prepare the prompt
        if prompt is None:
            if style == "instagram":
                prompt = f"Create an Instagram-style caption for this image."
            else:
                prompt = f"Describe this image in detail."
        
        # Add context if provided
        if context:
            prompt = f"Based on the video context: '{context}', {prompt}"
        
        # Process inputs
        inputs = processor(
            images=[image],  # Pass as list
            text=prompt,
            return_tensors="pt"
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
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        processing_time = time.time() - start_time
        
        # Decode output
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract assistant response
        if "assistant" in generated_text.lower():
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                assistant_response = parts[-1].strip()
                # Clean up the response
                assistant_response = assistant_response.replace('"', '').strip()
            else:
                assistant_response = generated_text
        else:
            assistant_response = generated_text
        
        # Clean up the caption
        caption = clean_caption(assistant_response)
        
        return {
            "caption": caption,
            "instagram_caption": caption,
            "raw_output": generated_text,
            "context": context,
            "prompt": prompt,
            "confidence": 0.8,  # Placeholder
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {e}")

def clean_caption(text: str) -> str:
    """Clean up generated caption text."""
    # Remove common artifacts
    text = text.replace("<|reserved_5|>", "")
    text = text.replace("<|reserved_6|>", "")
    text = text.replace("<|reserved_7|>", "")
    text = text.replace("<|reserved_8|>", "")
    text = text.replace("<|reserved_9|>", "")
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove leading/trailing quotes
    text = text.strip('"\'')
    
    return text

def simple_aggregate_context(frame_descriptions: List[str]) -> str:
    """Simple context aggregation without video context modules."""
    if not frame_descriptions:
        return "No video context available"
    
    # Remove duplicates and combine
    unique_descriptions = list(set(frame_descriptions))
    if len(unique_descriptions) == 1:
        return unique_descriptions[0]
    
    return f"Video context from {len(frame_descriptions)} frames: {'; '.join(unique_descriptions)}"

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "device": device,
        "video_context_available": False,  # Simple version
        "timestamp": time.time()
    }

@app.post("/generate_caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    """Generate a caption for a single image."""
    try:
        # Preprocess image
        image = preprocess_image(request.image_data)
        
        # Generate caption
        result = generate_caption_with_context(
            image=image,
            context=request.context or "",
            style=request.style,
            prompt=request.prompt
        )
        
        return CaptionResponse(**result)
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_video_context_caption", response_model=VideoContextResponse)
async def generate_video_context_caption(request: VideoContextRequest):
    """Generate a caption using simple video context from multiple frames."""
    try:
        # Preprocess current frame
        current_image = preprocess_image(request.current_frame)
        
        # Preprocess context frames and generate descriptions
        frame_descriptions = []
        
        for frame_data in request.frames:
            try:
                frame_image = preprocess_image(frame_data)
                
                # Generate description for this frame
                frame_result = generate_caption_with_context(
                    image=frame_image,
                    context="",
                    style="descriptive",
                    prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition."
                )
                frame_descriptions.append(frame_result["caption"])
                
            except Exception as e:
                logger.warning(f"Failed to process frame: {e}")
                continue
        
        # Aggregate context from frame descriptions
        video_context = simple_aggregate_context(frame_descriptions)
        
        # Calculate context consistency
        context_consistency = min(1.0, len(frame_descriptions) / max(1, len(request.frames)))
        
        # Generate final caption with video context
        start_time = time.time()
        
        final_result = generate_caption_with_context(
            image=current_image,
            context=video_context,
            style=request.style,
            prompt=request.prompt
        )
        
        processing_time = time.time() - start_time
        
        return VideoContextResponse(
            caption=final_result["caption"],
            instagram_caption=final_result["instagram_caption"],
            raw_output=final_result["raw_output"],
            video_context=video_context,
            frame_descriptions=frame_descriptions,
            context_consistency=context_consistency,
            confidence=final_result["confidence"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Video context caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_context")
async def extract_context(request: CaptionRequest):
    """Extract context from a single frame (for compatibility)."""
    try:
        # Preprocess image
        image = preprocess_image(request.image_data)
        
        # Generate description
        result = generate_caption_with_context(
            image=image,
            context="",
            style="descriptive",
            prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition."
        )
        
        return {
            "context": result["caption"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
