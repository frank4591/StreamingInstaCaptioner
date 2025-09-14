#!/usr/bin/env python3
"""
Pipeline Integration Server - Uses existing VideoContextImageCaptioning pipeline
This server directly uses the working pipeline from VideoContextImageCaptioning project
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
import tempfile
import cv2
import numpy as np

# Add the VideoContextImageCaptioning src to path
video_context_path = Path(__file__).parent.parent.parent.parent / "VideoContextImageCaptioning" / "src"
if video_context_path.exists():
    sys.path.insert(0, str(video_context_path))
    try:
        # Import with absolute imports
        import pipeline
        import context_extractor
        import video_processor
        import liquid_integration
        VIDEO_CONTEXT_AVAILABLE = True
        logging.info("Video context modules loaded successfully")
    except ImportError as e:
        logging.warning(f"Video context modules not available: {e}")
        VIDEO_CONTEXT_AVAILABLE = False
else:
    logging.warning(f"VideoContextImageCaptioning project not found at {video_context_path}")
    VIDEO_CONTEXT_AVAILABLE = False

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
pipeline = None
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

def load_pipeline():
    """Load the VideoContextImageCaptioning pipeline."""
    global pipeline, device
    
    if not VIDEO_CONTEXT_AVAILABLE:
        logger.warning("Video context modules not available, using fallback mode")
        pipeline = None
        return
    
    try:
        model_path = "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"
        
        logger.info("Initializing VideoContextImageCaptioning pipeline...")
        pipeline = pipeline.VideoContextCaptionPipeline(
            model_path=model_path,
            device=device,
            frame_extraction_strategy="key_frames",
            max_frames=10,
            context_aggregation="weighted_average",
            context_weight=0.7
        )
        
        logger.info(f"✅ VideoContextImageCaptioning pipeline loaded successfully")
        logger.info(f"Pipeline device: {device}")
        
        # Test the pipeline with a simple image
        logger.info("Testing pipeline...")
        test_image = Image.new('RGB', (224, 224), color='white')
        test_result = pipeline.generate_caption(
            image=test_image,
            text_prompt="Create an Instagram-style caption for this image."
        )
        logger.info("✅ Pipeline test completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        logger.warning("Continuing with fallback mode")
        pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline, device
    
    logger.info("🚀 Starting Pipeline Integration Server...")
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Load pipeline
    load_pipeline()
    
    logger.info("✅ Pipeline Integration Server ready!")
    
    yield
    
    logger.info("🛑 Shutting down Pipeline Integration Server...")

# Create FastAPI app
app = FastAPI(
    title="Pipeline Integration Streaming Instagram Captioner API",
    description="API for real-time Instagram caption generation using VideoContextImageCaptioning pipeline",
    version="4.0.0-pipeline",
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

def create_temp_video(frames: List[Image.Image]) -> str:
    """Create a temporary video file from frames."""
    if not frames:
        raise ValueError("No frames provided")
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    # Get video dimensions from first frame
    height, width = frames[0].size[1], frames[0].size[0]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, 1.0, (width, height))
    
    try:
        for frame in frames:
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        
        out.release()
        return temp_video.name
        
    except Exception as e:
        out.release()
        os.unlink(temp_video.name)
        raise e

def generate_caption_with_pipeline(
    image: Image.Image,
    context: str = "",
    style: str = "instagram",
    prompt: str = None
) -> Dict:
    """Generate caption using the VideoContextImageCaptioning pipeline."""
    global pipeline
    
    try:
        # Prepare the prompt
        if prompt is None:
            if style == "instagram":
                prompt = "Create an Instagram-style caption for this image."
            else:
                prompt = "Describe this image in detail."
        
        # Add context if provided
        if context:
            prompt = f"Based on the video context: '{context}', {prompt}"
        
        # Generate caption using pipeline
        start_time = time.time()
        
        result = pipeline.generate_caption(
            image=image,
            text_prompt=prompt
        )
        
        processing_time = time.time() - start_time
        
        return {
            "caption": result.get("instagram_caption", result.get("caption", "No caption generated")),
            "instagram_caption": result.get("instagram_caption", result.get("caption", "No caption generated")),
            "raw_output": result.get("raw_output", ""),
            "context": context,
            "prompt": prompt,
            "confidence": result.get("confidence", 0.8),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Pipeline caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline caption generation failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "device": device,
        "video_context_available": VIDEO_CONTEXT_AVAILABLE,
        "timestamp": time.time()
    }

@app.post("/generate_caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    """Generate a caption for a single image."""
    try:
        # Preprocess image
        image = preprocess_image(request.image_data)
        
        # Generate caption
        result = generate_caption_with_pipeline(
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
    """Generate a caption using video context from multiple frames."""
    try:
        if not VIDEO_CONTEXT_AVAILABLE or pipeline is None:
            raise HTTPException(
                status_code=503, 
                detail="Video context pipeline not available"
            )
        
        # Preprocess current frame
        current_image = preprocess_image(request.current_frame)
        
        # Preprocess context frames
        context_frames = []
        frame_descriptions = []
        
        for frame_data in request.frames:
            try:
                frame_image = preprocess_image(frame_data)
                context_frames.append(frame_image)
                
                # Generate description for this frame using pipeline
                frame_result = generate_caption_with_pipeline(
                    image=frame_image,
                    context="",
                    style="descriptive",
                    prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition."
                )
                frame_descriptions.append(frame_result["caption"])
                
            except Exception as e:
                logger.warning(f"Failed to process frame: {e}")
                continue
        
        # Create temporary video from frames for context extraction
        temp_video_path = None
        video_context = "No video context available"
        
        try:
            if context_frames:
                temp_video_path = create_temp_video(context_frames)
                
                # Use pipeline to extract video context
                pipeline_result = pipeline.generate_caption(
                    image=current_image,
                    video_path=temp_video_path,
                    text_prompt=request.prompt or "Create an Instagram-style caption using the video context."
                )
                
                video_context = pipeline_result.get("context", "Video context extracted")
                
        except Exception as e:
            logger.warning(f"Video context extraction failed: {e}")
            # Fallback to simple aggregation
            if frame_descriptions:
                unique_descriptions = list(set(frame_descriptions))
                video_context = f"Video context from {len(frame_descriptions)} frames: {'; '.join(unique_descriptions)}"
        
        finally:
            # Clean up temporary video
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
        # Calculate context consistency
        context_consistency = min(1.0, len(frame_descriptions) / max(1, len(request.frames)))
        
        # Generate final caption with video context
        start_time = time.time()
        
        final_result = generate_caption_with_pipeline(
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
    """Extract context from a single frame."""
    try:
        # Preprocess image
        image = preprocess_image(request.image_data)
        
        # Generate description using pipeline
        result = generate_caption_with_pipeline(
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
